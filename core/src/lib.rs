//! Cémantix solver core: vectors, exact-score constraints, entropy-based guess choice.
//! No I/O and no network, so it runs natively (feature `parallel` = rayon) and in wasm.

use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Words beyond this frequency rank are not considered plausible secrets.
pub const PRIOR_RANK: usize = 50_000;
pub const MAX_PROBES: usize = 2000;
pub const MAX_TARGETS: usize = 4000;
/// Tolerance ladder on |cos − score|, in units of the game's rounding step
/// (Cémantix rounds the cosine to 1e-4, QuelMot to 1e-3).
pub const TOL_MULT: [f64; 5] = [1.0, 2.0, 5.0, 10.0, 30.0];
/// Cémantix: score = round(cos × 10 000) / 10 000.
pub const SCALE_CEMANTIX: f64 = 10_000.0;
/// QuelMot: score = round(cos × 1 000), an integer between -1000 and 1000.
pub const SCALE_QUELMOT: f64 = 1_000.0;

/// Smallest tolerance level whose tolerance covers half a rounding step plus the
/// model's own score error (e.g. the float16 compression error).
pub fn tol_level_for(scale: f64, model_error: f64) -> usize {
    let needed = 0.5 + model_error * scale;
    TOL_MULT
        .iter()
        .position(|&m| m >= needed)
        .unwrap_or(TOL_MULT.len() - 1)
}

/// SIMD-friendly dot product (8 independent accumulators + scalar tail).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut tail = 0f32;
    for (x, y) in ra.iter().zip(rb) {
        tail += x * y;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail
}

#[inline]
pub fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

/// Round a cosine the way a game does (`scale` = 10 000 for Cémantix, 1 000 for QuelMot).
#[inline]
pub fn round_scale(x: f64, scale: f64) -> f64 {
    (x * scale).round() / scale
}

pub fn is_plausible_word(w: &str) -> bool {
    w.chars().count() >= 3
        && !w.starts_with('-')
        && !w.ends_with('-')
        && w.chars()
            .all(|c| (c.is_alphabetic() && c.is_lowercase()) || c == '-')
}

/// IEEE 754 half precision helpers (used for the compact web model).
pub mod f16 {
    pub fn to_f32(h: u16) -> f32 {
        let sign = ((h >> 15) & 1) as u32;
        let exp = ((h >> 10) & 0x1f) as u32;
        let frac = (h & 0x3ff) as u32;
        let bits = if exp == 0 {
            if frac == 0 {
                sign << 31
            } else {
                let mut e: i32 = -14;
                let mut f = frac;
                while f & 0x400 == 0 {
                    f <<= 1;
                    e -= 1;
                }
                (sign << 31) | (((e + 127) as u32) << 23) | ((f & 0x3ff) << 13)
            }
        } else if exp == 31 {
            (sign << 31) | 0x7f80_0000 | (frac << 13)
        } else {
            (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
        };
        f32::from_bits(bits)
    }

    /// Round-to-nearest-even conversion.
    pub fn from_f32(x: f32) -> u16 {
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32;
        let frac = b & 0x7f_ffff;
        if exp == 0xff {
            return sign | 0x7c00 | if frac != 0 { 0x200 } else { 0 };
        }
        let e = exp - 127 + 15;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
        if e <= 0 {
            if e < -10 {
                return sign;
            }
            let m = frac | 0x80_0000;
            let shift = (14 - e) as u32;
            let mut half = (m >> shift) as u16;
            let rem = m & ((1u32 << shift) - 1);
            let halfway = 1u32 << (shift - 1);
            if rem > halfway || (rem == halfway && (half & 1) == 1) {
                half += 1;
            }
            return sign | half;
        }
        let mut half = (((e as u32) << 10) | (frac >> 13)) as u16;
        let rem = frac & 0x1fff;
        if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
            half += 1;
        }
        sign | half
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn roundtrip_is_accurate() {
            for i in -20000..20000 {
                let x = i as f32 / 20000.0 * 0.4;
                let y = to_f32(from_f32(x));
                assert!((x - y).abs() <= x.abs() * 1e-3 + 1e-7, "{x} -> {y}");
            }
            assert_eq!(to_f32(from_f32(0.0)), 0.0);
            assert_eq!(to_f32(from_f32(1.0)), 1.0);
            assert_eq!(to_f32(from_f32(-0.5)), -0.5);
        }
    }
}

/// Entropy (bits) of the rounded-score partition induced by probe `p` over `targets`,
/// with scores rounded at `scale`.
pub fn partition_entropy(
    vecs: &[f32],
    dim: usize,
    p: usize,
    targets: &[u32],
    scale: f64,
) -> (f64, usize) {
    let q = &vecs[p * dim..(p + 1) * dim];
    let mut keys: Vec<i32> = targets
        .iter()
        .map(|&t| {
            let t = t as usize;
            (dot(&vecs[t * dim..(t + 1) * dim], q) as f64 * scale).round() as i32
        })
        .collect();
    keys.sort_unstable();
    let total = keys.len() as f64;
    let mut h = 0f64;
    let mut buckets = 0usize;
    let mut i = 0;
    while i < keys.len() {
        let mut j = i + 1;
        while j < keys.len() && keys[j] == keys[i] {
            j += 1;
        }
        let pr = (j - i) as f64 / total;
        h -= pr * pr.log2();
        buckets += 1;
        i = j;
    }
    (h, buckets)
}

/// Normalised word vectors plus the plausible-secret mask.
pub struct Vectors {
    pub name: String,
    pub dim: usize,
    pub words: Vec<String>,
    /// n * dim, L2-normalised rows.
    pub vecs: Vec<f32>,
    pub index: HashMap<String, u32>,
    pub plausible: Vec<bool>,
    pub opener: Option<u32>,
}

impl Vectors {
    pub fn new(
        name: String,
        dim: usize,
        words: Vec<String>,
        vecs: Vec<f32>,
        prior_rank: usize,
    ) -> Self {
        assert_eq!(vecs.len(), words.len() * dim, "vector buffer size mismatch");
        let mut index = HashMap::with_capacity(words.len());
        for (i, w) in words.iter().enumerate() {
            index.entry(w.clone()).or_insert(i as u32);
        }
        let plausible = words
            .iter()
            .enumerate()
            .map(|(i, w)| i < prior_rank && is_plausible_word(w))
            .collect();
        Vectors {
            name,
            dim,
            words,
            vecs,
            index,
            plausible,
            opener: None,
        }
    }

    /// Build from little-endian f16 rows (the compact web model).
    pub fn from_f16(
        name: String,
        dim: usize,
        words: Vec<String>,
        bytes: &[u8],
        prior_rank: usize,
    ) -> Self {
        let vecs: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| f16::to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        Self::new(name, dim, words, vecs, prior_rank)
    }

    pub fn n(&self) -> usize {
        self.words.len()
    }

    #[inline]
    pub fn vec(&self, i: usize) -> &[f32] {
        &self.vecs[i * self.dim..(i + 1) * self.dim]
    }

    pub fn lookup(&self, w: &str) -> Option<u32> {
        self.index.get(w).copied()
    }

    pub fn plausible_count(&self) -> usize {
        self.plausible.iter().filter(|&&p| p).count()
    }

    pub fn plausible_indices(&self) -> Vec<u32> {
        (0..self.n() as u32)
            .filter(|&i| self.plausible[i as usize])
            .collect()
    }

    /// out[j] = cos(vec i, vec j) for every j.
    pub fn dots_into(&self, i: usize, out: &mut [f32]) {
        let q = self.vec(i);
        let vecs = &self.vecs;
        let dim = self.dim;
        let fill = |base: usize, chunk: &mut [f32]| {
            for (k, o) in chunk.iter_mut().enumerate() {
                let r = (base + k) * dim;
                *o = dot(&vecs[r..r + dim], q);
            }
        };
        #[cfg(feature = "parallel")]
        out.par_chunks_mut(2048)
            .enumerate()
            .for_each(|(ci, chunk)| fill(ci * 2048, chunk));
        #[cfg(not(feature = "parallel"))]
        fill(0, out);
    }

    pub fn dots(&self, i: usize) -> Vec<f32> {
        let mut out = vec![0f32; self.n()];
        self.dots_into(i, &mut out);
        out
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub idx: u32,
    pub score: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ObserveInfo {
    pub before: usize,
    pub after: usize,
    pub relaxed: bool,
}

#[derive(Clone, Debug)]
pub struct Choice {
    pub idx: u32,
    pub entropy: f64,
    pub buckets: usize,
    pub probes: usize,
    pub candidates: usize,
}

/// Constraint solver: every observed score is an exact constraint on the secret.
pub struct Solver<'a> {
    pub vectors: &'a Vectors,
    pub alive: Vec<bool>,
    pub banned: Vec<bool>,
    pub restricted: bool,
    /// Score rounding scale of the game (see `SCALE_CEMANTIX`, `SCALE_QUELMOT`).
    pub scale: f64,
    pub tol_level: usize,
    pub obs: Vec<Observation>,
    pub alive_count: usize,
    sims: Vec<f32>,
}

impl<'a> Solver<'a> {
    /// Solver for Cémantix scoring (4 decimals).
    pub fn new(vectors: &'a Vectors) -> Self {
        Self::with_scale(vectors, SCALE_CEMANTIX)
    }

    pub fn with_scale(vectors: &'a Vectors, scale: f64) -> Self {
        let n = vectors.n();
        let alive = vectors.plausible.clone();
        let alive_count = alive.iter().filter(|&&a| a).count();
        Solver {
            vectors,
            alive,
            banned: vec![false; n],
            restricted: true,
            scale,
            tol_level: 0,
            obs: Vec::new(),
            alive_count,
            sims: vec![0f32; n],
        }
    }

    pub fn tol(&self) -> f64 {
        TOL_MULT[self.tol_level] / self.scale
    }

    pub fn ban(&mut self, idx: u32) {
        self.banned[idx as usize] = true;
        if self.alive[idx as usize] {
            self.alive[idx as usize] = false;
            self.alive_count -= 1;
        }
    }

    fn recount(&mut self) {
        self.alive_count = self.alive.iter().filter(|&&a| a).count();
    }

    /// Keep only candidates whose cosine with the guess matches the observed score.
    fn apply_score(&mut self, o: &Observation) {
        self.vectors.dots_into(o.idx as usize, &mut self.sims);
        let tol = self.tol();
        let sims = &self.sims;
        let gi = o.idx as usize;
        let score = o.score;
        let test = |j: usize, a: &mut bool| {
            if *a && (j == gi || (sims[j] as f64 - score).abs() > tol) {
                *a = false;
            }
        };
        #[cfg(feature = "parallel")]
        self.alive
            .par_iter_mut()
            .enumerate()
            .for_each(|(j, a)| test(j, a));
        #[cfg(not(feature = "parallel"))]
        self.alive
            .iter_mut()
            .enumerate()
            .for_each(|(j, a)| test(j, a));
        self.recount();
    }

    fn rebuild(&mut self) {
        let n = self.vectors.n();
        for j in 0..n {
            self.alive[j] = !self.banned[j] && (!self.restricted || self.vectors.plausible[j]);
        }
        self.recount();
        let obs = self.obs.clone();
        for o in &obs {
            self.apply_score(o);
        }
    }

    pub fn observe(&mut self, o: Observation) -> ObserveInfo {
        let before = self.alive_count;
        self.obs.push(o);
        self.apply_score(&o);
        let mut relaxed = false;
        if self.alive_count == 0 && self.restricted {
            self.restricted = false;
            relaxed = true;
            self.rebuild();
        }
        while self.alive_count == 0 && self.tol_level + 1 < TOL_MULT.len() {
            self.tol_level += 1;
            relaxed = true;
            self.rebuild();
        }
        ObserveInfo {
            before,
            after: self.alive_count,
            relaxed,
        }
    }

    pub fn candidates(&self) -> Vec<u32> {
        self.alive
            .iter()
            .enumerate()
            .filter(|&(_, &a)| a)
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// First k candidates in frequency order (= most plausible secrets).
    pub fn top(&self, k: usize) -> Vec<u32> {
        self.alive
            .iter()
            .enumerate()
            .filter(|&(_, &a)| a)
            .map(|(i, _)| i as u32)
            .take(k)
            .collect()
    }

    fn stride_sample(v: &[u32], max: usize) -> Vec<u32> {
        if v.len() <= max {
            return v.to_vec();
        }
        let step = v.len() as f64 / max as f64;
        (0..max).map(|k| v[(k as f64 * step) as usize]).collect()
    }

    /// Expected information of playing `idx` now (entropy over the remaining candidates).
    pub fn evaluate(&self, idx: u32) -> (f64, usize) {
        let cands = self.candidates();
        partition_entropy(
            &self.vectors.vecs,
            self.vectors.dim,
            idx as usize,
            &cands,
            self.scale,
        )
    }

    /// Pick the guess that maximises the expected information (entropy of the
    /// rounded-score partition over the remaining candidates).
    pub fn next_guess(&mut self) -> Choice {
        let cands = self.candidates();
        if self.obs.is_empty()
            && let Some(op) = self.vectors.opener
            && !self.banned[op as usize]
        {
            return Choice {
                idx: op,
                entropy: f64::NAN,
                buckets: 0,
                probes: 0,
                candidates: cands.len(),
            };
        }
        if cands.is_empty() {
            let idx = (0..self.vectors.n() as u32)
                .find(|&i| !self.banned[i as usize] && self.vectors.plausible[i as usize])
                .unwrap_or(0);
            return Choice {
                idx,
                entropy: 0.0,
                buckets: 0,
                probes: 0,
                candidates: 0,
            };
        }
        if cands.len() == 1 {
            return Choice {
                idx: cands[0],
                entropy: 0.0,
                buckets: 1,
                probes: 1,
                candidates: 1,
            };
        }
        let probes = Self::stride_sample(&cands, MAX_PROBES);
        let targets = Self::stride_sample(&cands, MAX_TARGETS);
        let vecs = &self.vectors.vecs;
        let dim = self.vectors.dim;
        let scale = self.scale;
        let eval = |p: &u32| {
            let (h, b) = partition_entropy(vecs, dim, *p as usize, &targets, scale);
            (h, b, *p)
        };
        // higher entropy wins; ties go to the more frequent word (lower index)
        let better = |a: (f64, usize, u32), b: (f64, usize, u32)| {
            if b.0 > a.0 + 1e-9 || ((b.0 - a.0).abs() <= 1e-9 && b.2 < a.2) {
                b
            } else {
                a
            }
        };
        let init = (f64::NEG_INFINITY, 0usize, u32::MAX);
        #[cfg(feature = "parallel")]
        let best = probes.par_iter().map(eval).reduce(|| init, better);
        #[cfg(not(feature = "parallel"))]
        let best = probes.iter().map(eval).fold(init, better);
        Choice {
            idx: best.2,
            entropy: best.0,
            buckets: best.1,
            probes: probes.len(),
            candidates: cands.len(),
        }
    }
}
