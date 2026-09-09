//! Cémantix / QuelMot solver core: vectors, score constraints, entropy-based guess choice.
//! No I/O and no network, so it runs natively (feature `parallel` = rayon) and in wasm.
//!
//! Two games, two kinds of constraint:
//!
//! - **Cémantix** returns the cosine between the guess and the secret, rounded to four
//!   decimals. Each answer is an exact equation and the secret is trilaterated.
//! - **QuelMot** returns `1000 − rank of the guess among the secret's neighbours`, floored
//!   at −999. Each answer places the guess in a *band of ranks* around the secret. Testing
//!   that for every candidate would be an N² job per turn, so neighbour similarities are
//!   tabulated once (see [`RankTable`]) and a rank constraint then costs one matvec plus
//!   an interpolation.

use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Words beyond this frequency rank are not considered plausible secrets.
pub const PRIOR_RANK: usize = 50_000;
pub const MAX_PROBES: usize = 2000;
pub const MAX_TARGETS: usize = 4000;
/// Tolerance ladder on |cos − score|, in units of the game's rounding step
/// (Cémantix rounds the cosine to 1e-4).
pub const TOL_MULT: [f64; 5] = [1.0, 2.0, 5.0, 10.0, 30.0];
/// Cémantix: score = round(cos × 10 000) / 10 000.
pub const SCALE_CEMANTIX: f64 = 10_000.0;

/// Ranks at which the neighbour similarities of each word are tabulated. A rank score
/// only needs to be localised in log-rank, so geometric levels are enough — but the gaps
/// have to stay well under the accepted window, or the interpolation error alone would
/// rule out the secret. Measured on a synthetic lexicon, halving the gaps from ×2.2 to
/// ×1.5 takes the 99th percentile of the estimation error from ×1.41 to ×1.08, for
/// 36 bytes per word instead of 24.
pub const RANK_LEVELS: [u32; 18] = [
    1, 2, 3, 4, 6, 8, 12, 18, 26, 40, 60, 90, 140, 250, 450, 800, 1400, 2000,
];
/// Rank estimates are clamped here: past the last level the exact value stops mattering,
/// every score down there is floored anyway.
pub const RANK_CAP: f64 = 1e6;
/// The rank window is widened by raising it to these powers when no candidate survives.
pub const RANK_TOL_MULT: [i32; 5] = [1, 2, 3, 5, 8];
/// Probe/target caps of the rank solver, kept low: the browser has a single thread.
pub const RANK_PROBES: usize = 300;
pub const RANK_TARGETS: usize = 1200;
/// Frequent words always tried as probes, candidates or not. A rank score says something
/// only when the guess is a close neighbour of the secret, so the best probes are hubs.
pub const RANK_HUBS: usize = 300;

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

/// Round a cosine the way a game does (`scale` = 10 000 for Cémantix).
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

/// Shannon entropy (bits) and bucket count of a partition given by its sorted keys.
fn entropy_of_keys(keys: &mut [i32]) -> (f64, usize) {
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

/// Entropy (bits) of the rounded-cosine partition induced by probe `p` over `targets`,
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
    entropy_of_keys(&mut keys)
}

/// Entropy (bits) of the rank partition induced by probe `p` over `targets`.
///
/// Buckets are geometric in rank, one window wide: that is the resolution a rank
/// constraint actually has, so counting finer would overstate the information. Every
/// target whose answer would be floored falls in one bucket (they are indistinguishable),
/// and the target that *is* the probe gets its own (the game ends there).
pub fn rank_partition_entropy(
    vecs: &[f32],
    dim: usize,
    table: &RankTable,
    p: usize,
    targets: &[u32],
    model: &RankModel,
) -> (f64, usize) {
    let q = &vecs[p * dim..(p + 1) * dim];
    let far = model.floor_rank().min(table.reach());
    let step = model.window.ln().max(1e-6);
    let mut keys: Vec<i32> = targets
        .iter()
        .map(|&t| {
            let t = t as usize;
            if t == p {
                return i32::MIN;
            }
            let sim = dot(&vecs[t * dim..(t + 1) * dim], q);
            match table.rank(t, sim) {
                Some(r) if r < far => (r.ln() / step).floor() as i32,
                _ => i32::MAX,
            }
        })
        .collect();
    entropy_of_keys(&mut keys)
}

/// Per-word similarity quantiles: `sims[row(w) * L + k]` is the cosine between `w` and
/// its `levels[k]`-th nearest neighbour, ranks being counted inside the ranking lexicon.
///
/// Built once with a single N² pass, it answers "at which rank does word g sit among the
/// neighbours of c?" by interpolation, for every candidate c at the price of one matvec.
#[derive(Clone, Debug)]
pub struct RankTable {
    /// Tabulated ranks, ascending.
    pub levels: Vec<u32>,
    /// Row of each word, `u32::MAX` for words outside the ranking lexicon.
    pub row_of: Vec<u32>,
    /// `rows × levels.len()` cosines, decreasing along a row.
    pub sims: Vec<f32>,
}

impl RankTable {
    pub fn rows(&self) -> usize {
        self.sims.len() / self.levels.len()
    }

    /// Last tabulated rank: beyond it the estimate is an extrapolation, good enough to
    /// order words but not to compare against a target rank.
    pub fn reach(&self) -> f64 {
        self.levels.last().copied().unwrap_or(1) as f64
    }

    fn blank(n: usize, rows: &[u32], levels: &[u32]) -> RankTable {
        let mut row_of = vec![u32::MAX; n];
        for (r, &w) in rows.iter().enumerate() {
            row_of[w as usize] = r as u32;
        }
        RankTable {
            levels: levels.to_vec(),
            row_of,
            sims: vec![0f32; rows.len() * levels.len()],
        }
    }

    /// Tabulate the neighbour similarities of every word of `rows`, ranks counted inside
    /// `rows` itself (the plausible lexicon). This is the one N² pass.
    pub fn build(v: &Vectors, rows: &[u32], levels: &[u32]) -> RankTable {
        let (m, dim, l) = (rows.len(), v.dim, levels.len());
        assert!(m >= 2 && l > 0, "empty ranking lexicon");
        // Gather the lexicon contiguously: the inner loop then walks memory in order.
        let mut lex = vec![0f32; m * dim];
        for (r, &w) in rows.iter().enumerate() {
            lex[r * dim..(r + 1) * dim].copy_from_slice(v.vec(w as usize));
        }
        let take = (*levels.last().unwrap() as usize).min(m - 1);
        let mut table = Self::blank(v.n(), rows, levels);
        let fill = |first: usize, chunk: &mut [f32]| {
            let mut buf = vec![0f32; m];
            for (k, out) in chunk.chunks_mut(l).enumerate() {
                let r = first + k;
                let q = &lex[r * dim..(r + 1) * dim];
                for (j, b) in buf.iter_mut().enumerate() {
                    *b = dot(&lex[j * dim..(j + 1) * dim], q);
                }
                buf[r] = f32::NEG_INFINITY; // a word is not its own neighbour
                buf.select_nth_unstable_by(take, |a, b| b.total_cmp(a));
                let head = &mut buf[..=take];
                head.sort_unstable_by(|a, b| b.total_cmp(a));
                for (i, &lv) in levels.iter().enumerate() {
                    out[i] = head[(lv as usize - 1).min(head.len() - 1)];
                }
            }
        };
        #[cfg(feature = "parallel")]
        table
            .sims
            .par_chunks_mut(l * 64)
            .enumerate()
            .for_each(|(c, chunk)| fill(c * 64, chunk));
        #[cfg(not(feature = "parallel"))]
        fill(0, &mut table.sims);
        table
    }

    pub fn to_f16(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.sims.len() * 2);
        for &x in &self.sims {
            out.extend_from_slice(&f16::from_f32(x).to_le_bytes());
        }
        out
    }

    /// Rebuild from the little-endian f16 dump, `rows` being the same lexicon, in order.
    pub fn from_f16(n: usize, rows: &[u32], levels: &[u32], bytes: &[u8]) -> Option<RankTable> {
        if bytes.len() != rows.len() * levels.len() * 2 {
            return None;
        }
        let mut t = Self::blank(n, rows, levels);
        for (o, c) in t.sims.iter_mut().zip(bytes.chunks_exact(2)) {
            *o = f16::to_f32(u16::from_le_bytes([c[0], c[1]]));
        }
        Some(t)
    }

    /// Rank of a word sitting at cosine `sim` among `word`'s neighbours. Log-rank is
    /// interpolated linearly between the tabulated levels, and extrapolated along the
    /// last segment below the table. `None` when `word` has no row.
    pub fn rank(&self, word: usize, sim: f32) -> Option<f64> {
        let r = *self.row_of.get(word)?;
        if r == u32::MAX {
            return None;
        }
        let l = self.levels.len();
        let row = &self.sims[r as usize * l..r as usize * l + l];
        let sim = sim as f64;
        let ln = |i: usize| (self.levels[i] as f64).ln();
        if sim >= row[0] as f64 {
            return Some(self.levels[0] as f64);
        }
        for i in 0..l - 1 {
            if sim >= row[i + 1] as f64 {
                let (a, b) = (row[i] as f64, row[i + 1] as f64);
                let t = if a > b { (a - sim) / (a - b) } else { 1.0 };
                return Some((ln(i) + t * (ln(i + 1) - ln(i))).exp());
            }
        }
        let (a, b) = (row[l - 2] as f64, row[l - 1] as f64);
        let slope = if a > b {
            (ln(l - 1) - ln(l - 2)) / (a - b)
        } else {
            0.0
        };
        Some((ln(l - 1) + (b - sim) * slope).exp().min(RANK_CAP))
    }
}

/// QuelMot's scoring: `score = top − rank of the guess among the secret's neighbours`,
/// floored at `floor`. Their lexicon is larger than ours, so a site rank is about
/// `alpha` times the rank we measure locally.
#[derive(Clone, Copy, Debug)]
pub struct RankModel {
    /// Site lexicon size divided by ours: site_rank ≈ alpha × local_rank.
    pub alpha: f64,
    /// Multiplicative half-width of the accepted rank window.
    pub window: f64,
    /// Additive slack, so the window still breathes at rank 1 or 2.
    pub slack: f64,
    pub top: f64,
    pub floor: f64,
}

impl RankModel {
    /// Measured on quelmot.fr in September 2026: their lexicon is about 1.5× ours.
    pub const QUELMOT: RankModel = RankModel {
        alpha: 1.5,
        window: 1.3,
        slack: 0.5,
        top: 1000.0,
        floor: -999.0,
    };

    /// Local rank implied by a score, `None` when the score sits on the floor.
    pub fn local_rank(&self, score: f64) -> Option<f64> {
        (score > self.floor).then(|| ((self.top - score) / self.alpha).max(1.0))
    }

    /// Smallest local rank whose score is floored: below it, an answer says nothing
    /// beyond "far away".
    pub fn floor_rank(&self) -> f64 {
        ((self.top - self.floor) / self.alpha).max(1.0)
    }

    /// Score the site would return for a guess sitting at local rank `r`.
    pub fn score_at(&self, r: f64) -> f64 {
        (self.top - (self.alpha * r).round()).max(self.floor)
    }

    /// Is local rank `r` compatible with `score`? `mult` widens the window; `reach` is the
    /// last rank the table resolves, past which an estimate only means "far away".
    pub fn accepts(&self, score: f64, r: f64, mult: i32, reach: f64) -> bool {
        let w = self.window.powi(mult);
        match self.local_rank(score) {
            // A floored score, or one pointing past the table: only a lower bound holds.
            None => r >= (self.floor_rank() / w).min(reach) - self.slack,
            Some(t) if t >= reach => r >= (t / w).min(reach) - self.slack,
            Some(t) => r >= t / w - self.slack && r <= t * w + self.slack,
        }
    }
}

impl Default for RankModel {
    fn default() -> Self {
        Self::QUELMOT
    }
}

/// What a score means, and therefore what constraint it puts on the secret.
#[derive(Clone, Copy, Debug)]
pub enum Scoring {
    /// The score is the cosine, rounded to 1/`scale` (Cémantix).
    Cosine { scale: f64 },
    /// The score is a rank in disguise (QuelMot).
    Rank(RankModel),
}

impl Scoring {
    pub const CEMANTIX: Scoring = Scoring::Cosine {
        scale: SCALE_CEMANTIX,
    };
    pub const QUELMOT: Scoring = Scoring::Rank(RankModel::QUELMOT);

    pub fn is_rank(&self) -> bool {
        matches!(self, Scoring::Rank(_))
    }

    /// Has the secret just been found?
    pub fn solved(&self, score: f64, percentile: Option<u32>) -> bool {
        match self {
            Scoring::Cosine { .. } => percentile == Some(1000) || score >= 0.99995,
            Scoring::Rank(m) => score >= m.top - 0.5,
        }
    }
}

impl Default for Scoring {
    fn default() -> Self {
        Scoring::CEMANTIX
    }
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
    /// Opener for the rank games, where the Cémantix one is worthless.
    pub rank_opener: Option<u32>,
    /// Neighbour quantiles, needed by [`Scoring::Rank`].
    pub ranks: Option<RankTable>,
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
            rank_opener: None,
            ranks: None,
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

    /// out[k] = cos(vec i, vec subset[k]). Filtering only ever looks at words that are
    /// still alive, so the solver never pays for the whole vocabulary.
    pub fn dots_subset_into(&self, i: usize, subset: &[u32], out: &mut [f32]) {
        assert_eq!(subset.len(), out.len());
        let q = self.vec(i);
        let vecs = &self.vecs;
        let dim = self.dim;
        let fill = |idx: &[u32], chunk: &mut [f32]| {
            for (k, o) in chunk.iter_mut().enumerate() {
                let r = idx[k] as usize * dim;
                *o = dot(&vecs[r..r + dim], q);
            }
        };
        #[cfg(feature = "parallel")]
        out.par_chunks_mut(2048)
            .zip(subset.par_chunks(2048))
            .for_each(|(chunk, idx)| fill(idx, chunk));
        #[cfg(not(feature = "parallel"))]
        fill(subset, out);
    }

    /// Exact rank of every word of `lexicon` among `of`'s neighbours (1 = nearest, `of`
    /// itself excluded). Counted, not interpolated: this is the reference the tabulated
    /// estimate is checked against.
    pub fn exact_ranks(&self, of: usize, lexicon: &[u32]) -> Vec<u32> {
        let mut sims = vec![0f32; lexicon.len()];
        self.dots_subset_into(of, lexicon, &mut sims);
        let mut order: Vec<u32> = (0..lexicon.len() as u32).collect();
        order.sort_unstable_by(|&a, &b| sims[b as usize].total_cmp(&sims[a as usize]));
        let mut ranks = vec![0u32; lexicon.len()];
        let mut r = 0u32;
        for &k in &order {
            if lexicon[k as usize] as usize == of {
                ranks[k as usize] = 0;
            } else {
                r += 1;
                ranks[k as usize] = r;
            }
        }
        ranks
    }
}

/// One answer from the game. `score` is what the game printed: a cosine for Cémantix,
/// the raw integer for QuelMot.
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

/// Constraint solver: every observed score narrows the set of possible secrets.
pub struct Solver<'a> {
    pub vectors: &'a Vectors,
    pub alive: Vec<bool>,
    pub banned: Vec<bool>,
    pub restricted: bool,
    pub scoring: Scoring,
    /// Index into [`TOL_MULT`] (cosine) or [`RANK_TOL_MULT`] (rank).
    pub tol_level: usize,
    pub obs: Vec<Observation>,
    pub alive_count: usize,
    pub max_probes: usize,
    pub max_targets: usize,
    sims: Vec<f32>,
}

impl<'a> Solver<'a> {
    /// Solver for Cémantix scoring (4 decimals).
    pub fn new(vectors: &'a Vectors) -> Self {
        Self::with_scoring(vectors, Scoring::CEMANTIX)
    }

    pub fn with_scoring(vectors: &'a Vectors, scoring: Scoring) -> Self {
        let n = vectors.n();
        let alive = vectors.plausible.clone();
        let alive_count = alive.iter().filter(|&&a| a).count();
        let (max_probes, max_targets) = if scoring.is_rank() {
            (RANK_PROBES, RANK_TARGETS)
        } else {
            (MAX_PROBES, MAX_TARGETS)
        };
        Solver {
            vectors,
            alive,
            banned: vec![false; n],
            restricted: true,
            scoring,
            tol_level: 0,
            obs: Vec::new(),
            alive_count,
            max_probes,
            max_targets,
            sims: Vec::new(),
        }
    }

    fn tol_levels(&self) -> usize {
        match self.scoring {
            Scoring::Cosine { .. } => TOL_MULT.len(),
            Scoring::Rank(_) => RANK_TOL_MULT.len(),
        }
    }

    /// Current tolerance: a cosine half-width for Cémantix, a rank ratio for QuelMot.
    pub fn tol(&self) -> f64 {
        match self.scoring {
            Scoring::Cosine { scale } => TOL_MULT[self.tol_level] / scale,
            Scoring::Rank(m) => m.window.powi(RANK_TOL_MULT[self.tol_level]),
        }
    }

    pub fn tol_label(&self) -> String {
        match self.scoring {
            Scoring::Cosine { .. } => format!("±{}", self.tol()),
            Scoring::Rank(_) => format!("×÷{:.2}", self.tol()),
        }
    }

    /// The opener this game should start from.
    pub fn opener(&self) -> Option<u32> {
        match self.scoring {
            Scoring::Cosine { .. } => self.vectors.opener,
            Scoring::Rank(_) => self.vectors.rank_opener,
        }
    }

    pub fn ban(&mut self, idx: u32) {
        self.banned[idx as usize] = true;
        if self.alive[idx as usize] {
            self.alive[idx as usize] = false;
            self.alive_count -= 1;
        }
    }

    /// Keep only the candidates whose relation to the guess matches the observed score.
    fn apply_score(&mut self, o: &Observation) {
        let cands = self.candidates();
        if cands.is_empty() {
            return;
        }
        let mut sims = std::mem::take(&mut self.sims);
        sims.clear();
        sims.resize(cands.len(), 0f32);
        self.vectors
            .dots_subset_into(o.idx as usize, &cands, &mut sims);
        let gi = o.idx as usize;
        let score = o.score;
        let keep: Vec<bool> = match self.scoring {
            Scoring::Cosine { .. } => {
                let tol = self.tol();
                let test =
                    |(&j, &s): (&u32, &f32)| j as usize != gi && (s as f64 - score).abs() <= tol;
                #[cfg(feature = "parallel")]
                {
                    cands.par_iter().zip(sims.par_iter()).map(test).collect()
                }
                #[cfg(not(feature = "parallel"))]
                {
                    cands.iter().zip(sims.iter()).map(test).collect()
                }
            }
            Scoring::Rank(m) => {
                let mult = RANK_TOL_MULT[self.tol_level];
                let table = self.vectors.ranks.as_ref();
                let reach = table.map(|t| t.reach()).unwrap_or(1.0);
                let test = |(&j, &s): (&u32, &f32)| {
                    j as usize != gi
                        && table
                            .and_then(|t| t.rank(j as usize, s))
                            .is_some_and(|r| m.accepts(score, r, mult, reach))
                };
                #[cfg(feature = "parallel")]
                {
                    cands.par_iter().zip(sims.par_iter()).map(test).collect()
                }
                #[cfg(not(feature = "parallel"))]
                {
                    cands.iter().zip(sims.iter()).map(test).collect()
                }
            }
        };
        for (k, &j) in cands.iter().enumerate() {
            if !keep[k] {
                self.alive[j as usize] = false;
            }
        }
        self.alive_count = keep.iter().filter(|&&k| k).count();
        self.sims = sims;
    }

    fn rebuild(&mut self) {
        let n = self.vectors.n();
        for j in 0..n {
            self.alive[j] = !self.banned[j] && (!self.restricted || self.vectors.plausible[j]);
        }
        self.alive_count = self.alive.iter().filter(|&&a| a).count();
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
        // Dropping the frequency prior only makes sense for cosine scoring: the rank
        // table is defined on the plausible lexicon, outside it there is nothing to test.
        if self.alive_count == 0 && self.restricted && !self.scoring.is_rank() {
            self.restricted = false;
            relaxed = true;
            self.rebuild();
        }
        while self.alive_count == 0 && self.tol_level + 1 < self.tol_levels() {
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

    fn entropy_over(&self, p: u32, targets: &[u32]) -> (f64, usize) {
        match self.scoring {
            Scoring::Cosine { scale } => partition_entropy(
                &self.vectors.vecs,
                self.vectors.dim,
                p as usize,
                targets,
                scale,
            ),
            Scoring::Rank(m) => match &self.vectors.ranks {
                Some(t) => rank_partition_entropy(
                    &self.vectors.vecs,
                    self.vectors.dim,
                    t,
                    p as usize,
                    targets,
                    &m,
                ),
                None => (0.0, 0),
            },
        }
    }

    /// Expected information of playing `idx` now (entropy over the remaining candidates).
    pub fn evaluate(&self, idx: u32) -> (f64, usize) {
        let cands = self.candidates();
        self.entropy_over(idx, &cands)
    }

    /// Words worth evaluating as the next guess.
    fn probe_set(&self, cands: &[u32]) -> Vec<u32> {
        let mut probes = Self::stride_sample(cands, self.max_probes);
        if self.scoring.is_rank() {
            // A rank answer is informative only when the guess is a near neighbour of the
            // secret, so the best probe is often a hub rather than a candidate.
            probes.extend(
                (0..self.vectors.n() as u32)
                    .filter(|&i| self.vectors.plausible[i as usize] && !self.banned[i as usize])
                    .take(RANK_HUBS),
            );
            probes.sort_unstable();
            probes.dedup();
        }
        probes
    }

    /// Pick the guess that maximises the expected information.
    pub fn next_guess(&self) -> Choice {
        let cands = self.candidates();
        if self.obs.is_empty()
            && let Some(op) = self.opener()
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
        let probes = self.probe_set(&cands);
        let targets = Self::stride_sample(&cands, self.max_targets);
        let eval = |p: &u32| {
            let (h, b) = self.entropy_over(*p, &targets);
            (h, b, *p)
        };
        // higher entropy wins; ties go to a live candidate (it might be the secret),
        // then to the more frequent word
        let alive = &self.alive;
        let is_cand = |i: u32| i != u32::MAX && alive[i as usize];
        let better = |a: (f64, usize, u32), b: (f64, usize, u32)| {
            if b.0 > a.0 + 1e-9 {
                return b;
            }
            if a.0 > b.0 + 1e-9 {
                return a;
            }
            match (is_cand(a.2), is_cand(b.2)) {
                (false, true) => b,
                (true, false) => a,
                _ if b.2 < a.2 => b,
                _ => a,
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

// ───────────────────────────── indices ─────────────────────────────
//
// The solver knows the answer long before the player does, so a hint has to be built out
// of what sits *near* the surviving candidates without ever being one of them. Two
// measurements on the real model (frWac 500d cbow, 48 965 words) shape the design.
//
// The mean vector of the surviving candidates is not a semantic object. Cémantix
// candidates are the shell of words at a fixed cosine from the opener, not a cluster:
// their centroid has norm ≈ 0.19 and ranks the secret ~30 000th against itself. A hint
// word is therefore scored by the *minimum* of its cosines to every surviving candidate,
// which is true of all of them by construction and degenerates to plain neighbour rank
// once a single candidate is left — the regime 81 % of Cémantix games reach on the second
// answer.
//
// Absolute cosine thresholds cannot be calibrated either: the nearest neighbour of a word
// sits anywhere between 0.45 and 0.69 depending on the word. The bands below are therefore
// *positions* in that shared ordering, chosen so that no rung came up short on 120 random
// targets. `cargo run -- hints --audit N` re-measures them.

/// Above this many surviving candidates, no statement is true of all of them.
pub const HINT_MAX_SET: usize = 8;
/// Lowest pairwise cosine the candidates must reach for a shared field to exist. Scattered
/// Cémantix sets measure 0.01–0.22 ({méfier, intellectuellement, chipoter}); sets that do
/// share a field start at 0.32.
pub const HINT_MIN_COHESION: f32 = 0.30;
/// Hints are drawn from the frequent part of the lexicon: past this frequency rank frWac
/// is mostly proper nouns and typos (« nallet », « net-iris », « ziki »).
pub const HINT_FAMILIAR_RANK: usize = 20_000;
/// Two words of one rung may not exceed this cosine, or the rung says one thing twice.
pub const HINT_DIVERSITY: f32 = 0.55;
/// Folded letters in common that make two words morphological relatives.
pub const HINT_STEM_PREFIX: usize = 4;
/// `(lo, hi, take)`: the rung takes `take` words from positions `lo..hi` of the shared
/// neighbourhood.
pub const HINT_WIDE: (usize, usize, usize) = (200, 900, 4);
pub const HINT_TIGHT: (usize, usize, usize) = (25, 120, 3);
pub const HINT_NEAR: (usize, usize, usize) = (2, 12, 1);
/// A short band is widened by this factor, at most twice, before it is declared exhausted.
pub const HINT_WIDEN: usize = 2;

/// Lowercase form without French diacritics, for the stem filter. Hand-rolled, and
/// deliberately not `char::to_lowercase`: that pulls the whole Unicode case table into the
/// wasm (measured at +40 KB on a 160 KB binary) to fold twenty French letters. The lexicon
/// is lowercase already — see [`is_plausible_word`] — so the uppercase arms are only there
/// for words typed by hand.
pub fn fold_accents(w: &str) -> String {
    let mut out = String::with_capacity(w.len());
    for c in w.chars() {
        match c {
            'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' => out.push('a'),
            'ç' | 'Ç' => out.push('c'),
            'è' | 'é' | 'ê' | 'ë' | 'È' | 'É' | 'Ê' | 'Ë' => out.push('e'),
            'ì' | 'í' | 'î' | 'ï' | 'Ì' | 'Í' | 'Î' | 'Ï' => out.push('i'),
            'ñ' | 'Ñ' => out.push('n'),
            'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' => out.push('o'),
            'ù' | 'ú' | 'û' | 'ü' | 'Ù' | 'Ú' | 'Û' | 'Ü' => out.push('u'),
            'ý' | 'ÿ' | 'Ý' | 'Ÿ' => out.push('y'),
            'æ' | 'Æ' => out.push_str("ae"),
            'œ' | 'Œ' => out.push_str("oe"),
            _ => out.push(c.to_ascii_lowercase()),
        }
    }
    out
}

/// Are these two words morphological relatives? Containment once folded, or
/// [`HINT_STEM_PREFIX`] leading letters in common — « blog »/« blogueur »,
/// « démonstration »/« démontrer ». A single such word would hand the answer over, so the
/// test errs on the side of dropping a usable hint.
pub fn shares_stem(a: &str, b: &str) -> bool {
    let (fa, fb) = (fold_accents(a), fold_accents(b));
    let (short, long) = if fa.len() <= fb.len() {
        (&fa, &fb)
    } else {
        (&fb, &fa)
    };
    if short.chars().count() >= HINT_STEM_PREFIX && long.contains(short.as_str()) {
        return true;
    }
    fa.chars()
        .zip(fb.chars())
        .take_while(|(x, y)| x == y)
        .count()
        >= HINT_STEM_PREFIX
}

/// The rungs of the ladder, from vaguest to most precise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HintLevel {
    /// The semantic field, seen from far away.
    Field,
    /// The same field, closer in.
    Tight,
    /// One word sitting right next to the secret — and provably not it.
    Near,
}

impl HintLevel {
    pub const LADDER: [HintLevel; 3] = [Self::Field, Self::Tight, Self::Near];

    pub fn at(i: usize) -> Option<Self> {
        Self::LADDER.get(i).copied()
    }

    /// `(lo, hi, take)` of this rung's band.
    pub fn band(self) -> (usize, usize, usize) {
        match self {
            Self::Field => HINT_WIDE,
            Self::Tight => HINT_TIGHT,
            Self::Near => HINT_NEAR,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Field => "field",
            Self::Tight => "tight",
            Self::Near => "near",
        }
    }
}

/// Why no hint can be given right now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HintLocked {
    /// The reported scores contradict each other; nothing survives.
    NoCandidate,
    /// Too many candidates for a statement to be true of all of them.
    TooMany { alive: usize },
    /// The candidates share no field at all.
    Scattered { alive: usize },
    /// The band came up empty even after widening.
    Exhausted,
    /// Every rung has been handed out.
    Ended,
}

impl HintLocked {
    pub fn name(self) -> &'static str {
        match self {
            Self::NoCandidate => "no_candidate",
            Self::TooMany { .. } => "too_many",
            Self::Scattered { .. } => "scattered",
            Self::Exhausted => "exhausted",
            Self::Ended => "ended",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Hint {
    Words { level: HintLevel, words: Vec<u32> },
    Locked(HintLocked),
}

/// How well a word fits the surviving candidates, next to the player's own best guess.
#[derive(Clone, Copy, Debug)]
pub struct Warmth {
    pub fit: f32,
    /// The player's own guess that fits best, and its fit.
    pub best: Option<(u32, f32)>,
}

impl<'a> Solver<'a> {
    /// The candidates every hint must be true of, or `None` when there are too many — or
    /// none — for any statement to hold.
    pub fn hint_targets(&self) -> Option<Vec<u32>> {
        (1..=HINT_MAX_SET)
            .contains(&self.alive_count)
            .then(|| self.candidates())
    }

    /// Lowest cosine between two surviving candidates; 1.0 for a single one. Below
    /// [`HINT_MIN_COHESION`] the candidates have no field in common, and a hint about
    /// "the" field would not be honest.
    pub fn hint_cohesion(&self, targets: &[u32]) -> f32 {
        let mut lo = 1.0f32;
        for (k, &a) in targets.iter().enumerate() {
            for &b in &targets[k + 1..] {
                lo = lo.min(dot(
                    self.vectors.vec(a as usize),
                    self.vectors.vec(b as usize),
                ));
            }
        }
        lo
    }

    /// `out[j]` = the smallest cosine between word `j` and any of `targets`: how well `j`
    /// fits *every* surviving candidate. At most [`HINT_MAX_SET`] matvecs.
    pub fn hint_scores(&self, targets: &[u32], out: &mut Vec<f32>) {
        let n = self.vectors.n();
        out.clear();
        out.resize(n, f32::INFINITY);
        let mut buf = vec![0f32; n];
        for &t in targets {
            self.vectors.dots_into(t as usize, &mut buf);
            for (o, &s) in out.iter_mut().zip(buf.iter()) {
                *o = o.min(s);
            }
        }
    }

    /// Smallest cosine between `idx` and any of `targets`.
    fn fit_to(&self, idx: u32, targets: &[u32]) -> f32 {
        let q = self.vectors.vec(idx as usize);
        targets
            .iter()
            .map(|&t| dot(q, self.vectors.vec(t as usize)))
            .fold(f32::INFINITY, f32::min)
    }

    /// Can `j` be handed out as a hint? It must be a familiar word, not a candidate, not
    /// already played or banned, not of the same family as a candidate, and neither a
    /// relative nor a near-duplicate of a word already revealed or picked for this rung.
    fn hint_word_fits(&self, j: u32, targets: &[u32], seen: &[&[u32]]) -> bool {
        let i = j as usize;
        if i >= HINT_FAMILIAR_RANK || self.alive[i] || self.banned[i] {
            return false;
        }
        if !is_plausible_word(&self.vectors.words[i]) || self.obs.iter().any(|o| o.idx == j) {
            return false;
        }
        let w = &self.vectors.words[i];
        if targets
            .iter()
            .any(|&t| shares_stem(w, &self.vectors.words[t as usize]))
        {
            return false;
        }
        !seen.iter().flat_map(|s| s.iter()).any(|&p| {
            shares_stem(w, &self.vectors.words[p as usize])
                || dot(
                    self.vectors.vec(i),
                    self.vectors.vec(p as usize),
                ) > HINT_DIVERSITY
        })
    }

    /// Words sitting at positions `lo..hi` of the shared neighbourhood of `targets`, with
    /// everything [`Solver::hint_word_fits`] rejects skipped along the way. The frequency
    /// cap is a walk filter, not a re-ranking: re-ranking inside the frequent subset
    /// dilutes the wide band badly. A short band is widened rather than returned short.
    pub fn hint_words(
        &self,
        targets: &[u32],
        lo: usize,
        hi: usize,
        take: usize,
        exclude: &[u32],
    ) -> Vec<u32> {
        let n = self.vectors.n();
        if targets.is_empty() || take == 0 || n == 0 {
            return Vec::new();
        }
        let mut scores = Vec::new();
        self.hint_scores(targets, &mut scores);
        // One sort of the whole lexicon, reused by every widening. Ties are broken by index
        // so the ordering is a total one: the page freezes a revealed rung and re-renders
        // it forever, and a selector that reordered ties would be a latent bug.
        let cmp = |a: &u32, b: &u32| {
            scores[*b as usize]
                .total_cmp(&scores[*a as usize])
                .then(a.cmp(b))
        };
        let mut picked: Vec<u32> = Vec::with_capacity(take);
        let mut end = hi;
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_unstable_by(cmp);
        for _ in 0..=HINT_WIDEN {
            let k = end.min(n - 1);
            let head = &order[..=k];
            picked.clear();
            for &j in &head[lo.min(head.len())..] {
                if self.hint_word_fits(j, targets, &[exclude, &picked]) {
                    picked.push(j);
                    if picked.len() == take {
                        return picked;
                    }
                }
            }
            if k + 1 >= n {
                break;
            }
            end = end.saturating_mul(HINT_WIDEN);
        }
        picked
    }

    /// One rung of the ladder. Stateless: the caller owns the level counter, so a locked
    /// answer costs the player nothing.
    pub fn hint(&self, level: usize, revealed: &[u32]) -> Hint {
        let Some(rung) = HintLevel::at(level) else {
            return Hint::Locked(HintLocked::Ended);
        };
        if self.alive_count == 0 {
            return Hint::Locked(HintLocked::NoCandidate);
        }
        let Some(targets) = self.hint_targets() else {
            return Hint::Locked(HintLocked::TooMany {
                alive: self.alive_count,
            });
        };
        if self.hint_cohesion(&targets) < HINT_MIN_COHESION {
            return Hint::Locked(HintLocked::Scattered {
                alive: self.alive_count,
            });
        }
        let (lo, hi, take) = rung.band();
        let words = self.hint_words(&targets, lo, hi, take, revealed);
        if words.is_empty() {
            return Hint::Locked(HintLocked::Exhausted);
        }
        Hint::Words { level: rung, words }
    }

    /// Warmer or colder for one word: how well it fits every surviving candidate, next to
    /// the best of the player's own guesses. `None` while the ladder is locked — the field
    /// is then too wide for the comparison to mean anything.
    pub fn warmth(&self, idx: u32) -> Option<Warmth> {
        let targets = self.hint_targets()?;
        if self.hint_cohesion(&targets) < HINT_MIN_COHESION {
            return None;
        }
        let best = self
            .obs
            .iter()
            .map(|o| (o.idx, self.fit_to(o.idx, &targets)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        Some(Warmth {
            fit: self.fit_to(idx, &targets),
            best,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic model: pseudo-random unit vectors, every word plausible.
    fn toy(n: usize, dim: usize) -> Vectors {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 8_388_608.0 - 1.0
        };
        let mut vecs = vec![0f32; n * dim];
        for row in vecs.chunks_mut(dim) {
            for x in row.iter_mut() {
                *x = next();
            }
            let norm = dot(row, row).sqrt();
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
        // plausible words are lowercase letters only, so name them aaa, aab, ...
        let words: Vec<String> = (0..n)
            .map(|i| {
                let b = |k: u32| char::from(b'a' + (i as u32 / k % 26) as u8);
                format!("{}{}{}", b(676), b(26), b(1))
            })
            .collect();
        Vectors::new("toy".into(), dim, words, vecs, PRIOR_RANK)
    }

    /// Dense levels, so interpolation error stays small on a structureless model.
    const TOY_LEVELS: [u32; 12] = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 40, 80];

    fn toy_with_ranks(n: usize, dim: usize) -> Vectors {
        let mut v = toy(n, dim);
        let lex = v.plausible_indices();
        v.ranks = Some(RankTable::build(&v, &lex, &TOY_LEVELS));
        v
    }

    #[test]
    fn table_is_exact_at_the_tabulated_levels() {
        let v = toy_with_ranks(200, 16);
        let lex = v.plausible_indices();
        let table = v.ranks.as_ref().unwrap();
        for w in [0usize, 7, 91, 199] {
            let ranks = v.exact_ranks(w, &lex);
            for &lv in &TOY_LEVELS {
                let k = ranks.iter().position(|&r| r == lv).unwrap();
                let sim = dot(v.vec(w), v.vec(lex[k] as usize));
                let est = table.rank(w, sim).unwrap();
                assert!(
                    (est - lv as f64).abs() < 1e-6 * lv as f64 + 1e-6,
                    "word {w}, rank {lv}: estimated {est}"
                );
            }
        }
    }

    #[test]
    fn table_interpolates_between_levels_and_stays_monotone() {
        let v = toy_with_ranks(200, 16);
        let lex = v.plausible_indices();
        let table = v.ranks.as_ref().unwrap();
        for w in [3usize, 42, 150] {
            let ranks = v.exact_ranks(w, &lex);
            let mut last = 0f64;
            for k in 0..lex.len() {
                let r = ranks[k];
                if r == 0 || r > 80 {
                    continue;
                }
                let sim = dot(v.vec(w), v.vec(lex[k] as usize));
                let est = table.rank(w, sim).unwrap();
                assert!(
                    est >= r as f64 / 1.35 && est <= r as f64 * 1.35,
                    "word {w}, true rank {r}, estimated {est}"
                );
            }
            // rank must decrease as similarity grows
            for s in 0..40 {
                let sim = -0.4 + s as f32 * 0.02;
                let est = table.rank(w, sim).unwrap();
                assert!(est <= last || last == 0.0, "not monotone at {sim}");
                last = est;
            }
        }
    }

    #[test]
    fn table_survives_the_f16_roundtrip() {
        let v = toy_with_ranks(120, 16);
        let lex = v.plausible_indices();
        let table = v.ranks.as_ref().unwrap();
        let back =
            RankTable::from_f16(v.n(), &lex, &TOY_LEVELS, &table.to_f16()).expect("size mismatch");
        for (a, b) in table.sims.iter().zip(back.sims.iter()) {
            assert!((a - b).abs() <= a.abs() * 1e-3 + 1e-6);
        }
        assert!(RankTable::from_f16(v.n(), &lex, &TOY_LEVELS, &[0u8; 3]).is_none());
    }

    /// Play a rank game against exact local ranks and check the secret is never dropped.
    fn rank_game(model: RankModel, guesses: &[usize]) -> (usize, bool) {
        let v = toy_with_ranks(300, 16);
        let lex = v.plausible_indices();
        let secret = 123usize;
        let ranks = v.exact_ranks(secret, &lex);
        let mut solver = Solver::with_scoring(&v, Scoring::Rank(model));
        for &g in guesses {
            let score = model.score_at(ranks[g] as f64);
            solver.observe(Observation { idx: lex[g], score });
            assert!(
                solver.alive[secret],
                "secret dropped after guessing {g} (rank {}, score {score})",
                ranks[g]
            );
        }
        (solver.alive_count, solver.alive[secret])
    }

    #[test]
    fn rank_constraints_keep_the_secret_and_cut_the_field() {
        let model = RankModel {
            alpha: 1.0,
            ..RankModel::QUELMOT
        };
        // guesses at assorted distances from the secret
        let (left, kept) = rank_game(model, &[5, 60, 200, 288]);
        assert!(kept);
        assert!(left < 300, "no candidate was ruled out ({left} left)");
    }

    #[test]
    fn floored_scores_are_handled() {
        // floor at 950 => everything past local rank 50 is floored
        let model = RankModel {
            alpha: 1.0,
            floor: 950.0,
            ..RankModel::QUELMOT
        };
        assert_eq!(model.floor_rank(), 50.0);
        assert_eq!(model.local_rank(950.0), None);
        assert_eq!(model.local_rank(990.0), Some(10.0));
        assert!(model.accepts(950.0, 200.0, 1, 80.0));
        assert!(!model.accepts(950.0, 3.0, 1, 80.0));
        let (_, kept) = rank_game(model, &[5, 60, 200]);
        assert!(kept);
    }

    #[test]
    fn rank_solver_proposes_a_guess_and_scores_it() {
        let v = toy_with_ranks(300, 16);
        let solver = Solver::with_scoring(&v, Scoring::QUELMOT);
        let ch = solver.next_guess();
        assert!(ch.candidates == 300 && ch.probes > 0);
        assert!(ch.entropy > 0.0, "a probe should split something");
        let (h, b) = solver.evaluate(ch.idx);
        assert!(h >= 0.0 && b >= 1);
    }

    #[test]
    fn cosine_solver_still_trilaterates() {
        let v = toy(400, 16);
        let secret = 77usize;
        let sims = v.dots(secret);
        let mut solver = Solver::new(&v);
        for g in [3usize, 44, 300] {
            solver.observe(Observation {
                idx: g as u32,
                score: round_scale(sims[g] as f64, SCALE_CEMANTIX),
            });
        }
        assert!(solver.alive[secret]);
        assert_eq!(solver.alive_count, 1);
    }

    // ── indices ──

    #[test]
    fn stem_filter_catches_morphological_relatives() {
        // Pure strings, no model: these are the pairs that would hand the answer over.
        assert!(shares_stem("blog", "blogueur"));
        assert!(shares_stem("blog", "blogosphère"));
        assert!(shares_stem("démonstration", "démontrer"));
        assert!(shares_stem("oeuvre", "oeuvrer"));
        assert!(shares_stem("Fantôme", "fantome"));
        assert!(!shares_stem("fantôme", "zombie"));
        assert!(!shares_stem("pilier", "poutre"));
        assert!(!shares_stem("cactus", "bambou"));
        // A short word inside a long one is not a family: « art » is not « partage ».
        assert!(!shares_stem("art", "partage"));
        assert_eq!(fold_accents("Démonstration"), "demonstration");
        assert_eq!(fold_accents("cœur"), "coeur");
    }

    /// Drive the toy solver down to the single candidate `secret`, the way
    /// `cosine_solver_still_trilaterates` does.
    fn pinned(v: &Vectors, secret: usize) -> Solver<'_> {
        let sims = v.dots(secret);
        let mut solver = Solver::new(v);
        for g in [3usize, 44, 300] {
            solver.observe(Observation {
                idx: g as u32,
                score: round_scale(sims[g] as f64, SCALE_CEMANTIX),
            });
        }
        assert_eq!(solver.alive_count, 1, "the fixture should pin one candidate");
        solver
    }

    #[test]
    fn hints_are_locked_while_the_field_is_wide_open() {
        let v = toy(400, 16);
        let s = Solver::new(&v);
        assert!(matches!(
            s.hint(0, &[]),
            Hint::Locked(HintLocked::TooMany { .. })
        ));
        assert!(s.warmth(7).is_none());
    }

    #[test]
    fn scattered_candidates_are_refused() {
        let v = toy(400, 16);
        let mut s = Solver::new(&v);
        // Two nearly orthogonal words: no field is common to both.
        let sims = v.dots(0);
        let far = (1..400)
            .min_by(|&a, &b| sims[a].abs().total_cmp(&sims[b].abs()))
            .unwrap();
        s.alive.iter_mut().for_each(|a| *a = false);
        s.alive[0] = true;
        s.alive[far] = true;
        s.alive_count = 2;
        assert!(s.hint_cohesion(&[0, far as u32]) < HINT_MIN_COHESION);
        assert!(matches!(
            s.hint(0, &[]),
            Hint::Locked(HintLocked::Scattered { alive: 2 })
        ));
        assert!(s.warmth(5).is_none());
    }

    #[test]
    fn the_ladder_ends() {
        let v = toy(400, 16);
        let s = pinned(&v, 77);
        assert_eq!(HintLevel::LADDER.len(), 3);
        assert_eq!(
            s.hint(HintLevel::LADDER.len(), &[]),
            Hint::Locked(HintLocked::Ended)
        );
    }

    #[test]
    fn hints_never_name_a_candidate_a_banned_or_a_played_word() {
        let v = toy(400, 16);
        let secret = 77usize;
        let mut s = pinned(&v, secret);
        s.ban(120);
        let mut revealed: Vec<u32> = Vec::new();
        for level in 0..HintLevel::LADDER.len() {
            let Hint::Words { words, .. } = s.hint(level, &revealed) else {
                panic!("rung {level} should be available on a pinned candidate");
            };
            for &w in &words {
                let i = w as usize;
                assert!(!s.alive[i], "a candidate was revealed");
                assert_ne!(i, secret, "the secret was revealed");
                assert!(!s.banned[i], "a banned word was revealed");
                assert!(
                    !s.obs.iter().any(|o| o.idx == w),
                    "an already played word was revealed"
                );
                assert!(
                    !revealed.contains(&w),
                    "a word was handed out on two rungs"
                );
            }
            revealed.extend(words);
        }
    }

    #[test]
    fn the_ladder_gets_closer_rung_by_rung() {
        let v = toy(400, 16);
        let secret = 77usize;
        let s = pinned(&v, secret);
        let mut revealed: Vec<u32> = Vec::new();
        let mut fits = Vec::new();
        for level in 0..HintLevel::LADDER.len() {
            let Hint::Words { words, .. } = s.hint(level, &revealed) else {
                panic!("rung {level} should be available");
            };
            let mean: f32 = words
                .iter()
                .map(|&w| dot(v.vec(w as usize), v.vec(secret)))
                .sum::<f32>()
                / words.len() as f32;
            fits.push(mean);
            revealed.extend(words);
        }
        assert!(
            fits.windows(2).all(|p| p[1] > p[0]),
            "each rung must sit closer to the secret than the last: {fits:?}"
        );
    }

    #[test]
    fn a_rung_does_not_repeat_itself() {
        let v = toy(400, 16);
        let s = pinned(&v, 77);
        let Hint::Words { words, .. } = s.hint(0, &[]) else {
            panic!("the wide rung should be available");
        };
        assert!(words.len() > 1);
        for (k, &a) in words.iter().enumerate() {
            for &b in &words[k + 1..] {
                assert!(
                    dot(v.vec(a as usize), v.vec(b as usize)) <= HINT_DIVERSITY,
                    "two words of one rung say the same thing"
                );
                assert!(!shares_stem(&v.words[a as usize], &v.words[b as usize]));
            }
        }
    }

    #[test]
    fn hints_are_deterministic() {
        // The page freezes a revealed rung and re-renders it forever, so a selector that
        // reordered ties would be a latent bug.
        let v = toy(400, 16);
        let s = pinned(&v, 77);
        assert_eq!(s.hint(1, &[]), s.hint(1, &[]));
    }

    #[test]
    fn a_short_band_is_widened_before_it_is_declared_exhausted() {
        let v = toy(400, 16);
        let s = pinned(&v, 77);
        // A band with room for two words asked for six: widening must find them.
        let words = s.hint_words(&[77], 2, 4, 6, &[]);
        assert!(words.len() > 3, "the band was not widened ({words:?})");
    }

    #[test]
    fn warmth_compares_against_the_players_own_best_guess() {
        let v = toy(400, 16);
        let secret = 77usize;
        let s = pinned(&v, secret);
        let w = s.warmth(secret as u32).expect("one candidate unlocks warmth");
        assert!((w.fit - 1.0).abs() < 1e-5, "the secret fits itself");
        let (best, best_fit) = w.best.expect("three words were played");
        assert!(s.obs.iter().any(|o| o.idx == best));
        assert!(best_fit < w.fit);
    }
}
