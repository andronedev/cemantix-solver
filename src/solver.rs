//! Constraint solver: every observed score is an exact constraint on the secret's embedding.

use crate::model::{Model, partition_entropy};
use rayon::prelude::*;

pub const MAX_PROBES: usize = 2000;
pub const MAX_TARGETS: usize = 4000;
pub const TOLS: [f64; 5] = [1e-4, 2e-4, 5e-4, 1e-3, 3e-3];
/// Rank constraints cost one full matvec per candidate: only apply on small sets.
const RANK_MAX_CANDS_IN_TOP: usize = 120;
const RANK_MAX_CANDS_OUT: usize = 20;

#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub idx: u32,
    pub score: f64,
    pub percentile: Option<u32>,
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

pub struct Solver<'a> {
    pub model: &'a Model,
    pub alive: Vec<bool>,
    pub banned: Vec<bool>,
    pub restricted: bool,
    /// Rank (‰) constraints are off by default: the server ranks over its own
    /// filtered lexicon, so our full-vocabulary ranks do not match.
    pub use_rank: bool,
    pub tol_level: usize,
    pub obs: Vec<Observation>,
    pub alive_count: usize,
    sims: Vec<f32>,
}

impl<'a> Solver<'a> {
    pub fn new(model: &'a Model) -> Self {
        let n = model.n();
        let alive = model.plausible.clone();
        let alive_count = alive.iter().filter(|&&a| a).count();
        Solver {
            model,
            alive,
            banned: vec![false; n],
            restricted: true,
            use_rank: false,
            tol_level: 0,
            obs: Vec::new(),
            alive_count,
            sims: vec![0f32; n],
        }
    }

    pub fn tol(&self) -> f64 {
        TOLS[self.tol_level]
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
        self.model.dots_into(o.idx as usize, &mut self.sims);
        let tol = self.tol();
        let sims = &self.sims;
        self.alive
            .par_iter_mut()
            .enumerate()
            .filter(|(_, a)| **a)
            .for_each(|(j, a)| {
                if j == o.idx as usize || (sims[j] as f64 - o.score).abs() > tol {
                    *a = false;
                }
            });
        self.recount();
    }

    /// Rank constraint (soft): the guess must sit at the right position among the
    /// candidate's neighbours. Only applied when the candidate set is small.
    fn apply_rank(&mut self, o: &Observation) {
        let cands: Vec<u32> = self.candidates();
        let limit = if o.percentile.is_some() {
            RANK_MAX_CANDS_IN_TOP
        } else {
            RANK_MAX_CANDS_OUT
        };
        if cands.is_empty() || cands.len() > limit {
            return;
        }
        let model = self.model;
        let g = o.idx as usize;
        let mut buf = vec![0f32; model.n()];
        let mut keep = Vec::with_capacity(cands.len());
        for &c in &cands {
            let c = c as usize;
            model.dots_into(c, &mut buf);
            let sg = buf[g];
            let closer = buf
                .par_iter()
                .enumerate()
                .filter(|&(w, &s)| w != c && w != g && s > sg + 1e-6)
                .count();
            let ok = match o.percentile {
                Some(p) => {
                    let expected = 999i64 - p as i64;
                    (closer as i64 - expected).abs() <= 3
                }
                None => closer >= 995,
            };
            keep.push(ok);
        }
        let kept = keep.iter().filter(|&&k| k).count();
        if kept == 0 {
            return; // soft constraint: never wipe the set
        }
        for (i, &c) in cands.iter().enumerate() {
            if !keep[i] {
                self.alive[c as usize] = false;
            }
        }
        self.recount();
    }

    fn rebuild(&mut self) {
        let n = self.model.n();
        for j in 0..n {
            self.alive[j] = !self.banned[j] && (!self.restricted || self.model.plausible[j]);
        }
        self.recount();
        let obs = self.obs.clone();
        for o in &obs {
            self.apply_score(o);
        }
        if self.use_rank {
            for o in &obs {
                self.apply_rank(o);
            }
        }
    }

    pub fn observe(&mut self, o: Observation) -> ObserveInfo {
        let before = self.alive_count;
        self.obs.push(o);
        self.apply_score(&o);
        if self.use_rank {
            self.apply_rank(&o);
        }
        let mut relaxed = false;
        if self.alive_count == 0 && self.restricted {
            self.restricted = false;
            relaxed = true;
            self.rebuild();
        }
        while self.alive_count == 0 && self.tol_level + 1 < TOLS.len() {
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

    /// Pick the guess that maximises the expected information (entropy of the
    /// rounded-score partition over the remaining candidates).
    pub fn next_guess(&mut self) -> Choice {
        let cands = self.candidates();
        if self.obs.is_empty()
            && let Some(op) = self.model.opener
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
            // Nothing matches: fall back to the most frequent plausible unbanned word.
            let idx = (0..self.model.n() as u32)
                .find(|&i| !self.banned[i as usize] && self.model.plausible[i as usize])
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
        let vecs = &self.model.vecs;
        let dim = self.model.dim;
        let best = probes
            .par_iter()
            .map(|&p| {
                let (h, b) = partition_entropy(vecs, dim, p as usize, &targets);
                (h, b, p)
            })
            .reduce(
                || (f64::NEG_INFINITY, 0usize, u32::MAX),
                |a, b| {
                    // higher entropy wins; ties go to the more frequent word (lower index)
                    if b.0 > a.0 + 1e-9 || ((b.0 - a.0).abs() <= 1e-9 && b.2 < a.2) {
                        b
                    } else {
                        a
                    }
                },
            );
        Choice {
            idx: best.2,
            entropy: best.0,
            buckets: best.1,
            probes: probes.len(),
            candidates: cands.len(),
        }
    }
}
