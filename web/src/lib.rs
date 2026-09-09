//! WebAssembly bindings: an `Engine` holding the compact f16 model and one solver.
//! Results are returned as JSON strings to keep the glue minimal.

use cemantix_core::{Observation, PRIOR_RANK, Solver, Vectors};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Engine {
    vectors: &'static Vectors,
    solver: Solver<'static>,
}

#[derive(Serialize)]
struct GuessOut {
    word: String,
    idx: u32,
    entropy: f64,
    buckets: usize,
    candidates: usize,
    opener: bool,
}

#[derive(Serialize)]
struct ObserveOut {
    before: usize,
    after: usize,
    relaxed: bool,
    tol: f64,
    restricted: bool,
    top: Vec<String>,
}

#[derive(Serialize)]
struct EvalOut {
    word: String,
    idx: u32,
    entropy: f64,
    buckets: usize,
    candidate: bool,
}

#[wasm_bindgen]
impl Engine {
    /// `f16` = little-endian half-precision rows, `words` = one word per line,
    /// `opener` = precomputed first guess, `tol_level` = starting tolerance index.
    #[wasm_bindgen(constructor)]
    pub fn new(
        f16: &[u8],
        words: &str,
        dim: usize,
        opener: Option<String>,
        tol_level: usize,
    ) -> Result<Engine, JsError> {
        let words: Vec<String> = words.lines().map(str::to_owned).collect();
        if f16.len() != words.len() * dim * 2 {
            return Err(JsError::new(&format!(
                "taille du modèle incohérente : {} octets pour {} mots x {} dims",
                f16.len(),
                words.len(),
                dim
            )));
        }
        let mut v = Vectors::from_f16("web".into(), dim, words, f16, PRIOR_RANK);
        v.opener = opener.and_then(|w| v.lookup(&w));
        let vectors: &'static Vectors = Box::leak(Box::new(v));
        let mut solver = Solver::new(vectors);
        solver.tol_level = tol_level.min(cemantix_core::TOLS.len() - 1);
        Ok(Engine { vectors, solver })
    }

    pub fn reset(&mut self) {
        let level = self.solver.tol_level.min(1);
        self.solver = Solver::new(self.vectors);
        self.solver.tol_level = level;
    }

    pub fn n(&self) -> usize {
        self.vectors.n()
    }

    pub fn plausible(&self) -> usize {
        self.vectors.plausible_count()
    }

    pub fn candidates(&self) -> usize {
        self.solver.alive_count
    }

    pub fn lookup(&self, word: &str) -> Option<u32> {
        self.vectors.lookup(&word.trim().to_lowercase())
    }

    pub fn word(&self, idx: u32) -> String {
        self.vectors.words[idx as usize].clone()
    }

    pub fn next_guess(&mut self) -> String {
        let ch = self.solver.next_guess();
        let out = GuessOut {
            word: self.vectors.words[ch.idx as usize].clone(),
            idx: ch.idx,
            entropy: if ch.entropy.is_nan() {
                -1.0
            } else {
                ch.entropy
            },
            buckets: ch.buckets,
            candidates: ch.candidates,
            opener: ch.entropy.is_nan(),
        };
        serde_json::to_string(&out).unwrap()
    }

    /// Expected information of a user-chosen word.
    pub fn evaluate(&self, idx: u32) -> String {
        let (entropy, buckets) = self.solver.evaluate(idx);
        let out = EvalOut {
            word: self.vectors.words[idx as usize].clone(),
            idx,
            entropy,
            buckets,
            candidate: self.solver.alive[idx as usize],
        };
        serde_json::to_string(&out).unwrap()
    }

    pub fn observe(&mut self, idx: u32, score: f64) -> String {
        let info = self.solver.observe(Observation { idx, score });
        let out = ObserveOut {
            before: info.before,
            after: info.after,
            relaxed: info.relaxed,
            tol: self.solver.tol(),
            restricted: self.solver.restricted,
            top: self
                .solver
                .top(12)
                .iter()
                .map(|&i| self.vectors.words[i as usize].clone())
                .collect(),
        };
        serde_json::to_string(&out).unwrap()
    }

    pub fn ban(&mut self, idx: u32) {
        self.solver.ban(idx);
    }

    pub fn top(&self, k: usize) -> String {
        let v: Vec<String> = self
            .solver
            .top(k)
            .iter()
            .map(|&i| self.vectors.words[i as usize].clone())
            .collect();
        serde_json::to_string(&v).unwrap()
    }
}
