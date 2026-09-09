//! WebAssembly bindings: an `Engine` holding the compact f16 model and one solver.
//! Results are returned as JSON strings to keep the glue minimal.

use cemantix_core::{Observation, PRIOR_RANK, SCALE_CEMANTIX, Solver, Vectors, tol_level_for};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Engine {
    vectors: &'static Vectors,
    solver: Solver<'static>,
    scale: f64,
    model_error: f64,
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
    /// `opener` = precomputed first guess, `model_error` = max score error of the
    /// compressed model (from meta.json), `scale` = the game's score rounding
    /// (10 000 for Cémantix, 1 000 for QuelMot).
    #[wasm_bindgen(constructor)]
    pub fn new(
        f16: &[u8],
        words: &str,
        dim: usize,
        opener: Option<String>,
        model_error: f64,
        scale: Option<f64>,
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
        let scale = scale.unwrap_or(SCALE_CEMANTIX);
        let mut solver = Solver::with_scale(vectors, scale);
        solver.tol_level = tol_level_for(scale, model_error);
        Ok(Engine {
            vectors,
            solver,
            scale,
            model_error,
        })
    }

    /// Start a new game with the current scale.
    pub fn reset(&mut self) {
        self.solver = Solver::with_scale(self.vectors, self.scale);
        self.solver.tol_level = tol_level_for(self.scale, self.model_error);
    }

    /// Switch game (score rounding scale) and start a new game.
    pub fn set_scale(&mut self, scale: f64) {
        self.scale = scale;
        self.reset();
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn tol(&self) -> f64 {
        self.solver.tol()
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
        let mut ch = self.solver.next_guess();
        let opener = ch.entropy.is_nan();
        if opener {
            // The opener is precomputed: measure its entropy at this game's scale.
            let (h, b) = self.solver.evaluate(ch.idx);
            ch.entropy = h;
            ch.buckets = b;
        }
        let out = GuessOut {
            word: self.vectors.words[ch.idx as usize].clone(),
            idx: ch.idx,
            entropy: ch.entropy,
            buckets: ch.buckets,
            candidates: ch.candidates,
            opener,
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
