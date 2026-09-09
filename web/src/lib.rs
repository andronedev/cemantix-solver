//! WebAssembly bindings: an `Engine` holding the compact f16 model, its rank table and
//! one solver. Results are returned as JSON strings to keep the glue minimal.

use cemantix_core::{
    Hint, HintLevel, Observation, PRIOR_RANK, RankModel, RankTable, Scoring, Solver, Vectors,
    tol_level_for,
};
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Engine {
    vectors: &'static Vectors,
    solver: Solver<'static>,
    scoring: Scoring,
    rank_model: RankModel,
    model_error: f64,
    game: String,
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
    tol: String,
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

/// One rung of the hint ladder, flat so the page has nothing to unpack.
#[derive(Serialize)]
struct HintOut {
    /// "words" when a rung was granted, "locked" otherwise.
    kind: &'static str,
    /// "field" | "tight" | "near", or the refusal:
    /// "no_candidate" | "too_many" | "scattered" | "exhausted" | "ended".
    rung: &'static str,
    level: usize,
    /// Rungs still obtainable after this call. A refusal consumes none.
    remaining: usize,
    words: Vec<String>,
    /// The same words as indices, so the page can hand them back as `revealed`.
    idx: Vec<u32>,
    /// Surviving candidates, for the refusal messages.
    alive: usize,
}

/// Warmer or colder for one word, without ever saying whether it is a candidate.
#[derive(Serialize)]
struct WarmthOut {
    word: String,
    /// How well the word fits every surviving candidate, null while the ladder is locked.
    fit: Option<f32>,
    /// The player's own guess that fits best, and its fit.
    best: Option<String>,
    best_fit: Option<f32>,
    locked: bool,
}

#[wasm_bindgen]
impl Engine {
    /// Build the engine from the pieces of `meta.json`, which the page has already
    /// parsed. Deserialising it here instead would drag a JSON reader into the wasm and
    /// nearly double it, for nine numbers and three strings.
    ///
    /// - `f16` / `words`: the compact model, little-endian half-precision rows and one
    ///   word per line.
    /// - `ranks` / `rank_levels`: the neighbour-quantile dump and the ranks it tabulates.
    ///   Leave both empty to ship without the rank games.
    /// - `rank_model`: `[alpha, window, slack, top, floor]`, empty for QuelMot's defaults.
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        f16: &[u8],
        words: &str,
        dim: usize,
        opener: Option<String>,
        model_error: f64,
        ranks: &[u8],
        rank_levels: &[u32],
        rank_opener: Option<String>,
        rank_model: &[f64],
        game: Option<String>,
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
        v.opener = opener.as_deref().and_then(|w| v.lookup(w));
        v.rank_opener = rank_opener.as_deref().and_then(|w| v.lookup(w));
        if !ranks.is_empty() && !rank_levels.is_empty() {
            let lex = v.plausible_indices();
            v.ranks = RankTable::from_f16(v.n(), &lex, rank_levels, ranks);
            if v.ranks.is_none() {
                return Err(JsError::new(&format!(
                    "table de rangs incohérente : {} octets pour {} mots x {} niveaux",
                    ranks.len(),
                    lex.len(),
                    rank_levels.len()
                )));
            }
        }
        let rank_model = match *rank_model {
            [alpha, window, slack, top, floor] => RankModel {
                alpha,
                window,
                slack,
                top,
                floor,
            },
            [] => RankModel::QUELMOT,
            _ => return Err(JsError::new("rank_model attend 5 nombres")),
        };
        let vectors: &'static Vectors = Box::leak(Box::new(v));
        let mut engine = Engine {
            vectors,
            solver: Solver::new(vectors),
            scoring: Scoring::CEMANTIX,
            rank_model,
            model_error,
            game: "cemantix".into(),
        };
        engine.set_game(game.as_deref().unwrap_or("cemantix"))?;
        Ok(engine)
    }

    /// Start a new game with the current scoring.
    pub fn reset(&mut self) {
        self.solver = Solver::with_scoring(self.vectors, self.scoring);
        if let Scoring::Cosine { scale } = self.scoring {
            self.solver.tol_level = tol_level_for(scale, self.model_error);
        }
    }

    /// Switch game and start over. Fails when the export shipped no rank table.
    pub fn set_game(&mut self, game: &str) -> Result<(), JsError> {
        let scoring = match game {
            "cemantix" => Scoring::CEMANTIX,
            "quelmot" => {
                if self.vectors.ranks.is_none() {
                    return Err(JsError::new(
                        "ce jeu se joue aux rangs, or la table de rangs n'a pas été exportée",
                    ));
                }
                Scoring::Rank(self.rank_model)
            }
            other => return Err(JsError::new(&format!("jeu inconnu « {other} »"))),
        };
        self.scoring = scoring;
        self.game = game.to_string();
        self.reset();
        Ok(())
    }

    pub fn game(&self) -> String {
        self.game.clone()
    }

    /// Can the rank games be played with this export?
    pub fn has_ranks(&self) -> bool {
        self.vectors.ranks.is_some()
    }

    pub fn tol(&self) -> String {
        self.solver.tol_label()
    }

    /// Local rank a rank-game score stands for, `None` on the floor or in a cosine game.
    pub fn rank_for(&self, score: f64) -> Option<f64> {
        match self.scoring {
            Scoring::Rank(m) => m.local_rank(score),
            Scoring::Cosine { .. } => None,
        }
    }

    /// Rank past which a rank game floors its score, so all a floored answer says is
    /// "further than this".
    pub fn floor_rank(&self) -> Option<f64> {
        match self.scoring {
            Scoring::Rank(m) => Some(m.floor_rank()),
            Scoring::Cosine { .. } => None,
        }
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
            // The opener is precomputed: measure what it is worth in this game.
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

    /// Record one answer. `score` is what the site printed: a cosine for Cémantix, the
    /// raw integer for a rank game. Free 🎁 hints go in here like any other guess.
    pub fn observe(&mut self, idx: u32, score: f64) -> String {
        let info = self.solver.observe(Observation { idx, score });
        let out = ObserveOut {
            before: info.before,
            after: info.after,
            relaxed: info.relaxed,
            tol: self.solver.tol_label(),
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

    /// Number of rungs on the hint ladder.
    pub fn hint_levels(&self) -> usize {
        HintLevel::LADDER.len()
    }

    /// One rung of the hint ladder, `revealed` being the words the earlier rungs already
    /// handed out. Stateless: the page owns the level counter, so a refusal costs the
    /// player nothing.
    pub fn hint(&self, level: usize, revealed: &[u32]) -> String {
        let total = HintLevel::LADDER.len();
        let out = match self.solver.hint(level, revealed) {
            Hint::Words { level: rung, words } => HintOut {
                kind: "words",
                rung: rung.name(),
                level,
                remaining: total.saturating_sub(level + 1),
                words: words
                    .iter()
                    .map(|&i| self.vectors.words[i as usize].clone())
                    .collect(),
                idx: words,
                alive: self.solver.alive_count,
            },
            Hint::Locked(why) => HintOut {
                kind: "locked",
                rung: why.name(),
                level,
                remaining: total.saturating_sub(level),
                words: Vec::new(),
                idx: Vec::new(),
                alive: self.solver.alive_count,
            },
        };
        serde_json::to_string(&out).unwrap()
    }

    /// Warmer or colder for a word the player is typing: how well it fits the surviving
    /// candidates, against the best of their own guesses. Never says "candidate" — that
    /// would be a complete answer rather than a hint.
    pub fn warmth(&self, idx: u32) -> String {
        let w = self.solver.warmth(idx);
        let out = WarmthOut {
            word: self.vectors.words[idx as usize].clone(),
            fit: w.map(|w| w.fit),
            best: w
                .and_then(|w| w.best)
                .map(|(i, _)| self.vectors.words[i as usize].clone()),
            best_fit: w.and_then(|w| w.best).map(|(_, f)| f),
            locked: w.is_none(),
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
