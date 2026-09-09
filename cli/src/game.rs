//! Game loop shared by the live player and the offline simulator.

use crate::api::ScoreResp;
use crate::events::{Event, emoji, emoji_rank};
use crate::model::Model;
use anyhow::{Result, bail};
use cemantix_core::{Choice, Observation, RankModel, Scoring, Solver, round_scale};
use std::time::Instant;

pub trait Oracle {
    fn score(&mut self, word: &str, idx: u32) -> Result<ScoreResp>;
}

/// Offline oracle reproducing a server's scoring from the local model.
///
/// Cémantix is exact: the score *is* the local cosine. QuelMot is a simulation, since
/// their lexicon is not ours: the local rank of the guess is multiplied by `alpha` (their
/// lexicon is about 1.5× bigger) before being turned into a score, optionally with a
/// relative `jitter` to model the fact that the two lexicons do not dilate uniformly.
pub struct LocalOracle {
    pub secret: u32,
    pub scoring: Scoring,
    sims: Vec<f32>,
    thresh: f32,
    /// Local rank of every word among the secret's neighbours, `u32::MAX` outside the
    /// plausible lexicon. Only built for rank scoring.
    rank_of: Vec<u32>,
    alpha: f64,
    jitter: f64,
    seed: u64,
}

impl LocalOracle {
    pub fn new(model: &Model, secret: u32, scoring: Scoring) -> Self {
        Self::with_lexicon(model, secret, scoring, 0.0, 0)
    }

    /// `alpha` overrides the model's lexicon factor for the simulation (0 = keep it),
    /// `jitter` is the relative noise added to each simulated rank.
    pub fn with_lexicon(
        model: &Model,
        secret: u32,
        scoring: Scoring,
        jitter: f64,
        seed: u64,
    ) -> Self {
        let sims = model.dots(secret as usize);
        let mut sorted = sims.clone();
        sorted.sort_unstable_by(|a, b| b.total_cmp(a));
        let thresh = sorted[1000.min(sorted.len() - 1)];
        let mut rank_of = Vec::new();
        let mut alpha = 1.0;
        if let Scoring::Rank(m) = scoring {
            alpha = m.alpha;
            let lex = model.plausible_indices();
            let ranks = model.exact_ranks(secret as usize, &lex);
            rank_of = vec![u32::MAX; model.n()];
            for (k, &w) in lex.iter().enumerate() {
                rank_of[w as usize] = ranks[k];
            }
        }
        LocalOracle {
            secret,
            scoring,
            sims,
            thresh,
            rank_of,
            alpha,
            jitter,
            seed,
        }
    }

    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha;
    }

    /// Deterministic factor in [1 − jitter, 1 + jitter] for one (secret, guess) pair.
    fn noise(&self, idx: u32) -> f64 {
        if self.jitter <= 0.0 {
            return 1.0;
        }
        let mut h = self.seed ^ ((self.secret as u64) << 32) ^ idx as u64;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        let u = (h >> 11) as f64 / (1u64 << 53) as f64;
        1.0 + self.jitter * (2.0 * u - 1.0)
    }

    fn rank_score(&self, m: &RankModel, idx: u32) -> f64 {
        if idx == self.secret {
            return m.top;
        }
        match self.rank_of.get(idx as usize).copied() {
            Some(r) if r != u32::MAX && r > 0 => {
                let site = (self.alpha * r as f64 * self.noise(idx)).round();
                (m.top - site).max(m.floor)
            }
            // outside the plausible lexicon: far enough that the score is floored
            _ => m.floor,
        }
    }
}

impl Oracle for LocalOracle {
    fn score(&mut self, _word: &str, idx: u32) -> Result<ScoreResp> {
        if let Scoring::Rank(m) = self.scoring {
            return Ok(ScoreResp::Score {
                s: self.rank_score(&m, idx),
                p: None,
                solvers: None,
            });
        }
        let scale = match self.scoring {
            Scoring::Cosine { scale } => scale,
            Scoring::Rank(_) => unreachable!(),
        };
        let s = self.sims[idx as usize];
        let p = if idx == self.secret {
            Some(1000)
        } else if s >= self.thresh {
            let closer = self
                .sims
                .iter()
                .enumerate()
                .filter(|&(w, &x)| w as u32 != self.secret && x > s)
                .count() as i64;
            let p = 999 - closer;
            if p >= 1 { Some(p as u32) } else { None }
        } else {
            None
        };
        Ok(ScoreResp::Score {
            s: round_scale(s as f64, scale),
            p,
            solvers: None,
        })
    }
}

pub struct GameResult {
    pub guesses: u32,
    pub word: String,
    pub total_ms: u64,
}

/// How a score reads back to the player: degrees for Cémantix, the raw number and the
/// rank it stands for in a rank game.
fn score_text(scoring: Scoring, s: f64, p: Option<u32>) -> (String, &'static str) {
    match scoring {
        Scoring::Cosine { .. } => (format!("{:>7.2}°C", s * 100.0), emoji(s, p)),
        Scoring::Rank(m) => {
            let r = m.local_rank(s);
            let text = match r {
                Some(r) => format!("{s:>7} (rang ≈ {r:.0})"),
                None => format!("{s:>7} (hors du top {:.0})", m.floor_rank()),
            };
            (text, emoji_rank(r))
        }
    }
}

pub fn print_event(ev: &Event) {
    match ev {
        Event::Init {
            day,
            mode,
            game,
            plausible,
            opener,
            model,
        } => {
            println!(
                "▶ {} · {} (jour {}) — modèle {}, {} candidats plausibles, opener {}",
                game,
                mode,
                day.map(|d| d.to_string()).unwrap_or_else(|| "?".into()),
                model,
                plausible,
                opener.as_deref().unwrap_or("?")
            );
        }
        Event::NeedStart { message } => {
            if let Some(m) = message {
                println!("  {m}");
            }
        }
        Event::Probe {
            n,
            word,
            forced,
            entropy,
            buckets,
            probes,
            candidates,
            think_ms,
        } => {
            if *forced {
                println!(
                    "#{n:<2} {word:<20} votre choix : H={entropy:.2} bits ({buckets} tranches sur {candidates} candidats)"
                );
            } else if entropy.is_nan() {
                println!("#{n:<2} {word:<20} (opener précalculé)");
            } else {
                println!(
                    "#{n:<2} {word:<20} H={entropy:.2} bits ({buckets} tranches sur {candidates} candidats, {probes} sondes, {think_ms} ms)"
                );
            }
        }
        Event::Result {
            score_text,
            emoji,
            percentile,
            alive_before,
            alive_after,
            top,
            relaxed,
            tol,
            restricted,
            filter_ms,
            solvers,
            ..
        } => {
            let p = percentile.map(|p| format!(" ‰{p}")).unwrap_or_default();
            let v = solvers
                .map(|v| format!(" · {v} joueurs ont trouvé"))
                .unwrap_or_default();
            println!(
                "    → {} {}{}   candidats {} → {} ({filter_ms} ms){}{}",
                score_text,
                emoji,
                p,
                alive_before,
                alive_after,
                if *relaxed {
                    format!("  relâché : tol={tol}, plausible={restricted}")
                } else {
                    String::new()
                },
                v
            );
            if *alive_after <= 12 && *alive_after > 1 {
                println!("      restants : {}", top.join(", "));
            }
        }
        Event::Unknown { word, message } => println!("    ✗ {word} : {message} (retiré)"),
        Event::Solved {
            word,
            guesses,
            total_ms,
        } => {
            println!(
                "🥳 {word} trouvé en {guesses} coups ({:.1} s)",
                *total_ms as f64 / 1000.0
            )
        }
    }
}

/// Play one game. `forced` are guesses to play first (user-chosen openers, or the free
/// 🎁 hints a rank game hands out), then the solver takes over. `emit` receives every
/// event.
pub fn play_game(
    model: &Model,
    oracle: &mut dyn Oracle,
    scoring: Scoring,
    forced: &[u32],
    emit: &mut dyn FnMut(&Event),
) -> Result<GameResult> {
    if scoring.is_rank() && model.ranks.is_none() {
        bail!("ce jeu se joue aux rangs : lancez d'abord « build-ranks »");
    }
    let mut solver = Solver::with_scoring(model, scoring);
    let start = Instant::now();
    let mut n: u32 = 0;
    let mut unknown_streak = 0;
    let mut forced = forced.iter().copied();
    loop {
        if n >= 60 {
            bail!("abandon après {n} coups");
        }
        let t = Instant::now();
        let (ch, is_forced) = match forced.next() {
            Some(idx) => {
                let (entropy, buckets) = solver.evaluate(idx);
                (
                    Choice {
                        idx,
                        entropy,
                        buckets,
                        probes: 1,
                        candidates: solver.alive_count,
                    },
                    true,
                )
            }
            None => (solver.next_guess(), false),
        };
        let think_ms = t.elapsed().as_millis() as u64;
        let word = model.words[ch.idx as usize].clone();
        emit(&Event::Probe {
            n: n + 1,
            word: word.clone(),
            forced: is_forced,
            entropy: ch.entropy,
            buckets: ch.buckets,
            probes: ch.probes,
            candidates: ch.candidates,
            think_ms,
        });
        match oracle.score(&word, ch.idx)? {
            ScoreResp::Unknown(msg) => {
                unknown_streak += 1;
                if unknown_streak > 50 {
                    bail!("trop de mots inconnus d'affilée");
                }
                emit(&Event::Unknown { word, message: msg });
                solver.ban(ch.idx);
            }
            ScoreResp::Closed => bail!("ce puzzle n'est plus ouvert (réponse {{\"r\": true}})"),
            ScoreResp::Score { s, p, solvers } => {
                unknown_streak = 0;
                n += 1;
                if scoring.solved(s, p) {
                    let total_ms = start.elapsed().as_millis() as u64;
                    emit(&Event::Solved {
                        word: word.clone(),
                        guesses: n,
                        total_ms,
                    });
                    return Ok(GameResult {
                        guesses: n,
                        word,
                        total_ms,
                    });
                }
                let t = Instant::now();
                let info = solver.observe(Observation {
                    idx: ch.idx,
                    score: s,
                });
                let filter_ms = t.elapsed().as_millis() as u64;
                let top = solver
                    .top(12)
                    .iter()
                    .map(|&i| model.words[i as usize].clone())
                    .collect();
                let (text, icon) = score_text(scoring, s, p);
                emit(&Event::Result {
                    score_text: text,
                    emoji: icon,
                    percentile: p,
                    alive_before: info.before,
                    alive_after: info.after,
                    top,
                    tol: solver.tol_label(),
                    restricted: solver.restricted,
                    relaxed: info.relaxed,
                    filter_ms,
                    solvers,
                });
            }
        }
    }
}
