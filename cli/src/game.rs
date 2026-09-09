//! Game loop shared by the live player and the offline simulator.

use crate::api::ScoreResp;
use crate::events::{Event, emoji};
use crate::model::Model;
use anyhow::{Result, bail};
use cemantix_core::{Choice, Observation, Solver, partition_entropy, round_scale};
use std::time::Instant;

pub trait Oracle {
    fn score(&mut self, word: &str, idx: u32) -> Result<ScoreResp>;
}

/// Offline oracle reproducing the server's scoring from the local model.
pub struct LocalOracle {
    pub secret: u32,
    pub scale: f64,
    sims: Vec<f32>,
    thresh: f32,
}

impl LocalOracle {
    pub fn new(model: &Model, secret: u32, scale: f64) -> Self {
        let sims = model.dots(secret as usize);
        let mut sorted = sims.clone();
        sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        let thresh = sorted[1000.min(sorted.len() - 1)];
        LocalOracle {
            secret,
            scale,
            sims,
            thresh,
        }
    }
}

impl Oracle for LocalOracle {
    fn score(&mut self, _word: &str, idx: u32) -> Result<ScoreResp> {
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
            s: round_scale(s as f64, self.scale),
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

pub fn print_event(ev: &Event) {
    match ev {
        Event::Init {
            day,
            mode,
            plausible,
            opener,
            model,
        } => {
            println!(
                "▶ {} (jour {}) — modèle {}, {} candidats plausibles, opener {}",
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
            score,
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
                "    → {:>7.2}°C {}{}   candidats {} → {} ({filter_ms} ms){}{}",
                score * 100.0,
                emoji(*score, *percentile),
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

/// Play one game. `forced` are guesses to play first (user-chosen openers), then the
/// solver takes over. `emit` receives every event.
pub fn play_game(
    model: &Model,
    oracle: &mut dyn Oracle,
    scale: f64,
    forced: &[u32],
    emit: &mut dyn FnMut(&Event),
) -> Result<GameResult> {
    let mut solver = Solver::with_scale(model, scale);
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
                let cands = solver.candidates();
                let (entropy, buckets) =
                    partition_entropy(&model.vecs, model.dim, idx as usize, &cands, scale);
                (
                    Choice {
                        idx,
                        entropy,
                        buckets,
                        probes: 1,
                        candidates: cands.len(),
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
                if p == Some(1000) || s >= 0.99995 {
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
                emit(&Event::Result {
                    score: s,
                    percentile: p,
                    alive_before: info.before,
                    alive_after: info.after,
                    top,
                    tol: solver.tol(),
                    restricted: solver.restricted,
                    relaxed: info.relaxed,
                    filter_ms,
                    solvers,
                });
            }
        }
    }
}
