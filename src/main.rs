mod api;
mod events;
mod game;
mod model;
mod solver;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use events::Event;
use game::{LocalOracle, Oracle, play_game, print_event};
use model::Model;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(
    name = "cemantix",
    about = "Solver Cémantix : trilatération dans l'espace word2vec"
)]
struct Cli {
    /// Data directory (model + caches)
    #[arg(long, global = true, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/data"))]
    data: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download the word2vec model (2 GB) and build the caches (vectors, opener)
    FetchModel {
        /// Model name on embeddings.net (without .bin)
        #[arg(long, default_value = model::DEFAULT_MODEL)]
        model: String,
        #[arg(long)]
        force: bool,
    },
    /// Check that the local model reproduces the server's scores (uses yesterday's word)
    Verify,
    /// Offline benchmark: play N games against the local model
    Sim {
        #[arg(short, long, default_value_t = 200)]
        n: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Secrets are drawn among plausible words with frequency rank below this
        #[arg(long, default_value_t = 30000)]
        prior: usize,
        #[arg(short, long)]
        verbose: bool,
    },
    /// Solve today's puzzle (or an offline secret)
    Play {
        /// Puzzle number (default: today's, read from the home page)
        #[arg(long)]
        day: Option<u32>,
        /// Play offline against this secret word instead of the API
        #[arg(long)]
        dry_run: Option<String>,
        /// Force the first guess(es) (repeatable: --start mot1 --start mot2)
        #[arg(long)]
        start: Vec<String>,
        /// Ask for the starting word on stdin
        #[arg(long)]
        choose: bool,
        /// Minimum delay between two API calls in ms
        #[arg(long, default_value_t = 300)]
        delay: u64,
    },
}

struct ApiOracle {
    day: u32,
    delay: Duration,
    last: Option<Instant>,
}

impl Oracle for ApiOracle {
    fn score(&mut self, word: &str, _idx: u32) -> Result<api::ScoreResp> {
        if let Some(t) = self.last {
            let el = t.elapsed();
            if el < self.delay {
                sleep(self.delay - el);
            }
        }
        self.last = Some(Instant::now());
        api::score(self.day, word)
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::FetchModel { model: name, force } => model::build(&cli.data, &name, force),
        Cmd::Verify => verify(&cli.data),
        Cmd::Sim {
            n,
            seed,
            prior,
            verbose,
        } => sim(&cli.data, n, seed, prior, verbose),
        Cmd::Play {
            day,
            dry_run,
            start,
            choose,
            delay,
        } => play(&cli.data, day, dry_run, start, choose, delay),
    }
}

fn verify(data: &Path) -> Result<()> {
    let model = Model::load(data)?;
    let home = api::fetch_home()?;
    let y = home
        .yesterday
        .context("mot d'hier introuvable sur la page")?;
    println!("jour {} — mot d'hier : {y}", home.day);
    let yi = model.lookup(&y).context("mot d'hier absent du modèle")? as usize;
    let near = api::nearby(&y)?;
    let sims = model.dots(yi);
    let (mut missing, mut bad, mut max_dev) = (0, 0, 0f64);
    for (w, _p, s100) in &near {
        match model.lookup(w) {
            None => missing += 1,
            Some(i) => {
                let local = (sims[i as usize] as f64 * 10000.0).round() / 100.0;
                let dev = (local - s100).abs();
                max_dev = max_dev.max(dev);
                if dev > 0.011 {
                    bad += 1;
                    if bad <= 10 {
                        println!("  écart {w}: serveur {s100:.2} / local {local:.2}");
                    }
                }
            }
        }
    }
    println!(
        "{} voisins : {} absents du modèle, {} écarts de score > 0.01 (max {:.4})",
        near.len(),
        missing,
        bad,
        max_dev
    );
    if bad == 0 && missing == 0 {
        println!("✓ le modèle local reproduit exactement les scores du serveur");
    }
    Ok(())
}

fn sim(data: &Path, n: usize, seed: u64, prior: usize, verbose: bool) -> Result<()> {
    let model = Model::load(data)?;
    let pool: Vec<u32> = model
        .plausible_indices()
        .into_iter()
        .filter(|&i| (i as usize) < prior)
        .collect();
    println!(
        "{} parties, secrets tirés parmi {} mots (rang < {prior})",
        n,
        pool.len()
    );
    let mut rng = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut hist = vec![0usize; 64];
    let mut total_ms = 0u64;
    let mut worst: Vec<(u32, String)> = Vec::new();
    let t = Instant::now();
    for g in 0..n {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let secret = pool[((rng >> 33) as usize) % pool.len()];
        let mut oracle = LocalOracle::new(&model, secret);
        let mut emit = |ev: &Event| {
            if verbose {
                print_event(ev);
            }
        };
        let res = play_game(&model, &mut oracle, &[], &mut emit)?;
        hist[(res.guesses as usize).min(63)] += 1;
        total_ms += res.total_ms;
        if res.guesses >= 6 {
            worst.push((res.guesses, res.word.clone()));
        }
        if !verbose && (g + 1) % 50 == 0 {
            eprintln!("  {}/{n} ({:.1} s)", g + 1, t.elapsed().as_secs_f64());
        }
    }
    let total: usize = hist.iter().sum();
    let mean = hist.iter().enumerate().map(|(k, &c)| k * c).sum::<usize>() as f64 / total as f64;
    let mut cum = 0;
    let mut median = 0;
    for (k, &c) in hist.iter().enumerate() {
        cum += c;
        if cum * 2 >= total {
            median = k;
            break;
        }
    }
    let max = hist.iter().rposition(|&c| c > 0).unwrap_or(0);
    println!("\ncoups  parties");
    for (k, &c) in hist.iter().enumerate() {
        if c > 0 {
            println!("{k:>5}  {c:>6}  {}", "█".repeat((c * 60 / total).max(1)));
        }
    }
    println!(
        "\nmoyenne {mean:.2}  médiane {median}  max {max}  —  {:.0} ms/partie",
        total_ms as f64 / total as f64
    );
    if !worst.is_empty() {
        worst.sort_by(|a, b| b.0.cmp(&a.0));
        let s: Vec<String> = worst
            .iter()
            .take(10)
            .map(|(g, w)| format!("{w} ({g})"))
            .collect();
        println!("pires : {}", s.join(", "));
    }
    Ok(())
}

fn play(
    data: &Path,
    day: Option<u32>,
    dry_run: Option<String>,
    start: Vec<String>,
    choose: bool,
    delay: u64,
) -> Result<()> {
    let model = Model::load(data)?;
    let mut forced: Vec<u32> = Vec::new();
    for w in &start {
        let w = w.trim().to_lowercase();
        forced.push(
            model
                .lookup(&w)
                .with_context(|| format!("« {w} » est absent du modèle"))?,
        );
    }
    let (day, mode, mut oracle): (Option<u32>, String, Box<dyn Oracle>) = match &dry_run {
        Some(w) => {
            let idx = model
                .lookup(w)
                .with_context(|| format!("« {w} » est absent du modèle"))?;
            (
                None,
                format!("hors ligne · secret « {w} »"),
                Box::new(LocalOracle::new(&model, idx)),
            )
        }
        None => {
            let d = match day {
                Some(d) => d,
                None => api::fetch_home()?.day,
            };
            (
                Some(d),
                "en direct".to_string(),
                Box::new(ApiOracle {
                    day: d,
                    delay: Duration::from_millis(delay),
                    last: None,
                }),
            )
        }
    };

    print_event(&Event::Init {
        day,
        mode,
        model: model.name.clone(),
        plausible: model.plausible.iter().filter(|&&p| p).count(),
        opener: model.opener.map(|i| model.words[i as usize].clone()),
    });

    if choose {
        let mut message: Option<String> = None;
        loop {
            print_event(&Event::NeedStart {
                message: message.clone(),
            });
            print!("mot de départ (vide = opener) : ");
            std::io::stdout().flush()?;
            let mut line = String::new();
            std::io::stdin().read_line(&mut line)?;
            let w = line.trim().to_lowercase();
            if w.is_empty() {
                break;
            }
            match model.lookup(&w) {
                Some(idx) => {
                    forced.insert(0, idx);
                    break;
                }
                None => {
                    message = Some(format!(
                        "« {w} » est absent du modèle, essayez un autre mot"
                    ))
                }
            }
        }
    }

    let mut emit = |ev: &Event| print_event(ev);
    play_game(&model, oracle.as_mut(), &forced, &mut emit).map(|_| ())
}
