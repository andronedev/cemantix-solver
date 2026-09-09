mod api;
mod events;
mod game;
mod model;

use anyhow::{Context, Result, bail};
use cemantix_core::{RankModel, Scoring};
use clap::{Parser, Subcommand};
use events::Event;
use game::{LocalOracle, Oracle, play_game, print_event};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(
    name = "cemantix",
    about = "Solver Cémantix / QuelMot : trilatération et contraintes de rang dans l'espace word2vec"
)]
struct Cli {
    /// Data directory (model + caches)
    #[arg(long, global = true, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../data"))]
    data: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

/// How the chosen game scores a guess.
#[derive(Clone, clap::Args)]
struct GameOpts {
    /// cemantix (cosinus à 4 décimales) ou quelmot (score = 1000 − rang)
    #[arg(long, default_value = "cemantix")]
    game: String,
    /// Rapport entre le lexique du site et le nôtre (jeux à rangs)
    #[arg(long)]
    alpha: Option<f64>,
    /// Demi-largeur multiplicative de la fenêtre de rangs acceptée
    #[arg(long)]
    window: Option<f64>,
}

impl GameOpts {
    fn scoring(&self) -> Result<Scoring> {
        match self.game.as_str() {
            "cemantix" => Ok(Scoring::CEMANTIX),
            "quelmot" => {
                let mut m = RankModel::QUELMOT;
                if let Some(a) = self.alpha {
                    if a <= 0.0 {
                        bail!("--alpha doit être > 0");
                    }
                    m.alpha = a;
                }
                if let Some(w) = self.window {
                    if w <= 1.0 {
                        bail!("--window doit être > 1");
                    }
                    m.window = w;
                }
                Ok(Scoring::Rank(m))
            }
            other => bail!("jeu inconnu « {other} » (cemantix ou quelmot)"),
        }
    }
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
    /// Tabulate the neighbour ranks of every plausible word (needed by QuelMot)
    BuildRanks {
        #[arg(long)]
        force: bool,
    },
    /// Check that the local model reproduces the server's scores (uses yesterday's word)
    Verify,
    /// Check the rank model against real QuelMot answers, once the day's word is known
    CheckRanks {
        /// The secret of that day
        #[arg(long)]
        secret: String,
        /// Repeatable: --obs mot=score, the scores the site gave (hints included)
        #[arg(long = "obs", value_name = "MOT=SCORE", required = true)]
        obs: Vec<String>,
        #[command(flatten)]
        opts: GameOpts,
    },
    /// Export the plausible words as float16 + metadata for the web page (docs/)
    ExportWeb {
        #[arg(long, default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../docs"))]
        out: PathBuf,
    },
    /// Offline benchmark: play N games against the local model
    Sim {
        #[arg(short, long, default_value_t = 200)]
        n: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Secrets are drawn among plausible words with frequency rank below this
        #[arg(long, default_value_t = 30000)]
        prior: usize,
        #[command(flatten)]
        opts: GameOpts,
        /// Lexicon factor used by the simulated site (defaults to --alpha)
        #[arg(long)]
        sim_alpha: Option<f64>,
        /// Relative noise on each simulated rank: the two lexicons do not dilate evenly
        #[arg(long, default_value_t = 0.0)]
        jitter: f64,
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
        #[command(flatten)]
        opts: GameOpts,
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
        Cmd::BuildRanks { force } => model::build_ranks(&cli.data, force),
        Cmd::Verify => verify(&cli.data),
        Cmd::CheckRanks { secret, obs, opts } => check_ranks(&cli.data, &secret, &obs, &opts),
        Cmd::ExportWeb { out } => model::export_web(&cli.data, &out),
        Cmd::Sim {
            n,
            seed,
            prior,
            opts,
            sim_alpha,
            jitter,
            verbose,
        } => sim(&cli.data, n, seed, prior, &opts, sim_alpha, jitter, verbose),
        Cmd::Play {
            day,
            dry_run,
            start,
            choose,
            delay,
            opts,
        } => play(&cli.data, day, dry_run, start, choose, delay, &opts),
    }
}

fn verify(data: &Path) -> Result<()> {
    let model = model::load(data)?;
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

/// Replay real answers against a known secret: does `site rank ≈ alpha × local rank`
/// still hold, and does the tabulated estimate match the counted rank?
fn check_ranks(data: &Path, secret: &str, obs: &[String], opts: &GameOpts) -> Result<()> {
    let m = match opts.scoring()? {
        Scoring::Rank(m) => m,
        Scoring::Cosine { .. } => {
            bail!("check-ranks ne concerne que les jeux à rangs (--game quelmot)")
        }
    };
    let model = model::load(data)?;
    let table = model
        .ranks
        .as_ref()
        .context("table de rangs absente : lancez d'abord « build-ranks »")?;
    let secret = secret.trim().to_lowercase();
    let si = model
        .lookup(&secret)
        .with_context(|| format!("« {secret} » est absent du modèle"))? as usize;
    let lex = model.plausible_indices();
    let ranks = model.exact_ranks(si, &lex);
    let sims = model.dots(si);

    println!(
        "secret « {secret} », lexique local de {} mots, α attendu {:.2}",
        lex.len(),
        m.alpha
    );
    println!(
        "{:<18} {:>7} {:>10} {:>11} {:>10} {:>7}",
        "mot", "score", "rang site", "rang local", "estimé", "α"
    );
    let mut alphas: Vec<f64> = Vec::new();
    for spec in obs {
        let (w, s) = spec
            .split_once('=')
            .with_context(|| format!("attendu mot=score, reçu « {spec} »"))?;
        let w = w.trim().to_lowercase();
        let score: f64 = s.trim().parse().with_context(|| format!("score « {s} »"))?;
        let Some(i) = model.lookup(&w) else {
            println!("{w:<18} {score:>7}   absent du modèle");
            continue;
        };
        // The rank to compare is the guess's among the secret's neighbours, the very
        // quantity the site scores: rank is not symmetric, the guess's own row answers
        // the mirror question and is off by a factor of four here.
        let est = table.rank(si, sims[i as usize]);
        let local = lex.binary_search(&i).ok().map(|k| ranks[k]);
        let (local_txt, est_txt) = (
            local
                .map(|r| r.to_string())
                .unwrap_or_else(|| "hors lexique".into()),
            est.map(|r| format!("{r:.0}")).unwrap_or_else(|| "—".into()),
        );
        match (m.local_rank(score), local) {
            (Some(_), Some(r)) if r > 0 => {
                let a = (m.top - score) / r as f64;
                alphas.push(a);
                println!(
                    "{w:<18} {score:>7} {:>10.0} {local_txt:>11} {est_txt:>10} {a:>7.2}",
                    m.top - score
                );
            }
            (Some(_), _) => println!(
                "{w:<18} {score:>7} {:>10.0} {local_txt:>11} {est_txt:>10} {:>7}",
                m.top - score,
                "?"
            ),
            (None, _) => println!(
                "{w:<18} {score:>7} {:>10} {local_txt:>11} {est_txt:>10} {:>7}",
                "plafonné", "—"
            ),
        }
    }
    if alphas.is_empty() {
        println!("\naucun score non plafonné : rien à ajuster");
        return Ok(());
    }
    let mut sorted = alphas.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let mean = alphas.iter().sum::<f64>() / alphas.len() as f64;
    println!(
        "\nα médian {median:.2}, moyen {mean:.2}, étendue {:.2}–{:.2} sur {} observations",
        sorted[0],
        sorted[sorted.len() - 1],
        alphas.len()
    );
    let off = alphas
        .iter()
        .filter(|a| **a > m.alpha * m.window || **a < m.alpha / m.window)
        .count();
    if off > 0 {
        println!(
            "{off} observation(s) hors de la fenêtre ×÷{:.2} autour de α={:.2}",
            m.window, m.alpha
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn sim(
    data: &Path,
    n: usize,
    seed: u64,
    prior: usize,
    opts: &GameOpts,
    sim_alpha: Option<f64>,
    jitter: f64,
    verbose: bool,
) -> Result<()> {
    let scoring = opts.scoring()?;
    let model = model::load(data)?;
    let pool: Vec<u32> = model
        .plausible_indices()
        .into_iter()
        .filter(|&i| (i as usize) < prior)
        .collect();
    match scoring {
        Scoring::Cosine { scale } => println!(
            "{n} parties ({}, scores arrondis à 1/{scale}), secrets tirés parmi {} mots (rang < {prior})",
            opts.game,
            pool.len()
        ),
        Scoring::Rank(m) => println!(
            "{n} parties ({}, score = {:.0} − rang, plancher {:.0}), solver α={:.2} fenêtre ×÷{:.2}, \
             site simulé α={:.2} bruit ±{:.0} %, secrets tirés parmi {} mots (rang < {prior})",
            opts.game,
            m.top,
            m.floor,
            m.alpha,
            m.window,
            sim_alpha.unwrap_or(m.alpha),
            jitter * 100.0,
            pool.len()
        ),
    }
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
        let mut oracle = LocalOracle::with_lexicon(&model, secret, scoring, jitter, seed);
        if let Some(a) = sim_alpha {
            oracle.set_alpha(a);
        }
        let mut emit = |ev: &Event| {
            if verbose {
                print_event(ev);
            }
        };
        let res = play_game(&model, &mut oracle, scoring, &[], &mut emit)?;
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

#[allow(clippy::too_many_arguments)]
fn play(
    data: &Path,
    day: Option<u32>,
    dry_run: Option<String>,
    start: Vec<String>,
    choose: bool,
    delay: u64,
    opts: &GameOpts,
) -> Result<()> {
    let scoring = opts.scoring()?;
    let model = model::load(data)?;
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
                Box::new(LocalOracle::new(&model, idx, scoring)),
            )
        }
        None => {
            if scoring.is_rank() {
                bail!(
                    "l'API de {} exige un jeton d'authentification : jouez via la page web, \
                     ou hors ligne avec --dry-run <mot>",
                    opts.game
                );
            }
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

    let opener = if scoring.is_rank() {
        model.rank_opener
    } else {
        model.opener
    };
    print_event(&Event::Init {
        day,
        mode,
        game: opts.game.clone(),
        model: model.name.clone(),
        plausible: model.plausible.iter().filter(|&&p| p).count(),
        opener: opener.map(|i| model.words[i as usize].clone()),
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
    play_game(&model, oracle.as_mut(), scoring, &forced, &mut emit).map(|_| ())
}
