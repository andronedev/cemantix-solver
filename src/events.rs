//! Events produced by the game loop and printed on the console.

#[derive(Clone, Debug)]
pub enum Event {
    Init {
        day: Option<u32>,
        mode: String,
        model: String,
        plausible: usize,
        opener: Option<String>,
    },
    /// The engine is waiting for the user to choose the starting word.
    NeedStart {
        message: Option<String>,
    },
    Probe {
        n: u32,
        word: String,
        forced: bool,
        entropy: f64,
        buckets: usize,
        probes: usize,
        candidates: usize,
        think_ms: u64,
    },
    Result {
        score: f64,
        percentile: Option<u32>,
        alive_before: usize,
        alive_after: usize,
        top: Vec<String>,
        tol: f64,
        restricted: bool,
        relaxed: bool,
        filter_ms: u64,
        solvers: Option<u64>,
    },
    Unknown {
        word: String,
        message: String,
    },
    Solved {
        word: String,
        guesses: u32,
        total_ms: u64,
    },
}

pub fn emoji(score: f64, p: Option<u32>) -> &'static str {
    match p {
        Some(1000) => "🥳",
        Some(x) if x >= 999 => "😱",
        Some(x) if x >= 990 => "🔥",
        Some(x) if x >= 900 => "🥵",
        Some(_) => "😎",
        None if score > 0.0 => "🥶",
        None => "🧊",
    }
}
