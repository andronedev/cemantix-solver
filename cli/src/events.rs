//! Events produced by the game loop and printed on the console.

#[derive(Clone, Debug)]
pub enum Event {
    Init {
        day: Option<u32>,
        mode: String,
        game: String,
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
        /// The score, spelled out for a human: degrees for Cémantix, a rank elsewhere.
        score_text: String,
        emoji: &'static str,
        percentile: Option<u32>,
        alive_before: usize,
        alive_after: usize,
        top: Vec<String>,
        tol: String,
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

/// Cémantix's own thermometer: the ‰ rank when the server gives one, else the sign.
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

/// Same idea for a rank game, where the score *is* a rank: the closer the guess sits to
/// the secret, the hotter. `floored` marks the scores that only say "far away".
pub fn emoji_rank(local_rank: Option<f64>) -> &'static str {
    match local_rank {
        None => "🧊",
        Some(r) if r <= 3.0 => "😱",
        Some(r) if r <= 10.0 => "🔥",
        Some(r) if r <= 50.0 => "🥵",
        Some(r) if r <= 200.0 => "😎",
        Some(_) => "🥶",
    }
}
