//! Cémantix HTTP API.

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use std::thread::sleep;
use std::time::Duration;

pub const BASE: &str = "https://cemantix.certitudes.org";
const UA: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36";

pub struct Home {
    pub day: u32,
    pub yesterday: Option<String>,
}

fn strip_tags(s: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.trim().to_string()
}

pub fn fetch_home() -> Result<Home> {
    let html = ureq::get(BASE)
        .set("User-Agent", UA)
        .call()
        .context("GET home page")?
        .into_string()?;
    let key = "data-puzzle-number=\"";
    let day = html
        .find(key)
        .map(|i| &html[i + key.len()..])
        .and_then(|r| r.split('"').next())
        .context("puzzle number not found on home page")?
        .parse::<u32>()?;
    let yesterday = html.find("id=\"yesterday\"").and_then(|i| {
        let rest = &html[i..];
        let end = rest.find("</a>")?;
        let gt = rest.find('>')?;
        let w = strip_tags(&rest[gt + 1..end]);
        if w.is_empty() { None } else { Some(w) }
    });
    Ok(Home { day, yesterday })
}

#[derive(Deserialize, Debug)]
struct RawScore {
    s: Option<f64>,
    p: Option<u32>,
    v: Option<u64>,
    e: Option<String>,
    r: Option<bool>,
}

#[derive(Debug, Clone)]
pub enum ScoreResp {
    Score {
        s: f64,
        p: Option<u32>,
        solvers: Option<u64>,
    },
    Unknown(String),
    Closed,
}

fn with_retry<T>(mut f: impl FnMut() -> Result<T>) -> Result<T> {
    let mut last = None;
    for attempt in 0..4 {
        match f() {
            Ok(v) => return Ok(v),
            Err(e) => {
                eprintln!("  api error (attempt {}): {e:#}", attempt + 1);
                last = Some(e);
                sleep(Duration::from_millis(500 * (1 << attempt)));
            }
        }
    }
    Err(last.unwrap_or_else(|| anyhow!("unreachable")))
}

pub fn score(day: u32, word: &str) -> Result<ScoreResp> {
    let url = format!("{BASE}/score?n={day}");
    let raw: RawScore = with_retry(|| {
        let resp = ureq::post(&url)
            .set("User-Agent", UA)
            .set("Origin", BASE)
            .set("Referer", &format!("{BASE}/"))
            .send_form(&[("word", word)])
            .map_err(|e| match e {
                ureq::Error::Status(code, r) => {
                    anyhow!("HTTP {code}: {}", r.into_string().unwrap_or_default())
                }
                other => anyhow!(other),
            })?;
        Ok(resp.into_json::<RawScore>()?)
    })?;
    if let Some(e) = raw.e {
        return Ok(ScoreResp::Unknown(strip_tags(&e)));
    }
    if raw.r == Some(true) {
        return Ok(ScoreResp::Closed);
    }
    match raw.s {
        Some(s) => Ok(ScoreResp::Score {
            s,
            p: raw.p,
            solvers: raw.v,
        }),
        None => bail!("unexpected response: {raw:?}"),
    }
}

/// Neighbours of a revealed word: (word, percentile, score*100).
pub fn nearby(word: &str) -> Result<Vec<(String, u32, f64)>> {
    let v: serde_json::Value = with_retry(|| {
        Ok(ureq::post(&format!("{BASE}/nearby"))
            .set("User-Agent", UA)
            .set("Origin", BASE)
            .send_form(&[("word", word)])?
            .into_json()?)
    })?;
    let obj = v.as_object().context("nearby: not an object")?;
    let mut out = Vec::with_capacity(obj.len());
    for (w, arr) in obj {
        let a = arr.as_array().context("nearby: bad entry")?;
        let p = a.first().and_then(|x| x.as_u64()).unwrap_or(0) as u32;
        let s = a.get(1).and_then(|x| x.as_f64()).unwrap_or(0.0);
        out.push((w.clone(), p, s));
    }
    out.sort_by(|a, b| b.1.cmp(&a.1));
    Ok(out)
}
