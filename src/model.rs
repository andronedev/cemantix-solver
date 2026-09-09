//! Word2vec model: download, parse, normalise, cache, opener.

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Model used by the server (lemmatised frWac, CBOW, 500 dims, cut10), checked with `verify`.
pub const DEFAULT_MODEL: &str = "frWac_no_postag_phrase_500_cbow_cut10";
pub const MODEL_BASE_URL: &str = "https://embeddings.net/embeddings/";
/// Words beyond this frequency rank are not considered plausible secrets.
pub const PRIOR_RANK: usize = 50_000;
/// How many frequent plausible words are evaluated as opener candidates.
const OPENER_PROBES: usize = 3000;

pub struct Model {
    pub name: String,
    pub dim: usize,
    pub words: Vec<String>,
    /// n * dim, L2-normalised rows.
    pub vecs: Vec<f32>,
    pub index: HashMap<String, u32>,
    pub plausible: Vec<bool>,
    pub opener: Option<u32>,
}

/// SIMD-friendly dot product (8 independent accumulators + scalar tail).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut tail = 0f32;
    for (x, y) in ra.iter().zip(rb) {
        tail += x * y;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail
}

#[inline]
pub fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

pub fn is_plausible_word(w: &str) -> bool {
    w.chars().count() >= 3
        && !w.starts_with('-')
        && !w.ends_with('-')
        && w.chars()
            .all(|c| (c.is_alphabetic() && c.is_lowercase()) || c == '-')
}

impl Model {
    pub fn n(&self) -> usize {
        self.words.len()
    }

    #[inline]
    pub fn vec(&self, i: usize) -> &[f32] {
        &self.vecs[i * self.dim..(i + 1) * self.dim]
    }

    pub fn lookup(&self, w: &str) -> Option<u32> {
        self.index.get(w).copied()
    }

    /// out[j] = cos(vec i, vec j) for every j (parallel matvec).
    pub fn dots_into(&self, i: usize, out: &mut [f32]) {
        let q = self.vec(i);
        let vecs = &self.vecs;
        let dim = self.dim;
        out.par_chunks_mut(2048)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let base = ci * 2048;
                for (k, o) in chunk.iter_mut().enumerate() {
                    let r = (base + k) * dim;
                    *o = dot(&vecs[r..r + dim], q);
                }
            });
    }

    pub fn dots(&self, i: usize) -> Vec<f32> {
        let mut out = vec![0f32; self.n()];
        self.dots_into(i, &mut out);
        out
    }

    pub fn plausible_indices(&self) -> Vec<u32> {
        (0..self.n() as u32)
            .filter(|&i| self.plausible[i as usize])
            .collect()
    }

    pub fn load(data: &Path) -> Result<Model> {
        let t = Instant::now();
        let words: Vec<String> = fs::read_to_string(data.join("words.txt"))
            .with_context(|| format!("missing {}/words.txt (run fetch-model)", data.display()))?
            .lines()
            .map(str::to_owned)
            .collect();
        let n = words.len();
        let raw = fs::read(data.join("vecs.f32")).context("missing vecs.f32 (run fetch-model)")?;
        if n == 0 || raw.len() % (n * 4) != 0 {
            bail!(
                "vecs.f32 size mismatch: {} bytes for {} words",
                raw.len(),
                n
            );
        }
        let dim = raw.len() / (n * 4);
        let vecs = bytes_to_f32(&raw);
        let name = fs::read_to_string(data.join("model.txt"))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "?".into());
        let mut index = HashMap::with_capacity(n);
        for (i, w) in words.iter().enumerate() {
            index.entry(w.clone()).or_insert(i as u32);
        }
        let plausible = words
            .iter()
            .enumerate()
            .map(|(i, w)| i < PRIOR_RANK && is_plausible_word(w))
            .collect();
        let opener = fs::read_to_string(data.join("opener.txt"))
            .ok()
            .and_then(|s| {
                s.lines()
                    .next()
                    .map(|l| l.split('\t').next().unwrap_or("").to_string())
            })
            .and_then(|w| index.get(&w).copied());
        eprintln!(
            "model {name}: {} words x {} dims loaded in {:.0} ms (opener: {})",
            n,
            dim,
            t.elapsed().as_secs_f64() * 1000.0,
            opener.map(|i| words[i as usize].as_str()).unwrap_or("none")
        );
        Ok(Model {
            name,
            dim,
            words,
            vecs,
            index,
            plausible,
            opener,
        })
    }
}

fn bytes_to_f32(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Download `<name>.bin` if missing, return its path.
pub fn ensure_downloaded(data: &Path, name: &str) -> Result<PathBuf> {
    fs::create_dir_all(data)?;
    let file = format!("{name}.bin");
    let path = data.join(&file);
    if fs::metadata(&path)
        .map(|m| m.len() > 50_000_000)
        .unwrap_or(false)
    {
        return Ok(path);
    }
    let url = format!("{MODEL_BASE_URL}{file}");
    eprintln!("downloading {url} ...");
    let resp = ureq::get(&url).call().context("download failed")?;
    let total: u64 = resp
        .header("Content-Length")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut reader = resp.into_reader();
    let tmp = data.join(format!("{file}.part"));
    let mut file = fs::File::create(&tmp)?;
    let mut buf = vec![0u8; 1 << 20];
    let mut done: u64 = 0;
    let mut last_pct = 0;
    loop {
        let k = reader.read(&mut buf)?;
        if k == 0 {
            break;
        }
        file.write_all(&buf[..k])?;
        done += k as u64;
        if total > 0 {
            let pct = (done * 100 / total) as u32;
            if pct >= last_pct + 5 {
                last_pct = pct;
                eprintln!("  {pct}% ({} MB)", done / 1_000_000);
            }
        }
    }
    drop(file);
    fs::rename(&tmp, &path)?;
    Ok(path)
}

/// Stream-parse the word2vec binary, keeping only words accepted by `keep`
/// (phrases with `_` are dropped: the server strips underscores anyway).
fn parse_bin(path: &Path, keep: impl Fn(&str) -> bool) -> Result<(Vec<String>, Vec<f32>, usize)> {
    use std::io::BufRead;
    let f = fs::File::open(path)?;
    let mut r = std::io::BufReader::with_capacity(1 << 20, f);
    let mut header = String::new();
    r.read_line(&mut header)?;
    let mut it = header.split_whitespace();
    let n: usize = it.next().context("header n")?.parse()?;
    let dim: usize = it.next().context("header dim")?.parse()?;
    if !(50..=2000).contains(&dim) {
        bail!("suspicious dimension {dim}");
    }
    let mut words = Vec::with_capacity(n / 2);
    let mut vecs: Vec<f32> = Vec::with_capacity(n / 2 * dim);
    let mut wbuf = Vec::with_capacity(64);
    let mut fbuf = vec![0u8; dim * 4];
    for i in 0..n {
        wbuf.clear();
        let k = r.read_until(b' ', &mut wbuf)?;
        if k == 0 {
            bail!("truncated file at word {i}");
        }
        if wbuf.last() == Some(&b' ') {
            wbuf.pop();
        }
        let start = wbuf.iter().position(|&b| b != b'\n').unwrap_or(wbuf.len());
        r.read_exact(&mut fbuf)
            .with_context(|| format!("truncated vector at word {i}"))?;
        let word = String::from_utf8_lossy(&wbuf[start..]);
        if keep(&word) {
            words.push(word.into_owned());
            vecs.extend(
                fbuf.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
        }
    }
    Ok((words, vecs, dim))
}

fn normalise(vecs: &mut [f32], dim: usize) {
    vecs.par_chunks_mut(dim).for_each(|row| {
        let norm = dot(row, row).sqrt();
        if norm > 0.0 {
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
    });
}

/// Entropy (bits) of the rounded-score partition induced by probe `p` over `targets`.
pub fn partition_entropy(vecs: &[f32], dim: usize, p: usize, targets: &[u32]) -> (f64, usize) {
    let q = &vecs[p * dim..(p + 1) * dim];
    let mut keys: Vec<i32> = targets
        .iter()
        .map(|&t| {
            let t = t as usize;
            (dot(&vecs[t * dim..(t + 1) * dim], q) as f64 * 10000.0).round() as i32
        })
        .collect();
    keys.sort_unstable();
    let total = keys.len() as f64;
    let mut h = 0f64;
    let mut buckets = 0usize;
    let mut i = 0;
    while i < keys.len() {
        let mut j = i + 1;
        while j < keys.len() && keys[j] == keys[i] {
            j += 1;
        }
        let pr = (j - i) as f64 / total;
        h -= pr * pr.log2();
        buckets += 1;
        i = j;
    }
    (h, buckets)
}

/// Build all cache files from `<name>.bin`.
pub fn build(data: &Path, name: &str, force: bool) -> Result<()> {
    let bin = ensure_downloaded(data, name)?;
    let same_model = fs::read_to_string(data.join("model.txt"))
        .map(|s| s.trim() == name)
        .unwrap_or(false);
    if !force && same_model && data.join("vecs.f32").exists() && data.join("opener.txt").exists() {
        eprintln!(
            "cache for {name} already built in {} (use --force to rebuild)",
            data.display()
        );
        return Ok(());
    }
    let t = Instant::now();
    let (words, mut vecs, dim) = parse_bin(&bin, |w| !w.contains('_'))?;
    eprintln!(
        "parsed {} single words x {dim} dims in {:.1} s",
        words.len(),
        t.elapsed().as_secs_f64()
    );
    normalise(&mut vecs, dim);
    fs::write(data.join("vecs.f32"), f32_to_bytes(&vecs))?;
    fs::write(data.join("words.txt"), words.join("\n") + "\n")?;
    fs::write(data.join("model.txt"), format!("{name}\n"))?;

    let t = Instant::now();
    let plausible: Vec<u32> = words
        .iter()
        .enumerate()
        .filter(|(i, w)| *i < PRIOR_RANK && is_plausible_word(w))
        .map(|(i, _)| i as u32)
        .collect();
    let probes: Vec<u32> = plausible.iter().copied().take(OPENER_PROBES).collect();
    let mut scored: Vec<(f64, usize, u32)> = probes
        .par_iter()
        .map(|&p| {
            let (h, b) = partition_entropy(&vecs, dim, p as usize, &plausible);
            (h, b, p)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.2.cmp(&b.2)));
    let mut out = String::new();
    for (h, b, p) in scored.iter().take(20) {
        out.push_str(&format!("{}\t{h:.4}\t{b}\n", words[*p as usize]));
    }
    fs::write(data.join("opener.txt"), out)?;
    eprintln!(
        "opener: {} ({:.3} bits over {} plausible words) computed in {:.1} s",
        words[scored[0].2 as usize],
        scored[0].0,
        plausible.len(),
        t.elapsed().as_secs_f64()
    );
    Ok(())
}
