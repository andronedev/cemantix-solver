//! Model download, parsing, caching and web export (CLI side, uses rayon).

use anyhow::{Context, Result, bail};
use cemantix_core::{
    PRIOR_RANK, SCALE_CEMANTIX, Vectors, dot, f16, is_plausible_word, partition_entropy,
};
use rayon::prelude::*;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub type Model = Vectors;

/// Model used by the server (lemmatised frWac, CBOW, 500 dims, cut10), checked with `verify`.
pub const DEFAULT_MODEL: &str = "frWac_no_postag_phrase_500_cbow_cut10";
pub const MODEL_BASE_URL: &str = "https://embeddings.net/embeddings/";
/// How many frequent plausible words are evaluated as opener candidates.
const OPENER_PROBES: usize = 3000;

pub fn load(data: &Path) -> Result<Model> {
    let t = Instant::now();
    if !data.join("vecs.f32").exists() && data.join("model.f16").exists() {
        return load_f16(data, t);
    }
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
    let mut model = Vectors::new(name, dim, words, vecs, PRIOR_RANK);
    model.opener = fs::read_to_string(data.join("opener.txt"))
        .ok()
        .and_then(|s| {
            s.lines()
                .next()
                .map(|l| l.split('\t').next().unwrap_or("").to_string())
        })
        .and_then(|w| model.lookup(&w));
    eprintln!(
        "model {}: {} words x {} dims loaded in {:.0} ms (opener: {})",
        model.name,
        n,
        dim,
        t.elapsed().as_secs_f64() * 1000.0,
        model
            .opener
            .map(|i| model.words[i as usize].as_str())
            .unwrap_or("none")
    );
    Ok(model)
}

/// Load the compact web export (plausible words only, float16) when the full
/// cache is absent: enough for `sim` and `play --dry-run`.
fn load_f16(data: &Path, t: Instant) -> Result<Model> {
    let words: Vec<String> = fs::read_to_string(data.join("words.txt"))?
        .lines()
        .map(str::to_owned)
        .collect();
    let bytes = fs::read(data.join("model.f16"))?;
    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(data.join("meta.json"))?)?;
    let dim = meta["dim"].as_u64().context("meta.json: dim")? as usize;
    if bytes.len() != words.len() * dim * 2 {
        bail!("model.f16 size mismatch");
    }
    let name = format!("{} (float16)", meta["model"].as_str().unwrap_or("?"));
    let n = words.len();
    let mut model = Vectors::from_f16(name, dim, words, &bytes, PRIOR_RANK);
    model.opener = meta["opener"].as_str().and_then(|w| model.lookup(w));
    eprintln!(
        "model {}: {} words x {} dims loaded in {:.0} ms",
        model.name,
        n,
        dim,
        t.elapsed().as_secs_f64() * 1000.0
    );
    Ok(model)
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
    let mut out = fs::File::create(&tmp)?;
    let mut buf = vec![0u8; 1 << 20];
    let mut done: u64 = 0;
    let mut last_pct = 0;
    loop {
        let k = reader.read(&mut buf)?;
        if k == 0 {
            break;
        }
        out.write_all(&buf[..k])?;
        done += k as u64;
        if total > 0 {
            let pct = (done * 100 / total) as u32;
            if pct >= last_pct + 5 {
                last_pct = pct;
                eprintln!("  {pct}% ({} MB)", done / 1_000_000);
            }
        }
    }
    drop(out);
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
            let (h, b) = partition_entropy(&vecs, dim, p as usize, &plausible, SCALE_CEMANTIX);
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

/// Export the plausible subset as float16 for the web page, and measure the
/// score error the compression introduces.
pub fn export_web(data: &Path, out: &Path) -> Result<()> {
    let model = load(data)?;
    let idx = model.plausible_indices();
    let dim = model.dim;
    fs::create_dir_all(out)?;
    let mut words = String::with_capacity(idx.len() * 10);
    let mut bytes = Vec::with_capacity(idx.len() * dim * 2);
    for &i in &idx {
        words.push_str(&model.words[i as usize]);
        words.push('\n');
        for &x in model.vec(i as usize) {
            bytes.extend_from_slice(&f16::from_f32(x).to_le_bytes());
        }
    }
    let dec = Vectors::from_f16(
        "check".into(),
        dim,
        words.lines().map(String::from).collect(),
        &bytes,
        PRIOR_RANK,
    );
    let step = (idx.len() / 400).max(1);
    let sample: Vec<usize> = (0..idx.len()).step_by(step).take(400).collect();
    let max_err = sample
        .par_iter()
        .map(|&a| {
            let mut m = 0f64;
            for &b in &sample {
                let orig = dot(model.vec(idx[a] as usize), model.vec(idx[b] as usize)) as f64;
                let approx = dot(dec.vec(a), dec.vec(b)) as f64;
                m = m.max((orig - approx).abs());
            }
            m
        })
        .reduce(|| 0f64, f64::max);
    let tol_level = if max_err <= 4e-5 { 0 } else { 1 };
    let opener = model.opener.map(|i| model.words[i as usize].clone());
    let meta = serde_json::json!({
        "n": idx.len(),
        "dim": dim,
        "model": model.name,
        "opener": opener,
        "tol_level": tol_level,
        "f16_max_score_error": max_err,
        "prior_rank": PRIOR_RANK,
    });
    fs::write(out.join("words.txt"), &words)?;
    fs::write(out.join("model.f16"), &bytes)?;
    fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    eprintln!(
        "export web: {} mots x {dim} dims → {:.1} Mo, erreur max f16 sur les scores {max_err:.2e} → tol_level {tol_level}",
        idx.len(),
        bytes.len() as f64 / 1e6
    );
    Ok(())
}
