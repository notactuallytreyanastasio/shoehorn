//! `shoehorn ui`: a local web page over the fit pipeline, for people who
//! would rather not learn the flags.
//!
//! The page drives the same binary: fits run as a `shoehorn fit` subprocess
//! with stdout/stderr streamed back to the browser, so the UI can never
//! drift from what the CLI does. State is one mutex; tiny_http handles
//! requests serially on the main thread while the fit runs in a worker.

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

const PAGE: &str = include_str!("ui.html");
const LLAMA_PORT: u16 = 8093;

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Idle,
    Running,
    Done,
    Failed,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Phase::Idle => "idle",
            Phase::Running => "running",
            Phase::Done => "done",
            Phase::Failed => "failed",
        }
    }
}

#[derive(Default)]
struct FitParams {
    ctx: u64,
    kv: String,
    /// original model input (path / repo id / URL), for eval --baseline
    spec: String,
}

struct State {
    phase: Phase,
    log: Vec<String>,
    /// path parsed from the subprocess's final "wrote <path> (<size>)" line
    output: Option<String>,
    fit_child: Option<Child>,
    cancelled: bool,
    /// --dry-run fit: succeeds without writing anything
    preview: bool,
    params: FitParams,
    serve_child: Option<Child>,
}

impl State {
    fn new() -> Self {
        State {
            phase: Phase::Idle,
            log: Vec::new(),
            output: None,
            fit_child: None,
            cancelled: false,
            preview: false,
            params: FitParams::default(),
            serve_child: None,
        }
    }
}

type Shared = Arc<Mutex<State>>;

pub fn serve_ui(port: u16, open_browser: bool) -> Result<()> {
    let server = tiny_http::Server::http(("127.0.0.1", port))
        .map_err(|e| anyhow!("could not bind 127.0.0.1:{port}: {e}"))?;
    let state: Shared = Arc::new(Mutex::new(State::new()));
    let url = format!("http://127.0.0.1:{port}");
    eprintln!("shoehorn ui: {url}");
    if open_browser {
        open_url(&url);
    }
    for mut req in server.incoming_requests() {
        let resp = route(&mut req, &state);
        let _ = req.respond(resp);
    }
    Ok(())
}

fn route(req: &mut tiny_http::Request, state: &Shared) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let url = req.url().to_string();
    let path = url.split('?').next().unwrap_or("");
    let post = req.method() == &tiny_http::Method::Post;
    match (post, path) {
        (false, "/") => html(PAGE),
        (false, "/api/info") => api_info(),
        (false, "/api/log") => api_log(&url, state),
        (false, "/api/search") => api_search(&url),
        (false, "/api/history") => json_resp(200, &json!({ "fits": read_history() })),
        (true, "/api/fit") => match api_fit(req, state) {
            Ok(v) => json_resp(200, &v),
            Err(e) => json_resp(400, &json!({ "error": e.to_string() })),
        },
        (true, "/api/serve") => match api_serve(req, state) {
            Ok(v) => json_resp(200, &v),
            Err(e) => json_resp(400, &json!({ "error": e.to_string() })),
        },
        (true, "/api/eval") => match api_eval(state) {
            Ok(v) => json_resp(200, &v),
            Err(e) => json_resp(400, &json!({ "error": e.to_string() })),
        },
        (true, "/api/discover") => match api_discover(state) {
            Ok(v) => json_resp(200, &v),
            Err(e) => json_resp(400, &json!({ "error": e.to_string() })),
        },
        (true, "/api/cancel") => {
            state.lock().unwrap().cancelled = true;
            json_resp(200, &json!({ "ok": true }))
        }
        _ => json_resp(404, &json!({ "error": "not found" })),
    }
}

fn html(body: &str) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    tiny_http::Response::from_string(body).with_header(
        tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap(),
    )
}

fn json_resp(code: u32, v: &Value) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    tiny_http::Response::from_string(v.to_string())
        .with_status_code(code as u16)
        .with_header(
            tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
        )
}

fn api_info() -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let probe = crate::vram::probe();
    let llama = Command::new("llama-server").arg("--version").output().is_ok();
    json_resp(
        200,
        &json!({
            "vram": probe.as_ref().map(|(b, ..)| b),
            "device": probe.as_ref().map(|(_, n, _)| n),
            "llama": llama,
        }),
    )
}

fn api_log(url: &str, state: &Shared) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let from: usize = url
        .split_once("from=")
        .and_then(|(_, v)| v.split('&').next().unwrap_or("").parse().ok())
        .unwrap_or(0);
    let s = state.lock().unwrap();
    let lines: Vec<&String> = s.log.iter().skip(from).collect();
    json_resp(
        200,
        &json!({
            "state": s.phase.name(),
            "lines": lines,
            "total": s.log.len(),
            // progress lines are collapsed in place, so the tail can change
            // without total growing; the client reads it from here
            "last": s.log.last(),
            "output": s.output,
            "preview": s.preview,
        }),
    )
}

/// Proxy a model search to Hugging Face so the page needs no cross-origin
/// requests. The query arrives percent-encoded and is passed through as-is.
fn api_search(url: &str) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    let q: String = url
        .split_once("q=")
        .map(|(_, v)| v.split('&').next().unwrap_or(""))
        .unwrap_or("")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '%' | '+' | '.' | '_' | '/' | '-'))
        .take(80)
        .collect();
    if q.len() < 2 {
        return json_resp(200, &json!({ "models": [] }));
    }
    let api = format!("https://huggingface.co/api/models?search={q}&filter=gguf&limit=8");
    let ids: Vec<String> = Command::new("curl")
        .args(["-sL", "--fail", "--max-time", "5", &api])
        .output()
        .ok()
        .and_then(|o| serde_json::from_slice::<Value>(&o.stdout).ok())
        .and_then(|v| {
            v.as_array().map(|arr| {
                arr.iter().filter_map(|m| m["id"].as_str().map(String::from)).collect()
            })
        })
        .unwrap_or_default();
    json_resp(200, &json!({ "models": ids }))
}

fn history_path() -> Option<std::path::PathBuf> {
    crate::fetch::cache_dir().ok().map(|d| d.join("fits.json"))
}

fn read_history() -> Vec<Value> {
    let list: Vec<Value> = history_path()
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    // A fit whose file was deleted isn't servable; don't offer it.
    list.into_iter()
        .filter(|e| e["output"].as_str().is_some_and(|p| std::path::Path::new(p).exists()))
        .collect()
}

fn record_fit(output: &str, ctx: u64, kv: &str, spec: &str) {
    let Some(path) = history_path() else { return };
    let abs = std::fs::canonicalize(output)
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| output.to_string());
    let mut list = read_history();
    list.retain(|e| e["output"].as_str() != Some(abs.as_str()));
    list.insert(0, json!({ "output": abs, "ctx": ctx, "kv": kv, "model": spec }));
    list.truncate(12);
    let _ = std::fs::write(path, serde_json::to_vec(&list).unwrap_or_default());
}

fn api_fit(req: &mut tiny_http::Request, state: &Shared) -> Result<Value> {
    let mut body = String::new();
    req.as_reader().read_to_string(&mut body)?;
    let v: Value = serde_json::from_str(&body).context("bad request body")?;
    let model = v["model"].as_str().unwrap_or("").trim().to_string();
    if model.is_empty() {
        return Err(anyhow!("pick a model first"));
    }
    let ctx = v["ctx"].as_u64().unwrap_or(8192);
    let kv = v["kv"].as_str().unwrap_or("f16").to_string();
    let budget = v["budget"].as_str().unwrap_or("").trim().to_string();
    let calibrate = v["calibrate"].as_bool().unwrap_or(false);
    let preview = v["preview"].as_bool().unwrap_or(false);

    let mut s = state.lock().unwrap();
    if s.phase == Phase::Running {
        return Err(anyhow!("a fit is already running"));
    }
    // A new fit invalidates whatever the old llama-server was serving.
    if let Some(mut old) = s.serve_child.take() {
        let _ = old.kill();
    }

    let exe = std::env::current_exe()?;
    let mut cmd = Command::new(exe);
    cmd.arg("fit")
        .arg(&model)
        .args(["--ctx", &ctx.to_string(), "--kv", &kv])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if !budget.is_empty() {
        cmd.args(["--budget", &budget]);
    }
    if calibrate && !preview {
        cmd.arg("--calibrate");
    }
    if preview {
        cmd.arg("--dry-run");
    }
    let child = cmd.spawn().context("starting the fit")?;

    s.phase = Phase::Running;
    s.log = vec![format!(
        "$ shoehorn fit {model} --ctx {ctx} --kv {kv}{}",
        if preview { " --dry-run" } else { "" }
    )];
    s.output = None;
    s.cancelled = false;
    s.preview = preview;
    s.params = FitParams { ctx, kv, spec: model };
    attach_job(s, child, state);
    Ok(json!({ "ok": true }))
}

/// Run the fitted model and its original source through `shoehorn eval` to
/// put a perplexity number on what the fit cost.
fn api_eval(state: &Shared) -> Result<Value> {
    let mut s = state.lock().unwrap();
    if s.phase == Phase::Running {
        return Err(anyhow!("something is already running"));
    }
    let output = s.output.clone().ok_or_else(|| anyhow!("nothing fitted yet"))?;
    let spec = s.params.spec.clone();
    if spec.is_empty() {
        return Err(anyhow!("the original model for this fit is unknown"));
    }
    // Both models get loaded onto the GPU in turn; make room first.
    if let Some(mut old) = s.serve_child.take() {
        let _ = old.kill();
    }
    let exe = std::env::current_exe()?;
    let mut cmd = Command::new(exe);
    cmd.arg("eval")
        .args(["-m", &output, "--baseline", &spec])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let child = cmd.spawn().context("starting the eval")?;
    s.phase = Phase::Running;
    s.log = vec![format!("$ shoehorn eval -m {output} --baseline {spec}")];
    s.cancelled = false;
    s.preview = false;
    attach_job(s, child, state);
    Ok(json!({ "ok": true }))
}

/// Rank fit-worthy models for this machine. Blocks for the scan (a few
/// seconds of Hugging Face calls), so it's refused while a job is running —
/// the log poller shares this thread.
fn api_discover(state: &Shared) -> Result<Value> {
    if state.lock().unwrap().phase == Phase::Running {
        return Err(anyhow!("wait for the current job to finish"));
    }
    let exe = std::env::current_exe()?;
    let out = Command::new(exe)
        .args(["discover", "--json"])
        .stdin(Stdio::null())
        .output()
        .context("running discover")?;
    if !out.status.success() {
        return Err(anyhow!("discovery failed: {}", String::from_utf8_lossy(&out.stderr)));
    }
    serde_json::from_slice(&out.stdout).context("parsing discovery output")
}

/// Wire a spawned subprocess into the shared job state: stream pumps for
/// both pipes plus the completion watcher.
fn attach_job(mut s: std::sync::MutexGuard<'_, State>, mut child: Child, state: &Shared) {
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    s.fit_child = Some(child);
    drop(s);

    for stream in [Box::new(stdout) as Box<dyn Read + Send>, Box::new(stderr)] {
        let st = state.clone();
        std::thread::spawn(move || pump_lines(stream, &st));
    }
    let st = state.clone();
    std::thread::spawn(move || watch_fit(&st));
}

/// Feed a subprocess stream into the shared log, splitting on \n and \r so
/// curl's carriage-return progress meter surfaces as lines too.
fn pump_lines(mut stream: Box<dyn Read + Send>, state: &Shared) {
    let mut buf = [0u8; 4096];
    let mut line = Vec::new();
    loop {
        let n = match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        for &b in &buf[..n] {
            if b == b'\n' || b == b'\r' {
                if !line.is_empty() {
                    push_line(state, String::from_utf8_lossy(&line).into_owned());
                    line.clear();
                }
            } else {
                line.push(b);
            }
        }
    }
    if !line.is_empty() {
        push_line(state, String::from_utf8_lossy(&line).into_owned());
    }
}

/// A curl --progress-bar line: hashes, spaces, and a percentage.
fn is_progress(line: &str) -> bool {
    line.contains('%') && line.chars().all(|c| matches!(c, '#' | ' ' | '.' | '%' | '0'..='9'))
}

fn push_line(state: &Shared, text: String) {
    let mut s = state.lock().unwrap();
    if let Some(rest) = text.strip_prefix("wrote ")
        && let Some((path, _)) = rest.rsplit_once(" (")
    {
        s.output = Some(path.to_string());
    }
    // Collapse download progress in place so a 60 GB pull is one log line,
    // not thousands. api_log's "last" field carries the updates.
    if is_progress(&text)
        && let Some(last) = s.log.last_mut()
        && is_progress(last)
    {
        *last = text;
        return;
    }
    s.log.push(text);
}

/// Poll the fit subprocess to completion (or kill it on cancel), then flip
/// the phase. Polling instead of wait() keeps the state lock uncontended.
fn watch_fit(state: &Shared) {
    loop {
        std::thread::sleep(std::time::Duration::from_millis(200));
        let mut s = state.lock().unwrap();
        let cancelled = s.cancelled;
        let Some(child) = s.fit_child.as_mut() else { return };
        if cancelled {
            let _ = child.kill();
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                let _ = s.fit_child.take();
                if s.cancelled {
                    s.phase = Phase::Idle;
                    s.log.push("stopped".into());
                } else if status.success() && (s.preview || s.output.is_some()) {
                    s.phase = Phase::Done;
                    if let Some(out) = &s.output {
                        record_fit(out, s.params.ctx, &s.params.kv, &s.params.spec);
                    }
                } else {
                    s.phase = Phase::Failed;
                }
                return;
            }
            Ok(None) => {}
            Err(_) => {
                let _ = s.fit_child.take();
                s.phase = Phase::Failed;
                return;
            }
        }
    }
}

/// Serve the current fit, or — when the body names one — a fit from history.
fn api_serve(req: &mut tiny_http::Request, state: &Shared) -> Result<Value> {
    let mut body = String::new();
    req.as_reader().read_to_string(&mut body).ok();
    let v: Value = serde_json::from_str(&body).unwrap_or(Value::Null);

    let mut s = state.lock().unwrap();
    let (output, ctx, kv) = match v["output"].as_str() {
        Some(o) => (
            o.to_string(),
            v["ctx"].as_u64().unwrap_or(8192),
            v["kv"].as_str().unwrap_or("f16").to_string(),
        ),
        None => (
            s.output.clone().ok_or_else(|| anyhow!("nothing fitted yet"))?,
            s.params.ctx,
            s.params.kv.clone(),
        ),
    };
    if !std::path::Path::new(&output).exists() {
        return Err(anyhow!("{output} no longer exists"));
    }
    if let Some(mut old) = s.serve_child.take() {
        let _ = old.kill();
    }
    let mut cmd = Command::new("llama-server");
    cmd.arg("-m")
        .arg(&output)
        .args(["-c", &ctx.to_string(), "-ngl", "99", "--port", &LLAMA_PORT.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if kv != "f16" {
        cmd.args(["--cache-type-k", &kv, "--cache-type-v", &kv]);
    }
    s.serve_child = Some(cmd.spawn().context("starting llama-server (is llama.cpp installed?)")?);
    Ok(json!({ "url": format!("http://127.0.0.1:{LLAMA_PORT}") }))
}

/// Best-effort: pop the page open in the default browser.
fn open_url(url: &str) {
    let (cmd, args): (&str, &[&str]) = if cfg!(target_os = "macos") {
        ("open", &[])
    } else if cfg!(target_os = "windows") {
        ("cmd", &["/c", "start"])
    } else {
        ("xdg-open", &[])
    };
    let _ = Command::new(cmd).args(args).arg(url).spawn();
}
