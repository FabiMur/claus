use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{Mutex, oneshot};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Minimal Language Server Protocol client speaking JSON-RPC 2.0 over the
/// server's stdio with Content-Length framing.
pub struct LspClient {
    stdin: Mutex<ChildStdin>,
    pending: Arc<Mutex<HashMap<i64, oneshot::Sender<Value>>>>,
    next_id: AtomicI64,
    opened: Mutex<HashSet<String>>,
    /// Work-done progress tokens currently active (server is indexing/analyzing).
    active_progress: Arc<Mutex<HashSet<String>>>,
    /// Latest diagnostics published per document uri.
    diagnostics: Arc<Mutex<HashMap<String, Value>>>,
    started_at: std::time::Instant,
    _child: Child,
}

impl LspClient {
    /// Spawn the server, run the initialize handshake and start the reader task.
    pub async fn start(command: &str, root: &Path) -> Result<Arc<Self>> {
        let mut parts = command.split_whitespace();
        let program = parts.next().context("empty LSP command")?;
        let mut child = Command::new(program)
            .args(parts)
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("spawning LSP server: {command}"))?;

        let stdin = child.stdin.take().context("no stdin on LSP server")?;
        let stdout = child.stdout.take().context("no stdout on LSP server")?;

        let client = Arc::new(Self {
            stdin: Mutex::new(stdin),
            pending: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicI64::new(1),
            opened: Mutex::new(HashSet::new()),
            active_progress: Arc::new(Mutex::new(HashSet::new())),
            diagnostics: Arc::new(Mutex::new(HashMap::new())),
            started_at: std::time::Instant::now(),
            _child: child,
        });

        tokio::spawn(Self::reader_task(Arc::clone(&client), stdout));

        let root_uri = uri_for(root);
        client
            .request(
                "initialize",
                json!({
                    "processId": std::process::id(),
                    "rootUri": root_uri,
                    "workspaceFolders": [{"uri": root_uri, "name": "workspace"}],
                    "capabilities": {
                        "window": {"workDoneProgress": true},
                        "textDocument": {
                            "hover": {"contentFormat": ["plaintext", "markdown"]},
                            "definition": {}, "references": {},
                            "publishDiagnostics": {}
                        }
                    }
                }),
            )
            .await?;
        client.notify("initialized", json!({})).await?;
        Ok(client)
    }

    async fn reader_task(client: Arc<Self>, stdout: tokio::process::ChildStdout) {
        let mut reader = BufReader::new(stdout);
        loop {
            let Ok(Some(message)) = read_frame(&mut reader).await else {
                break; // server exited or protocol error
            };
            // Response to one of our requests?
            if let Some(id) = message.get("id").and_then(|v| v.as_i64())
                && message.get("method").is_none()
            {
                if let Some(tx) = client.pending.lock().await.remove(&id) {
                    let payload = message
                        .get("result")
                        .cloned()
                        .unwrap_or_else(|| json!({"error": message.get("error").cloned().unwrap_or(Value::Null)}));
                    let _ = tx.send(payload);
                }
                continue;
            }
            // Server-to-client request: acknowledge with a null result so the
            // server (e.g. rust-analyzer's registerCapability) never stalls.
            if let (Some(id), Some(_)) = (message.get("id"), message.get("method")) {
                let reply = json!({"jsonrpc": "2.0", "id": id, "result": Value::Null});
                let _ = client.write_message(&reply).await;
            }
            match message.get("method").and_then(|m| m.as_str()) {
                // Track indexing/analysis progress so tools can wait for
                // readiness instead of returning misleading empty results.
                Some("$/progress") => {
                    let params = &message["params"];
                    let token = params["token"].to_string();
                    match params["value"]["kind"].as_str() {
                        Some("begin") => {
                            client.active_progress.lock().await.insert(token);
                        }
                        Some("end") => {
                            client.active_progress.lock().await.remove(&token);
                        }
                        _ => {}
                    }
                }
                Some("textDocument/publishDiagnostics") => {
                    let params = &message["params"];
                    if let Some(uri) = params["uri"].as_str() {
                        client
                            .diagnostics
                            .lock()
                            .await
                            .insert(uri.to_string(), params["diagnostics"].clone());
                    }
                }
                _ => {}
            }
        }
    }

    /// Wait until the server has no active work-done progress (indexing,
    /// analysis). Servers report progress shortly after startup, so also hold
    /// a short grace period before trusting an empty progress set.
    pub async fn wait_ready(&self, timeout: Duration) {
        let deadline = tokio::time::Instant::now() + timeout;
        let grace = self.started_at + Duration::from_secs(2);
        let mut quiet_checks = 0;
        while tokio::time::Instant::now() < deadline {
            let busy = !self.active_progress.lock().await.is_empty();
            if busy {
                quiet_checks = 0;
            } else if std::time::Instant::now() >= grace {
                quiet_checks += 1;
                // Require sustained quiet so a begin/end gap is not mistaken
                // for readiness.
                if quiet_checks >= 3 {
                    return;
                }
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    /// Latest diagnostics the server published for a document, if any.
    pub async fn diagnostics_for(&self, uri: &str) -> Option<Value> {
        self.diagnostics.lock().await.get(uri).cloned()
    }

    async fn write_message(&self, message: &Value) -> Result<()> {
        let body = serde_json::to_string(message)?;
        let framed = format!("Content-Length: {}\r\n\r\n{body}", body.len());
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(framed.as_bytes()).await?;
        stdin.flush().await?;
        Ok(())
    }

    pub async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);
        self.write_message(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))
            .await?;
        let result = tokio::time::timeout(REQUEST_TIMEOUT, rx)
            .await
            .map_err(|_| anyhow!("LSP request {method} timed out"))?
            .map_err(|_| anyhow!("LSP server dropped the response"))?;
        if let Some(error) = result.get("error").filter(|e| !e.is_null()) {
            bail!("LSP error for {method}: {error}");
        }
        Ok(result)
    }

    pub async fn notify(&self, method: &str, params: Value) -> Result<()> {
        self.write_message(&json!({"jsonrpc": "2.0", "method": method, "params": params}))
            .await
    }

    /// Send didOpen for the file once, so position-based requests work on it.
    pub async fn ensure_open(&self, path: &Path) -> Result<()> {
        let uri = uri_for(path);
        if !self.opened.lock().await.insert(uri.clone()) {
            return Ok(());
        }
        let text = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("reading {}", path.display()))?;
        let language_id = match path.extension().and_then(|e| e.to_str()) {
            Some("rs") => "rust",
            Some("py") => "python",
            Some(other) => other,
            None => "plaintext",
        };
        self.notify(
            "textDocument/didOpen",
            json!({"textDocument": {"uri": uri, "languageId": language_id, "version": 1, "text": text}}),
        )
        .await
    }
}

async fn read_frame(reader: &mut BufReader<tokio::process::ChildStdout>) -> Result<Option<Value>> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).await? == 0 {
            return Ok(None); // EOF
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some(value) = line.strip_prefix("Content-Length:") {
            content_length = value.trim().parse().ok();
        }
    }
    let length = content_length.context("missing Content-Length header")?;
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).await?;
    Ok(Some(serde_json::from_slice(&body)?))
}

pub fn uri_for(path: &Path) -> String {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    };
    format!("file://{}", absolute.display())
}

/// Render an LSP `Location | Location[] | LocationLink[]` result as `path:line`.
pub fn format_locations(result: &Value) -> String {
    let locations = match result {
        Value::Array(items) => items.clone(),
        Value::Null => Vec::new(),
        single => vec![single.clone()],
    };
    if locations.is_empty() {
        return "no results".to_string();
    }
    locations
        .iter()
        .filter_map(|loc| {
            let uri = loc["uri"].as_str().or(loc["targetUri"].as_str())?;
            let range = if loc["range"].is_object() {
                &loc["range"]
            } else {
                &loc["targetRange"]
            };
            let line = range["start"]["line"].as_u64().unwrap_or(0) + 1;
            let column = range["start"]["character"].as_u64().unwrap_or(0) + 1;
            Some(format!("{}:{line}:{column}", uri.trim_start_matches("file://")))
        })
        .collect::<Vec<_>>()
        .join("\n")
}
