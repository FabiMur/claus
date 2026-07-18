use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{Mutex, oneshot};

const PROTOCOL_VERSION: &str = "2025-06-18";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// MCP server declarations, loaded from `.claus/mcp.json`:
/// `{"servers": {"name": {"command": "npx", "args": ["-y", "..."]}}}`
#[derive(Debug, Default, Deserialize)]
pub struct McpConfig {
    #[serde(default)]
    pub servers: HashMap<String, ServerConfig>,
}

#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
}

impl McpConfig {
    pub fn load(project_root: &Path) -> Self {
        let path = project_root.join(".claus/mcp.json");
        std::fs::read_to_string(path)
            .ok()
            .and_then(|content| serde_json::from_str(&content).ok())
            .unwrap_or_default()
    }
}

/// A tool advertised by an MCP server via `tools/list`.
#[derive(Clone, Debug, Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// Client for one MCP server speaking JSON-RPC 2.0 over newline-delimited
/// stdio, per the Model Context Protocol stdio transport.
pub struct McpClient {
    stdin: Mutex<ChildStdin>,
    pending: Arc<Mutex<HashMap<i64, oneshot::Sender<Value>>>>,
    next_id: AtomicI64,
    _child: Child,
}

impl McpClient {
    pub async fn connect(config: &ServerConfig) -> Result<(Arc<Self>, Vec<McpToolInfo>)> {
        let mut child = Command::new(&config.command)
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("spawning MCP server: {}", config.command))?;

        let stdin = child.stdin.take().context("no stdin on MCP server")?;
        let stdout = child.stdout.take().context("no stdout on MCP server")?;

        let client = Arc::new(Self {
            stdin: Mutex::new(stdin),
            pending: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicI64::new(1),
            _child: child,
        });
        tokio::spawn(Self::reader_task(Arc::clone(&client), stdout));

        client
            .request(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "claus", "version": env!("CARGO_PKG_VERSION")}
                }),
            )
            .await?;
        client.notify("notifications/initialized", json!({})).await?;

        let tools_result = client.request("tools/list", json!({})).await?;
        let tools = serde_json::from_value(tools_result["tools"].clone()).context("parsing tools/list response")?;
        Ok((client, tools))
    }

    async fn reader_task(client: Arc<Self>, stdout: tokio::process::ChildStdout) {
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let Ok(message) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            if let Some(id) = message.get("id").and_then(|v| v.as_i64())
                && message.get("method").is_none()
                && let Some(tx) = client.pending.lock().await.remove(&id)
            {
                let payload = message
                    .get("result")
                    .cloned()
                    .unwrap_or_else(|| json!({"error": message.get("error").cloned().unwrap_or(Value::Null)}));
                let _ = tx.send(payload);
            }
            // Server requests and notifications are not supported; ignore them.
        }
    }

    async fn write_line(&self, message: &Value) -> Result<()> {
        let mut body = serde_json::to_string(message)?;
        body.push('\n');
        let mut stdin = self.stdin.lock().await;
        stdin.write_all(body.as_bytes()).await?;
        stdin.flush().await?;
        Ok(())
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);
        self.write_line(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))
            .await?;
        let result = tokio::time::timeout(REQUEST_TIMEOUT, rx)
            .await
            .map_err(|_| anyhow!("MCP request {method} timed out"))?
            .map_err(|_| anyhow!("MCP server dropped the response"))?;
        if let Some(error) = result.get("error").filter(|e| !e.is_null()) {
            bail!("MCP error for {method}: {error}");
        }
        Ok(result)
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        self.write_line(&json!({"jsonrpc": "2.0", "method": method, "params": params}))
            .await
    }

    /// Invoke a tool and flatten its content blocks into plain text.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<String> {
        let result = self
            .request("tools/call", json!({"name": name, "arguments": arguments}))
            .await?;
        let text = result["content"]
            .as_array()
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|block| block["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();
        if result["isError"].as_bool().unwrap_or(false) {
            bail!("tool reported an error: {text}");
        }
        Ok(text)
    }
}
