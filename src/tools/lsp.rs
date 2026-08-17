use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::lsp::{LspClient, format_locations, uri_for};
use crate::tools::fs::required_str;
use crate::tools::{Tool, clip};

const MAX_OUTPUT: usize = 20_000;

/// Lazily spawns and caches one language server per configured command,
/// choosing the server from the file extension.
pub struct LspManager {
    root: PathBuf,
    servers: HashMap<String, String>, // extension -> command
    clients: Mutex<HashMap<String, Arc<LspClient>>>,
}

impl LspManager {
    pub fn new(root: PathBuf) -> Arc<Self> {
        let mut servers = HashMap::new();
        servers.insert("rs".to_string(), "rust-analyzer".to_string());
        servers.insert("py".to_string(), "pyright-langserver --stdio".to_string());
        Arc::new(Self {
            root,
            servers,
            clients: Mutex::new(HashMap::new()),
        })
    }

    async fn client_for(&self, path: &Path) -> Result<Arc<LspClient>> {
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or_default();
        let Some(command) = self.servers.get(extension) else {
            bail!("no language server configured for .{extension} files");
        };
        let mut clients = self.clients.lock().await;
        if let Some(client) = clients.get(command) {
            return Ok(Arc::clone(client));
        }
        let client = LspClient::start(command, &self.root).await?;
        clients.insert(command.clone(), Arc::clone(&client));
        Ok(client)
    }

    /// Common preamble for position-based requests: resolve, open, wait for
    /// the server to finish indexing (else it answers with misleading empty
    /// results), then build params.
    async fn position_params(&self, input: &Value) -> Result<(Arc<LspClient>, Value)> {
        let path = PathBuf::from(required_str(input, "path")?);
        let line = input["line"].as_u64().context("missing required parameter: line")?;
        let column = input["column"].as_u64().unwrap_or(1);
        let client = self.client_for(&path).await?;
        client.ensure_open(&path).await?;
        client.wait_ready(std::time::Duration::from_secs(60)).await;
        let params = json!({
            "textDocument": {"uri": uri_for(&path)},
            "position": {"line": line.saturating_sub(1), "character": column.saturating_sub(1)}
        });
        Ok((client, params))
    }
}

fn position_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the source file"},
            "line": {"type": "integer", "description": "1-based line number"},
            "column": {"type": "integer", "description": "1-based column number (default 1)"}
        },
        "required": ["path", "line"]
    })
}

pub struct LspDefinition(pub Arc<LspManager>);

#[async_trait]
impl Tool for LspDefinition {
    fn name(&self) -> &str {
        "lsp_definition"
    }

    fn description(&self) -> &str {
        "Jump to the definition of the symbol at a file position using the language server."
    }

    fn input_schema(&self) -> Value {
        position_schema()
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let (client, params) = self.0.position_params(&input).await?;
        let result = client.request("textDocument/definition", params).await?;
        Ok(format_locations(&result))
    }
}

pub struct LspReferences(pub Arc<LspManager>);

#[async_trait]
impl Tool for LspReferences {
    fn name(&self) -> &str {
        "lsp_references"
    }

    fn description(&self) -> &str {
        "List all references to the symbol at a file position using the language server."
    }

    fn input_schema(&self) -> Value {
        position_schema()
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let (client, mut params) = self.0.position_params(&input).await?;
        params["context"] = json!({"includeDeclaration": true});
        let result = client.request("textDocument/references", params).await?;
        Ok(clip(format_locations(&result), MAX_OUTPUT))
    }
}

pub struct LspDiagnostics(pub Arc<LspManager>);

#[async_trait]
impl Tool for LspDiagnostics {
    fn name(&self) -> &str {
        "lsp_diagnostics"
    }

    fn description(&self) -> &str {
        "List compiler/analyzer diagnostics (errors, warnings) for a file via the language server."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the source file"}
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let path = PathBuf::from(required_str(&input, "path")?);
        let client = self.0.client_for(&path).await?;
        client.ensure_open(&path).await?;
        client.wait_ready(std::time::Duration::from_secs(60)).await;

        // Diagnostics are pushed, not pulled: poll briefly for the server to
        // publish after analysis settles.
        let uri = uri_for(&path);
        let mut diagnostics = None;
        for _ in 0..20 {
            diagnostics = client.diagnostics_for(&uri).await;
            if diagnostics.is_some() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }

        let Some(Value::Array(items)) = diagnostics else {
            return Ok("no diagnostics published for this file (it may be clean)".to_string());
        };
        if items.is_empty() {
            return Ok("no diagnostics: the file is clean".to_string());
        }
        let formatted: Vec<String> = items
            .iter()
            .map(|diagnostic| {
                let severity = match diagnostic["severity"].as_u64() {
                    Some(1) => "error",
                    Some(2) => "warning",
                    Some(3) => "info",
                    _ => "hint",
                };
                let line = diagnostic["range"]["start"]["line"].as_u64().unwrap_or(0) + 1;
                let message = diagnostic["message"].as_str().unwrap_or_default();
                format!("{line}: [{severity}] {message}")
            })
            .collect();
        Ok(clip(formatted.join("\n"), MAX_OUTPUT))
    }
}

pub struct LspHover(pub Arc<LspManager>);

#[async_trait]
impl Tool for LspHover {
    fn name(&self) -> &str {
        "lsp_hover"
    }

    fn description(&self) -> &str {
        "Show type information and documentation for the symbol at a file position."
    }

    fn input_schema(&self) -> Value {
        position_schema()
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let (client, params) = self.0.position_params(&input).await?;
        let result = client.request("textDocument/hover", params).await?;
        let contents = &result["contents"];
        let text = if let Some(value) = contents["value"].as_str() {
            value.to_string()
        } else if let Some(items) = contents.as_array() {
            items
                .iter()
                .filter_map(|item| item.as_str().or_else(|| item["value"].as_str()))
                .collect::<Vec<_>>()
                .join("\n")
        } else if let Some(value) = contents.as_str() {
            value.to_string()
        } else {
            "no hover information".to_string()
        };
        Ok(clip(text, MAX_OUTPUT))
    }
}
