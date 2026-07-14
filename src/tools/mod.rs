pub mod fs;
pub mod rag;
pub mod search;
pub mod shell;

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

use crate::api::types::ToolDefinition;

/// A capability the model can invoke through the API's tool-use protocol.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn input_schema(&self) -> Value;
    async fn execute(&self, input: Value) -> Result<String>;
}

/// Set of tools exposed to one agent, addressable by name.
#[derive(Clone, Default)]
pub struct Registry {
    tools: Vec<Arc<dyn Tool>>,
}

impl Registry {
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.push(tool);
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
            })
            .collect()
    }

    /// Runs the named tool; the `Err` string is sent back as an `is_error` tool result.
    pub async fn execute(&self, name: &str, input: Value) -> Result<String, String> {
        let tool = self
            .tools
            .iter()
            .find(|t| t.name() == name)
            .ok_or_else(|| format!("unknown tool: {name}"))?;
        tool.execute(input).await.map_err(|e| format!("{e:#}"))
    }

    /// Copy of this registry without the named tool (used to bound sub-agent recursion).
    pub fn without(&self, name: &str) -> Self {
        Self {
            tools: self.tools.iter().filter(|t| t.name() != name).cloned().collect(),
        }
    }
}

/// Truncate long tool output so a single result cannot flood the context window.
pub fn clip(output: String, max_chars: usize) -> String {
    if output.len() <= max_chars {
        return output;
    }
    let cut = output
        .char_indices()
        .take_while(|(i, _)| *i < max_chars)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    format!("{}\n... [truncated {} bytes]", &output[..cut], output.len() - cut)
}
