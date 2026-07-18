use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

use crate::mcp::{McpClient, McpConfig, McpToolInfo};
use crate::tools::{Registry, Tool, clip};

const MAX_OUTPUT: usize = 30_000;

/// One MCP server tool exposed to the agent as `mcp__<server>__<tool>`.
pub struct McpTool {
    client: Arc<McpClient>,
    qualified_name: String,
    info: McpToolInfo,
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.qualified_name
    }

    fn description(&self) -> &str {
        &self.info.description
    }

    fn input_schema(&self) -> Value {
        self.info.input_schema.clone()
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let output = self.client.call_tool(&self.info.name, input).await?;
        Ok(clip(output, MAX_OUTPUT))
    }
}

/// Connect every configured MCP server and register its tools.
/// Servers that fail to start are reported and skipped, never fatal.
pub async fn register_mcp_tools(registry: &mut Registry, config: &McpConfig) -> Vec<String> {
    let mut notes = Vec::new();
    for (server_name, server_config) in &config.servers {
        match McpClient::connect(server_config).await {
            Ok((client, tools)) => {
                notes.push(format!("mcp server '{server_name}': {} tool(s)", tools.len()));
                for info in tools {
                    registry.register(Arc::new(McpTool {
                        client: Arc::clone(&client),
                        qualified_name: format!("mcp__{server_name}__{}", info.name),
                        info,
                    }));
                }
            }
            Err(error) => notes.push(format!("mcp server '{server_name}' failed: {error:#}")),
        }
    }
    notes
}
