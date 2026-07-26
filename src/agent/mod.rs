pub mod subagent;

use anyhow::{Result, bail};
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;

use crate::api::client::Client;
use crate::api::types::{ContentBlock, Message, Role, Usage};
use crate::tools::Registry;

const MAX_ITERATIONS: usize = 50;

/// Progress notifications emitted while a turn runs, consumed by the TUI.
#[derive(Clone, Debug)]
pub enum AgentEvent {
    AssistantText(String),
    ToolCall {
        name: String,
        input: Value,
    },
    ToolResult {
        name: String,
        output: String,
        is_error: bool,
    },
    /// Usage of one API round-trip; `input_tokens + output_tokens` of the
    /// latest one approximates the current context size.
    ApiUsage {
        usage: Usage,
    },
    TurnComplete,
    Error(String),
}

/// Drives the model/tool loop: send messages, execute requested tools,
/// feed results back, and repeat until the model ends its turn.
pub struct AgentLoop {
    client: Client,
    registry: Registry,
    system: String,
    messages: Vec<Message>,
    events: Option<UnboundedSender<AgentEvent>>,
}

impl AgentLoop {
    pub fn new(client: Client, registry: Registry, system: String) -> Self {
        Self {
            client,
            registry,
            system,
            messages: Vec::new(),
            events: None,
        }
    }

    pub fn with_events(mut self, events: UnboundedSender<AgentEvent>) -> Self {
        self.events = Some(events);
        self
    }

    fn emit(&self, event: AgentEvent) {
        if let Some(tx) = &self.events {
            let _ = tx.send(event);
        }
    }

    /// Run one user turn to completion and return the final assistant text.
    pub async fn run(&mut self, user_input: String) -> Result<String> {
        self.messages.push(Message::user_text(user_input));
        let tools = self.registry.definitions();
        let mut final_text = String::new();

        for _ in 0..MAX_ITERATIONS {
            let response = self
                .client
                .send(Some(self.system.clone()), self.messages.clone(), tools.clone())
                .await?;
            self.emit(AgentEvent::ApiUsage {
                usage: response.usage.clone(),
            });

            self.messages.push(Message {
                role: Role::Assistant,
                content: response.content.clone(),
            });

            for block in &response.content {
                if let ContentBlock::Text { text } = block
                    && !text.is_empty()
                {
                    final_text = text.clone();
                    self.emit(AgentEvent::AssistantText(text.clone()));
                }
            }

            match response.stop_reason.as_str() {
                "tool_use" => {
                    let results = self.execute_tools(&response.content).await;
                    self.messages.push(Message::tool_results(results));
                }
                "refusal" => bail!("the model declined this request (stop_reason: refusal)"),
                "max_tokens" => bail!("response was cut off by the max_tokens limit"),
                _ => {
                    self.emit(AgentEvent::TurnComplete);
                    return Ok(final_text);
                }
            }
        }
        bail!("agent stopped after {MAX_ITERATIONS} tool iterations without finishing")
    }

    /// Execute every tool_use block of an assistant message; all results go
    /// back in a single user message, preserving parallel tool-use behavior.
    async fn execute_tools(&self, content: &[ContentBlock]) -> Vec<ContentBlock> {
        let mut results = Vec::new();
        for block in content {
            let ContentBlock::ToolUse { id, name, input } = block else {
                continue;
            };
            self.emit(AgentEvent::ToolCall {
                name: name.clone(),
                input: input.clone(),
            });
            let (output, is_error) = match self.registry.execute(name, input.clone()).await {
                Ok(output) => (output, false),
                Err(error) => (error, true),
            };
            self.emit(AgentEvent::ToolResult {
                name: name.clone(),
                output: output.clone(),
                is_error,
            });
            results.push(ContentBlock::ToolResult {
                tool_use_id: id.clone(),
                content: output,
                is_error: is_error.then_some(true),
            });
        }
        results
    }
}

pub fn default_system_prompt(cwd: &str) -> String {
    format!(
        "You are claus, a terminal coding agent. You help with software engineering tasks \
         in the repository at {cwd}.\n\n\
         Use the available tools to explore, edit and verify code. Prefer rag_search to find \
         relevant code by meaning, search_text for exact strings, and the LSP tools for precise \
         navigation. Verify changes by running builds or tests with the shell tool when possible. \
         For broad subtasks that would flood your context (wide exploration, independent workstreams), \
         delegate to dispatch_agent when it is available.\n\n\
         Be concise. When you finish, summarize what you did."
    )
}
