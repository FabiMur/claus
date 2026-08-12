pub mod subagent;

use anyhow::{Result, bail};
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;

use crate::api::client::Client;
use crate::api::types::{ContentBlock, Message, Role, Usage};
use crate::tools::Registry;

const MAX_ITERATIONS: usize = 50;
/// Rough context budget in characters (~4 chars/token). Above this, old tool
/// results are cleared — they are the bulk of agent context and the cheapest
/// to drop, since the model has already acted on them.
const CONTEXT_CLEAR_CHARS: usize = 600_000;
/// Never clear tool results in the most recent messages.
const KEEP_RECENT_MESSAGES: usize = 8;
const CLEARED_MARKER: &str = "[old tool result cleared to save context]";

/// Progress notifications emitted while a turn runs, consumed by the TUI.
#[derive(Clone, Debug)]
pub enum AgentEvent {
    /// A streamed fragment of the assistant's reply, in arrival order.
    TextDelta(String),
    /// The complete text of one finished reply block (finalizes the deltas).
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
    /// The user cancelled the running turn.
    Interrupted,
    /// Background information for the UI (reindexing, context trimming...).
    Info(String),
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

    /// Drop the conversation history, starting the next turn from scratch.
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Restore API-valid history after a turn was cancelled mid-flight: any
    /// trailing assistant tool_use without a result gets an "interrupted"
    /// tool_result, since the API rejects unanswered tool calls.
    pub fn repair_interrupted(&mut self) {
        let Some(last) = self.messages.last() else {
            return;
        };
        if last.role != Role::Assistant {
            return;
        }
        let results: Vec<ContentBlock> = last
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolUse { id, .. } => Some(ContentBlock::ToolResult {
                    tool_use_id: id.clone(),
                    content: "interrupted by the user before execution".to_string(),
                    is_error: Some(true),
                }),
                _ => None,
            })
            .collect();
        if !results.is_empty() {
            self.messages.push(Message::tool_results(results));
        }
    }

    fn emit(&self, event: AgentEvent) {
        if let Some(tx) = &self.events {
            let _ = tx.send(event);
        }
    }

    /// Run one user turn to completion and return the final assistant text.
    pub async fn run(&mut self, user_input: String) -> Result<String> {
        // After an interrupted turn the history may already end in a user
        // message; merge instead of stacking two consecutive user turns.
        match self.messages.last_mut() {
            Some(message) if message.role == Role::User => {
                message.content.push(ContentBlock::Text { text: user_input });
            }
            _ => self.messages.push(Message::user_text(user_input)),
        }
        let tools = self.registry.definitions();
        let mut final_text = String::new();

        for _ in 0..MAX_ITERATIONS {
            self.trim_context();
            let events = self.events.clone();
            let mut on_delta = |fragment: &str| {
                if let (Some(tx), false) = (&events, fragment.is_empty()) {
                    let _ = tx.send(AgentEvent::TextDelta(fragment.to_string()));
                }
            };
            let response = self
                .client
                .send(
                    Some(self.system.clone()),
                    self.messages.clone(),
                    tools.clone(),
                    &mut on_delta,
                )
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

    /// Keep the conversation within the context budget by clearing old tool
    /// results (oldest first, recent messages untouched). Trades a prompt
    /// cache miss for a bounded context, so it only fires past the budget.
    fn trim_context(&mut self) {
        if approx_chars(&self.messages) <= CONTEXT_CLEAR_CHARS {
            return;
        }
        let cutoff = self.messages.len().saturating_sub(KEEP_RECENT_MESSAGES);
        let mut cleared = 0;
        for index in 0..cutoff {
            for block in &mut self.messages[index].content {
                if let ContentBlock::ToolResult { content, .. } = block
                    && content.len() > CLEARED_MARKER.len()
                {
                    *content = CLEARED_MARKER.to_string();
                    cleared += 1;
                }
            }
            if approx_chars(&self.messages) <= CONTEXT_CLEAR_CHARS {
                break;
            }
        }
        if cleared > 0 {
            self.emit(AgentEvent::Info(format!(
                "context trimmed: cleared {cleared} old tool result(s)"
            )));
        }
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

/// Cheap size estimate of the conversation (block text lengths, no serialization).
fn approx_chars(messages: &[Message]) -> usize {
    messages
        .iter()
        .flat_map(|message| &message.content)
        .map(|block| match block {
            ContentBlock::Text { text } => text.len(),
            ContentBlock::Thinking { thinking, .. } => thinking.len(),
            ContentBlock::RedactedThinking { data } => data.len(),
            ContentBlock::ToolUse { input, .. } => input.to_string().len(),
            ContentBlock::ToolResult { content, .. } => content.len(),
        })
        .sum()
}

pub fn default_system_prompt(cwd: &str) -> String {
    format!(
        "You are claus, a terminal coding agent. You help with software engineering tasks \
         in the repository at {cwd}.\n\n\
         Use the available tools to explore, edit and verify code. Prefer rag_search to find \
         relevant code by meaning, search_text for exact strings, and the LSP tools for precise \
         navigation and diagnostics. Verify changes by running builds or tests with the shell tool \
         when possible. For broad subtasks that would flood your context (wide exploration, \
         independent workstreams), delegate to dispatch_agent when it is available.\n\n\
         Be concise. When you finish, summarize what you did."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_agent() -> AgentLoop {
        let client = Client::new("test-key".into(), "test-model".into(), 100);
        AgentLoop::new(client, Registry::default(), "system".into())
    }

    #[test]
    fn repair_adds_results_for_dangling_tool_use() {
        let mut agent = test_agent();
        agent.messages.push(Message::user_text("do something"));
        agent.messages.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "tu_1".into(),
                name: "shell".into(),
                input: json!({}),
            }],
        });
        agent.repair_interrupted();

        let last = agent.messages.last().unwrap();
        assert_eq!(last.role, Role::User);
        assert!(matches!(&last.content[0],
            ContentBlock::ToolResult { tool_use_id, is_error: Some(true), .. } if tool_use_id == "tu_1"));
    }

    #[test]
    fn repair_is_a_no_op_without_dangling_tool_use() {
        let mut agent = test_agent();
        agent.messages.push(Message::user_text("hi"));
        agent.repair_interrupted();
        assert_eq!(agent.messages.len(), 1);
    }

    #[test]
    fn trim_clears_old_tool_results_but_keeps_recent() {
        let mut agent = test_agent();
        let big = "x".repeat(CONTEXT_CLEAR_CHARS / 2);
        for i in 0..12 {
            agent
                .messages
                .push(Message::tool_results(vec![ContentBlock::ToolResult {
                    tool_use_id: format!("tu_{i}"),
                    content: big.clone(),
                    is_error: None,
                }]));
        }
        agent.trim_context();

        let cleared = |m: &Message| matches!(&m.content[0], ContentBlock::ToolResult { content, .. } if content == CLEARED_MARKER);
        assert!(cleared(&agent.messages[0]), "oldest result should be cleared");
        assert!(
            !cleared(agent.messages.last().unwrap()),
            "recent results must stay intact"
        );
    }
}
