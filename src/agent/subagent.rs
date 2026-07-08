use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::agent::{AgentLoop, default_system_prompt};
use crate::api::client::Client;
use crate::tools::fs::required_str;
use crate::tools::{Registry, Tool, clip};

const MAX_OUTPUT: usize = 30_000;

/// Tool that spawns a nested agent loop with its own conversation history.
/// The sub-agent gets every tool except this one, so delegation depth is 1.
pub struct DispatchAgent {
    client: Client,
    registry: Registry,
    cwd: String,
}

impl DispatchAgent {
    pub fn new(client: Client, registry: Registry, cwd: String) -> Self {
        Self { client, registry, cwd }
    }
}

#[async_trait]
impl Tool for DispatchAgent {
    fn name(&self) -> &str {
        "dispatch_agent"
    }

    fn description(&self) -> &str {
        "Delegate a self-contained task to a sub-agent with its own context and the same tools. \
         Give it a complete task description; it returns only its final report."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Complete, self-contained task description"}
            },
            "required": ["task"]
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let task = required_str(&input, "task")?;
        let registry = self.registry.without(self.name());
        let system = format!(
            "{}\n\nYou are a sub-agent working on one delegated task. \
             Finish it and reply with a final report for the main agent.",
            default_system_prompt(&self.cwd)
        );
        let mut agent = AgentLoop::new(self.client.clone(), registry, system);
        let report = agent.run(task.to_string()).await?;
        Ok(clip(report, MAX_OUTPUT))
    }
}
