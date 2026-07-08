use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::process::Command;

use crate::tools::fs::required_str;
use crate::tools::{Tool, clip};

const MAX_OUTPUT: usize = 30_000;
const DEFAULT_TIMEOUT_SECS: u64 = 120;

pub struct Shell;

#[async_trait]
impl Tool for Shell {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Run a shell command with `sh -c` and return its exit code, stdout and stderr."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command line to execute"},
                "timeout_secs": {"type": "integer", "description": "Timeout in seconds (default 120)"}
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let command = required_str(&input, "command")?;
        let timeout = Duration::from_secs(input["timeout_secs"].as_u64().unwrap_or(DEFAULT_TIMEOUT_SECS));

        let output = tokio::time::timeout(timeout, Command::new("sh").arg("-c").arg(command).output()).await;
        let output = match output {
            Ok(result) => result?,
            Err(_) => return Ok(format!("command timed out after {}s", timeout.as_secs())),
        };

        let mut report = format!("exit code: {}\n", output.status.code().unwrap_or(-1));
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stdout.is_empty() {
            report.push_str(&format!("stdout:\n{stdout}"));
        }
        if !stderr.is_empty() {
            report.push_str(&format!("stderr:\n{stderr}"));
        }
        Ok(clip(report, MAX_OUTPUT))
    }
}
