use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::process::Command;

use crate::tools::fs::required_str;
use crate::tools::{Tool, clip};

const MAX_OUTPUT: usize = 30_000;
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Decides whether a non-read-only shell command may run.
#[async_trait]
pub trait PermissionGate: Send + Sync {
    async fn approve(&self, command: &str) -> bool;
}

/// Stdin y/N prompt for non-interactive (`claus ask`) sessions.
pub struct ConsoleGate;

#[async_trait]
impl PermissionGate for ConsoleGate {
    async fn approve(&self, command: &str) -> bool {
        eprintln!("\n[claus] shell wants to run: {command}");
        eprint!("[claus] allow? [y/N] ");
        tokio::task::spawn_blocking(|| {
            let mut line = String::new();
            std::io::stdin().read_line(&mut line).ok();
            matches!(line.trim(), "y" | "Y" | "yes")
        })
        .await
        .unwrap_or(false)
    }
}

/// Commands whose first word is listed here never touch state.
const READ_ONLY_COMMANDS: &[&str] = &[
    "ls", "cat", "head", "tail", "grep", "rg", "find", "wc", "file", "stat", "pwd", "which", "du", "df", "sort",
    "uniq", "cut", "tr", "date", "uname", "basename", "dirname", "tree",
];
/// Subcommand allowlists for tools that mix read and write operations.
const READ_ONLY_GIT: &[&str] = &["status", "log", "diff", "show", "branch", "blame", "remote", "ls-files"];
const READ_ONLY_CARGO: &[&str] = &[
    "check",
    "build",
    "test",
    "fmt",
    "clippy",
    "tree",
    "metadata",
    "--version",
];
/// Anything that can chain, redirect or substitute makes a command opaque.
const SHELL_METACHARACTERS: &[char] = &[';', '&', '|', '>', '<', '`', '$', '(', ')', '\n'];

/// Conservative classifier: a command is read-only when it has no shell
/// metacharacters and its program (plus subcommand for git/cargo) is
/// allowlisted. Everything else requires approval.
pub fn is_read_only(command: &str) -> bool {
    if command.contains(SHELL_METACHARACTERS) {
        return false;
    }
    let mut words = command.split_whitespace();
    let Some(program) = words.next() else {
        return false;
    };
    match program {
        "git" => words.next().is_some_and(|sub| READ_ONLY_GIT.contains(&sub)),
        "cargo" => words.next().is_some_and(|sub| READ_ONLY_CARGO.contains(&sub)),
        _ => READ_ONLY_COMMANDS.contains(&program),
    }
}

/// Shell tool guarded by a permission gate: read-only commands run directly,
/// anything state-changing must be approved by the user first.
pub struct Shell {
    gate: Option<Arc<dyn PermissionGate>>,
}

impl Shell {
    pub fn new(gate: Option<Arc<dyn PermissionGate>>) -> Self {
        Self { gate }
    }
}

#[async_trait]
impl Tool for Shell {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Run a shell command with `sh -c` and return its exit code, stdout and stderr. \
         Read-only commands run directly; anything else asks the user for approval first."
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
        if !is_read_only(command) {
            match &self.gate {
                Some(gate) if gate.approve(command).await => {}
                Some(_) => bail!("the user denied this command"),
                None => bail!("command requires approval and no permission gate is available"),
            }
        }
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

#[cfg(test)]
mod tests {
    use super::is_read_only;

    #[test]
    fn plain_read_commands_are_read_only() {
        assert!(is_read_only("ls -la src"));
        assert!(is_read_only("grep -rn TODO src"));
        assert!(is_read_only("git status"));
        assert!(is_read_only("cargo test"));
    }

    #[test]
    fn state_changing_commands_need_approval() {
        assert!(!is_read_only("rm -rf /"));
        assert!(!is_read_only("git push origin main"));
        assert!(!is_read_only("cargo publish"));
        assert!(!is_read_only("touch file"));
        assert!(!is_read_only(""));
    }

    #[test]
    fn metacharacters_defeat_the_allowlist() {
        assert!(!is_read_only("ls; rm -rf /"));
        assert!(!is_read_only("cat a > b"));
        assert!(!is_read_only("grep x $(dangerous)"));
        assert!(!is_read_only("ls && touch pwned"));
        assert!(!is_read_only("cat `cmd`"));
    }
}
