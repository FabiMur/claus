use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};
use walkdir::WalkDir;

use crate::tools::fs::required_str;
use crate::tools::{Tool, clip};

const MAX_OUTPUT: usize = 30_000;
const MAX_MATCHES: usize = 200;
const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", ".venv", "__pycache__", ".claus"];

pub struct SearchText;

#[async_trait]
impl Tool for SearchText {
    fn name(&self) -> &str {
        "search_text"
    }

    fn description(&self) -> &str {
        "Search files recursively for a literal substring; returns path:line matches. \
         For meaning-based lookup over the indexed codebase prefer rag_search."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Literal substring to look for"},
                "path": {"type": "string", "description": "Root directory (default: current directory)"}
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let query = required_str(&input, "query")?.to_string();
        let root = input["path"].as_str().unwrap_or(".").to_string();

        let matches = tokio::task::spawn_blocking(move || search(&root, &query)).await??;
        if matches.is_empty() {
            return Ok("no matches".to_string());
        }
        Ok(clip(matches.join("\n"), MAX_OUTPUT))
    }
}

fn search(root: &str, query: &str) -> Result<Vec<String>> {
    let mut matches = Vec::new();
    let walker = WalkDir::new(root).into_iter().filter_entry(|e| {
        e.file_name()
            .to_str()
            .is_none_or(|name| !SKIP_DIRS.contains(&name) && !name.starts_with('.') || name == ".")
    });
    for entry in walker {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(entry.path()) else {
            continue; // skip binary / non-UTF-8 files
        };
        for (i, line) in content.lines().enumerate() {
            if line.contains(query) {
                matches.push(format!("{}:{}: {}", entry.path().display(), i + 1, line.trim()));
                if matches.len() >= MAX_MATCHES {
                    return Ok(matches);
                }
            }
        }
    }
    Ok(matches)
}
