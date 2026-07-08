use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::fs;

use crate::tools::{Tool, clip};

const MAX_OUTPUT: usize = 50_000;

pub struct ReadFile;

#[async_trait]
impl Tool for ReadFile {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read a UTF-8 text file and return its content with 1-based line numbers."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "offset": {"type": "integer", "description": "1-based line to start from (optional)"},
                "limit": {"type": "integer", "description": "Maximum number of lines (optional)"}
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let path = required_str(&input, "path")?;
        let content = fs::read_to_string(path)
            .await
            .with_context(|| format!("reading {path}"))?;
        let offset = input["offset"].as_u64().unwrap_or(1).max(1) as usize;
        let limit = input["limit"].as_u64().unwrap_or(2000) as usize;

        let numbered: Vec<String> = content
            .lines()
            .enumerate()
            .skip(offset - 1)
            .take(limit)
            .map(|(i, line)| format!("{:>5}\t{line}", i + 1))
            .collect();
        Ok(clip(numbered.join("\n"), MAX_OUTPUT))
    }
}

pub struct WriteFile;

#[async_trait]
impl Tool for WriteFile {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Create or overwrite a file with the given content, creating parent directories as needed."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Full file content"}
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let path = required_str(&input, "path")?;
        let content = required_str(&input, "content")?;
        if let Some(parent) = std::path::Path::new(path).parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).await?;
        }
        fs::write(path, content)
            .await
            .with_context(|| format!("writing {path}"))?;
        Ok(format!("wrote {} bytes to {path}", content.len()))
    }
}

pub struct EditFile;

#[async_trait]
impl Tool for EditFile {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Replace an exact string in a file. The old string must appear exactly once unless replace_all is true."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "old_string": {"type": "string", "description": "Exact text to replace"},
                "new_string": {"type": "string", "description": "Replacement text"},
                "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)"}
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let path = required_str(&input, "path")?;
        let old = required_str(&input, "old_string")?;
        let new = required_str(&input, "new_string")?;
        let replace_all = input["replace_all"].as_bool().unwrap_or(false);

        let content = fs::read_to_string(path)
            .await
            .with_context(|| format!("reading {path}"))?;
        let occurrences = content.matches(old).count();
        match occurrences {
            0 => bail!("old_string not found in {path}"),
            1 => {}
            n if !replace_all => bail!("old_string appears {n} times in {path}; pass replace_all or add context"),
            _ => {}
        }
        let updated = if replace_all {
            content.replace(old, new)
        } else {
            content.replacen(old, new, 1)
        };
        fs::write(path, updated).await?;
        Ok(format!(
            "replaced {} occurrence(s) in {path}",
            if replace_all { occurrences } else { 1 }
        ))
    }
}

pub struct ListDir;

#[async_trait]
impl Tool for ListDir {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn description(&self) -> &str {
        "List directory entries; directories are marked with a trailing slash."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: current directory)"}
            }
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let path = input["path"].as_str().unwrap_or(".");
        let mut entries = fs::read_dir(path).await.with_context(|| format!("listing {path}"))?;
        let mut names = Vec::new();
        while let Some(entry) = entries.next_entry().await? {
            let mut name = entry.file_name().to_string_lossy().to_string();
            if entry.file_type().await?.is_dir() {
                name.push('/');
            }
            names.push(name);
        }
        names.sort();
        Ok(clip(names.join("\n"), MAX_OUTPUT))
    }
}

pub fn required_str<'a>(input: &'a Value, key: &str) -> Result<&'a str> {
    input[key]
        .as_str()
        .with_context(|| format!("missing required parameter: {key}"))
}
