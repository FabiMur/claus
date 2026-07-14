use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::rag::embedder::{Embedder, InputType};
use crate::rag::store::Store;
use crate::tools::fs::required_str;
use crate::tools::{Tool, clip};

const MAX_OUTPUT: usize = 30_000;
const DEFAULT_LIMIT: u64 = 8;

/// Semantic search over the indexed codebase (Voyage embeddings + Qdrant).
pub struct RagSearch {
    embedder: Embedder,
    store: Arc<Store>,
}

impl RagSearch {
    pub fn new(embedder: Embedder, store: Arc<Store>) -> Self {
        Self { embedder, store }
    }
}

#[async_trait]
impl Tool for RagSearch {
    fn name(&self) -> &str {
        "rag_search"
    }

    fn description(&self) -> &str {
        "Find code in this repository by meaning using vector search over the indexed codebase. \
         Use natural-language queries like 'where are HTTP retries handled'."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language description of the code to find"},
                "limit": {"type": "integer", "description": "Maximum number of results (default 8)"}
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, input: Value) -> Result<String> {
        let query = required_str(&input, "query")?;
        let limit = input["limit"].as_u64().unwrap_or(DEFAULT_LIMIT);

        let vectors = self.embedder.embed(&[query.to_string()], InputType::Query).await?;
        let Some(vector) = vectors.into_iter().next() else {
            return Ok("embedding service returned no vector".to_string());
        };

        let hits = self.store.search(vector, limit).await?;
        if hits.is_empty() {
            return Ok("no results; has the project been indexed with `claus index`?".to_string());
        }

        let formatted: Vec<String> = hits
            .iter()
            .map(|hit| {
                format!(
                    "--- {}:{}-{} (score {:.3})\n{}",
                    hit.path, hit.start_line, hit.end_line, hit.score, hit.text
                )
            })
            .collect();
        Ok(clip(formatted.join("\n\n"), MAX_OUTPUT))
    }
}
