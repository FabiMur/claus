use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::rag::bm25;
use crate::rag::embedder::{Embedder, InputType};
use crate::rag::indexer::load_chunk_corpus;
use crate::rag::store::Store;
use crate::tools::fs::required_str;
use crate::tools::{Tool, clip};

const MAX_OUTPUT: usize = 30_000;
const DEFAULT_LIMIT: usize = 8;
/// How many candidates each retriever contributes before fusion.
const CANDIDATES_PER_RETRIEVER: usize = 20;
/// Reciprocal rank fusion constant (standard value from the RRF paper).
const RRF_K: f32 = 60.0;

/// Hybrid semantic + lexical search over the indexed codebase: Voyage/Qdrant
/// vector search and a local BM25 index, fused with reciprocal rank fusion.
pub struct RagSearch {
    embedder: Embedder,
    store: Arc<Store>,
    root: PathBuf,
}

#[derive(Default)]
struct Candidate {
    path: String,
    start_line: usize,
    end_line: usize,
    text: String,
    fused_score: f32,
}

impl RagSearch {
    pub fn new(embedder: Embedder, store: Arc<Store>, root: PathBuf) -> Self {
        Self { embedder, store, root }
    }
}

#[async_trait]
impl Tool for RagSearch {
    fn name(&self) -> &str {
        "rag_search"
    }

    fn description(&self) -> &str {
        "Find code in this repository by meaning and keywords (hybrid vector + BM25 search over \
         the indexed codebase). Use natural-language queries like 'where are HTTP retries handled'."
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
        let limit = input["limit"].as_u64().unwrap_or(DEFAULT_LIMIT as u64) as usize;

        // Fused candidates keyed by chunk location; each retriever adds
        // 1 / (RRF_K + rank), so chunks found by both retrievers rise.
        let mut candidates: HashMap<String, Candidate> = HashMap::new();
        let mut fuse = |key: String, rank: usize, fill: &dyn Fn(&mut Candidate)| {
            let entry = candidates.entry(key).or_default();
            entry.fused_score += 1.0 / (RRF_K + rank as f32);
            if entry.text.is_empty() {
                fill(entry);
            }
        };

        // Semantic half: embed the query and search Qdrant.
        let vectors = self.embedder.embed(&[query.to_string()], InputType::Query).await?;
        if let Some(vector) = vectors.into_iter().next() {
            let hits = self.store.search(vector, CANDIDATES_PER_RETRIEVER as u64).await?;
            for (rank, hit) in hits.into_iter().enumerate() {
                let key = format!("{}:{}", hit.path, hit.start_line);
                fuse(key, rank, &|entry| {
                    entry.path = hit.path.clone();
                    entry.start_line = hit.start_line;
                    entry.end_line = hit.end_line;
                    entry.text = hit.text.clone();
                });
            }
        }

        // Lexical half: BM25 over the local chunk corpus written at index time.
        let corpus = load_chunk_corpus(&self.root);
        let documents: Vec<String> = corpus.iter().map(|record| record.text.clone()).collect();
        for (rank, (index, _)) in bm25::rank(&documents, query, CANDIDATES_PER_RETRIEVER)
            .into_iter()
            .enumerate()
        {
            let record = &corpus[index];
            let key = format!("{}:{}", record.path, record.start_line);
            fuse(key, rank, &|entry| {
                entry.path = record.path.clone();
                entry.start_line = record.start_line;
                entry.end_line = record.end_line;
                entry.text = record.text.clone();
            });
        }

        let mut fused: Vec<Candidate> = candidates.into_values().collect();
        fused.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        fused.truncate(limit);

        if fused.is_empty() {
            return Ok("no results; has the project been indexed with `claus index`?".to_string());
        }
        let formatted: Vec<String> = fused
            .iter()
            .map(|hit| {
                format!(
                    "--- {}:{}-{} (rrf {:.4})\n{}",
                    hit.path, hit.start_line, hit.end_line, hit.fused_score, hit.text
                )
            })
            .collect();
        Ok(clip(formatted.join("\n\n"), MAX_OUTPUT))
    }
}
