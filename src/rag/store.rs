use anyhow::{Context, Result};
use qdrant_client::Payload;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter, PointStruct, SearchPointsBuilder,
    UpsertPointsBuilder, VectorParamsBuilder,
};
use serde_json::json;

use crate::rag::chunker::Chunk;
use crate::rag::embedder::EMBEDDING_DIM;

/// One retrieved chunk with its similarity score.
#[derive(Clone, Debug)]
pub struct SearchHit {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
    pub score: f32,
}

/// Vector store backed by a Qdrant collection (one collection per project).
pub struct Store {
    client: Qdrant,
    collection: String,
}

impl Store {
    pub async fn connect(url: &str, collection: String) -> Result<Self> {
        let client = Qdrant::from_url(url)
            .build()
            .context("connecting to qdrant (is the docker container running?)")?;
        let store = Self { client, collection };
        store.ensure_collection().await?;
        Ok(store)
    }

    async fn ensure_collection(&self) -> Result<()> {
        if self.client.collection_exists(&self.collection).await? {
            return Ok(());
        }
        self.client
            .create_collection(
                CreateCollectionBuilder::new(&self.collection)
                    .vectors_config(VectorParamsBuilder::new(EMBEDDING_DIM, Distance::Cosine)),
            )
            .await?;
        Ok(())
    }

    /// Stable point id derived from the chunk location; re-upserts overwrite.
    fn point_id(path: &str, chunk: &Chunk) -> u64 {
        let key = format!("{path}:{}", chunk.start_line);
        u64::from_le_bytes(blake3::hash(key.as_bytes()).as_bytes()[..8].try_into().unwrap())
    }

    pub async fn upsert_chunks(&self, path: &str, chunks: &[Chunk], vectors: Vec<Vec<f32>>) -> Result<()> {
        let points: Vec<PointStruct> = chunks
            .iter()
            .zip(vectors)
            .map(|(chunk, vector)| {
                let payload: Payload = json!({
                    "path": path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "text": chunk.text,
                })
                .try_into()
                .expect("payload is a JSON object");
                PointStruct::new(Self::point_id(path, chunk), vector, payload)
            })
            .collect();
        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection, points).wait(true))
            .await?;
        Ok(())
    }

    /// Drop every chunk previously indexed for a file (before re-indexing it).
    pub async fn delete_file(&self, path: &str) -> Result<()> {
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection)
                    .points(Filter::must([Condition::matches("path", path.to_string())]))
                    .wait(true),
            )
            .await?;
        Ok(())
    }

    pub async fn search(&self, vector: Vec<f32>, limit: u64) -> Result<Vec<SearchHit>> {
        let response = self
            .client
            .search_points(SearchPointsBuilder::new(&self.collection, vector, limit).with_payload(true))
            .await?;

        let hits = response
            .result
            .into_iter()
            .map(|point| {
                let payload = point.payload;
                SearchHit {
                    path: payload
                        .get("path")
                        .and_then(|v| v.as_str())
                        .cloned()
                        .unwrap_or_default(),
                    start_line: payload.get("start_line").and_then(|v| v.as_integer()).unwrap_or(0) as usize,
                    end_line: payload.get("end_line").and_then(|v| v.as_integer()).unwrap_or(0) as usize,
                    text: payload
                        .get("text")
                        .and_then(|v| v.as_str())
                        .cloned()
                        .unwrap_or_default(),
                    score: point.score,
                }
            })
            .collect();
        Ok(hits)
    }
}
