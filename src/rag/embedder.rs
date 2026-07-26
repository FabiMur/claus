use anyhow::{Context, Result, bail};
use reqwest::Client as HttpClient;
use serde::Deserialize;
use serde_json::json;

const VOYAGE_URL: &str = "https://api.voyageai.com/v1/embeddings";
const MODEL: &str = "voyage-code-3";
/// Output dimension of voyage-code-3 with default settings.
pub const EMBEDDING_DIM: u64 = 1024;
const BATCH_SIZE: usize = 128;
/// Keep each request comfortably under free-tier token-per-minute limits.
const MAX_BATCH_CHARS: usize = 30_000;
const MAX_RATE_LIMIT_RETRIES: u32 = 8;
const RATE_LIMIT_BACKOFF_SECS: u64 = 21;

/// What the texts are used for; Voyage tunes the embedding accordingly.
#[derive(Clone, Copy)]
pub enum InputType {
    Document,
    Query,
}

impl InputType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Document => "document",
            Self::Query => "query",
        }
    }
}

/// Client for the Voyage AI embeddings REST API.
#[derive(Clone)]
pub struct Embedder {
    http: HttpClient,
    api_key: String,
}

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
    index: usize,
}

impl Embedder {
    pub fn new(api_key: String) -> Self {
        Self {
            http: HttpClient::new(),
            api_key,
        }
    }

    /// Embed all texts, batching requests; result order matches input order.
    /// Batches are bounded by item count and total size to respect rate limits.
    pub async fn embed(&self, texts: &[String], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        let mut batch: Vec<String> = Vec::new();
        let mut batch_chars = 0;
        for text in texts {
            if !batch.is_empty() && (batch.len() >= BATCH_SIZE || batch_chars + text.len() > MAX_BATCH_CHARS) {
                embeddings.extend(self.embed_batch(&batch, input_type).await?);
                batch.clear();
                batch_chars = 0;
            }
            batch_chars += text.len();
            batch.push(text.clone());
        }
        if !batch.is_empty() {
            embeddings.extend(self.embed_batch(&batch, input_type).await?);
        }
        Ok(embeddings)
    }

    /// One request; waits out 429 responses (free-tier limits are per minute).
    async fn embed_batch(&self, batch: &[String], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        for _ in 0..MAX_RATE_LIMIT_RETRIES {
            match self.embed_once(batch, input_type).await {
                Err(error) if error.to_string().contains("429") => {
                    tokio::time::sleep(std::time::Duration::from_secs(RATE_LIMIT_BACKOFF_SECS)).await;
                }
                other => return other,
            }
        }
        bail!("voyage API still rate-limited after {MAX_RATE_LIMIT_RETRIES} retries")
    }

    async fn embed_once(&self, batch: &[String], input_type: InputType) -> Result<Vec<Vec<f32>>> {
        let response = self
            .http
            .post(VOYAGE_URL)
            .bearer_auth(&self.api_key)
            .json(&json!({
                "input": batch,
                "model": MODEL,
                "input_type": input_type.as_str(),
            }))
            .send()
            .await
            .context("calling voyage embeddings API")?;

        let status = response.status();
        if !status.is_success() {
            bail!(
                "voyage API error {status}: {}",
                response.text().await.unwrap_or_default()
            );
        }

        let mut data = response.json::<EmbeddingsResponse>().await?.data;
        data.sort_by_key(|item| item.index);
        Ok(data.into_iter().map(|item| item.embedding).collect())
    }
}
