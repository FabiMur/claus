use std::env;

use anyhow::{Context, Result};

/// Runtime configuration loaded from environment variables (via `.env` or the shell).
#[derive(Clone, Debug)]
pub struct Config {
    pub anthropic_api_key: String,
    pub voyage_api_key: Option<String>,
    pub model: String,
    pub max_tokens: u32,
    pub qdrant_url: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        let anthropic_api_key = env::var("ANTHROPIC_API_KEY").context("ANTHROPIC_API_KEY is not set")?;
        let voyage_api_key = env::var("VOYAGE_API_KEY").ok().filter(|k| !k.is_empty());
        let model = env::var("CLAUS_MODEL").unwrap_or_else(|_| "claude-opus-5".to_string());
        let max_tokens = env::var("CLAUS_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16_000);
        let qdrant_url = env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6334".to_string());

        Ok(Self {
            anthropic_api_key,
            voyage_api_key,
            model,
            max_tokens,
            qdrant_url,
        })
    }
}
