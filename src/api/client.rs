use std::time::Duration;

use anyhow::{Result, anyhow, bail};
use reqwest::Client as HttpClient;
use reqwest::StatusCode;

use crate::api::types::{ApiErrorBody, ApiRequest, ApiResponse, Message, ToolDefinition};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";
const MAX_RETRIES: u32 = 3;

/// Thin client for the Anthropic Messages REST API.
#[derive(Clone)]
pub struct Client {
    http: HttpClient,
    api_key: String,
    model: String,
    max_tokens: u32,
}

impl Client {
    pub fn new(api_key: String, model: String, max_tokens: u32) -> Self {
        let http = HttpClient::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .expect("failed to build HTTP client");
        Self {
            http,
            api_key,
            model,
            max_tokens,
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// Send one turn; retries transient failures (429 / 5xx / network) with backoff.
    pub async fn send(
        &self,
        system: Option<String>,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
    ) -> Result<ApiResponse> {
        let request = ApiRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            system,
            messages,
            tools,
        };

        let mut attempt = 0;
        loop {
            match self.send_once(&request).await {
                Ok(response) => return Ok(response),
                Err(err) if attempt < MAX_RETRIES && is_retryable(&err) => {
                    attempt += 1;
                    tokio::time::sleep(Duration::from_secs(2u64.pow(attempt))).await;
                }
                Err(err) => return Err(err),
            }
        }
    }

    async fn send_once(&self, request: &ApiRequest) -> Result<ApiResponse> {
        let response = self
            .http
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(request)
            .send()
            .await?;

        let status = response.status();
        if status.is_success() {
            return Ok(response.json::<ApiResponse>().await?);
        }

        let body = response.text().await.unwrap_or_default();
        let detail = serde_json::from_str::<ApiErrorBody>(&body)
            .map(|e| format!("{}: {}", e.error.kind, e.error.message))
            .unwrap_or(body);
        if is_retryable_status(status) {
            Err(anyhow!(RetryableError(format!("API error {status}: {detail}"))))
        } else {
            bail!("API error {status}: {detail}")
        }
    }
}

#[derive(Debug)]
struct RetryableError(String);

impl std::fmt::Display for RetryableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for RetryableError {}

fn is_retryable_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn is_retryable(err: &anyhow::Error) -> bool {
    if err.is::<RetryableError>() {
        return true;
    }
    err.downcast_ref::<reqwest::Error>()
        .is_some_and(|e| e.is_timeout() || e.is_connect())
}
