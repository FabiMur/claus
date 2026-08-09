use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use futures_util::StreamExt;
use reqwest::Client as HttpClient;
use reqwest::StatusCode;
use serde_json::Value;

use crate::api::types::{
    ApiErrorBody, ApiRequest, ApiResponse, ContentBlock, Message, SystemBlock, ToolDefinition, Usage,
};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";
const MAX_RETRIES: u32 = 3;

/// Called with each text fragment as the model produces it.
pub type OnDelta<'a> = &'a mut (dyn FnMut(&str) + Send);

/// Thin client for the Anthropic Messages REST API. Requests are streamed
/// (SSE) so long responses render token by token and avoid HTTP timeouts.
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

    /// Send one turn; retries transient connection failures (429 / 5xx /
    /// network) with backoff. Once the stream starts, errors are not retried
    /// so no text fragment is ever delivered twice.
    pub async fn send(
        &self,
        system: Option<String>,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
        on_delta: OnDelta<'_>,
    ) -> Result<ApiResponse> {
        // One cache breakpoint on the system block: the cached prefix covers
        // the tool definitions and system prompt, which are resent verbatim on
        // every iteration of the agent loop.
        let system = system.map(|text| {
            vec![SystemBlock {
                kind: "text",
                text,
                cache_control: Some(serde_json::json!({"type": "ephemeral"})),
            }]
        });
        let request = ApiRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            system,
            messages,
            tools,
            stream: true,
        };

        let mut attempt = 0;
        loop {
            match self.open_stream(&request).await {
                Ok(response) => return read_stream(response, on_delta).await,
                Err(err) if attempt < MAX_RETRIES && is_retryable(&err) => {
                    attempt += 1;
                    tokio::time::sleep(Duration::from_secs(2u64.pow(attempt))).await;
                }
                Err(err) => return Err(err),
            }
        }
    }

    /// Perform the HTTP request and fail before any event is consumed, so
    /// the retry loop only ever re-sends unstarted requests.
    async fn open_stream(&self, request: &ApiRequest) -> Result<reqwest::Response> {
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
            return Ok(response);
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

/// Consume the SSE body, forwarding text deltas and assembling the final
/// message exactly as a non-streaming response would look.
async fn read_stream(response: reqwest::Response, on_delta: OnDelta<'_>) -> Result<ApiResponse> {
    let mut stream = response.bytes_stream();
    let mut buffer: Vec<u8> = Vec::new();
    let mut assembler = MessageAssembler::default();

    while let Some(chunk) = stream.next().await {
        buffer.extend_from_slice(&chunk.context("reading SSE stream")?);
        // Events are separated by a blank line; only split on complete events
        // so multi-byte UTF-8 sequences are never cut in half.
        while let Some(position) = find_double_newline(&buffer) {
            let event: Vec<u8> = buffer.drain(..position + 2).collect();
            for line in String::from_utf8_lossy(&event).lines() {
                if let Some(data) = line.strip_prefix("data:") {
                    assembler.handle(data.trim_start(), on_delta)?;
                }
            }
        }
    }
    assembler.finish()
}

fn find_double_newline(buffer: &[u8]) -> Option<usize> {
    buffer.windows(2).position(|pair| pair == b"\n\n")
}

/// Rebuilds the response message from the SSE event sequence:
/// `message_start` → (`content_block_start` → `content_block_delta`* →
/// `content_block_stop`)* → `message_delta` → `message_stop`.
#[derive(Default)]
struct MessageAssembler {
    blocks: Vec<ContentBlock>,
    /// Accumulates `input_json_delta` fragments per tool_use block index.
    json_buffers: Vec<(usize, String)>,
    stop_reason: Option<String>,
    usage: Usage,
}

impl MessageAssembler {
    fn handle(&mut self, data: &str, on_delta: OnDelta<'_>) -> Result<()> {
        let event: Value = serde_json::from_str(data).with_context(|| format!("invalid SSE payload: {data}"))?;
        match event["type"].as_str().unwrap_or_default() {
            "message_start" => {
                let usage = &event["message"]["usage"];
                self.usage.input_tokens = usage["input_tokens"].as_u64().unwrap_or(0) as u32;
                self.usage.cache_creation_input_tokens =
                    usage["cache_creation_input_tokens"].as_u64().unwrap_or(0) as u32;
                self.usage.cache_read_input_tokens = usage["cache_read_input_tokens"].as_u64().unwrap_or(0) as u32;
            }
            "content_block_start" => self.start_block(&event)?,
            "content_block_delta" => self.apply_delta(&event, on_delta)?,
            "content_block_stop" => self.stop_block(&event)?,
            "message_delta" => {
                if let Some(reason) = event["delta"]["stop_reason"].as_str() {
                    self.stop_reason = Some(reason.to_string());
                }
                if let Some(tokens) = event["usage"]["output_tokens"].as_u64() {
                    self.usage.output_tokens = tokens as u32;
                }
            }
            "error" => bail!(
                "API stream error {}: {}",
                event["error"]["type"].as_str().unwrap_or("unknown"),
                event["error"]["message"].as_str().unwrap_or("")
            ),
            // "message_stop", "ping" and unknown future events need no action.
            _ => {}
        }
        Ok(())
    }

    fn start_block(&mut self, event: &Value) -> Result<()> {
        let index = event["index"].as_u64().unwrap_or(0) as usize;
        let block = &event["content_block"];
        let started = match block["type"].as_str().unwrap_or_default() {
            "text" => ContentBlock::Text {
                text: block["text"].as_str().unwrap_or_default().to_string(),
            },
            "thinking" => ContentBlock::Thinking {
                thinking: block["thinking"].as_str().unwrap_or_default().to_string(),
                signature: None,
            },
            "redacted_thinking" => ContentBlock::RedactedThinking {
                data: block["data"].as_str().unwrap_or_default().to_string(),
            },
            "tool_use" => {
                self.json_buffers.push((index, String::new()));
                ContentBlock::ToolUse {
                    id: block["id"].as_str().unwrap_or_default().to_string(),
                    name: block["name"].as_str().unwrap_or_default().to_string(),
                    input: Value::Null,
                }
            }
            other => bail!("unknown content block type in stream: {other}"),
        };
        self.blocks.push(started);
        Ok(())
    }

    fn apply_delta(&mut self, event: &Value, on_delta: OnDelta<'_>) -> Result<()> {
        let index = event["index"].as_u64().unwrap_or(0) as usize;
        let delta = &event["delta"];
        match delta["type"].as_str().unwrap_or_default() {
            "text_delta" => {
                let fragment = delta["text"].as_str().unwrap_or_default();
                if let Some(ContentBlock::Text { text }) = self.blocks.get_mut(index) {
                    text.push_str(fragment);
                }
                on_delta(fragment);
            }
            "thinking_delta" => {
                if let Some(ContentBlock::Thinking { thinking, .. }) = self.blocks.get_mut(index) {
                    thinking.push_str(delta["thinking"].as_str().unwrap_or_default());
                }
            }
            "signature_delta" => {
                if let Some(ContentBlock::Thinking { signature, .. }) = self.blocks.get_mut(index) {
                    let fragment = delta["signature"].as_str().unwrap_or_default();
                    match signature {
                        Some(existing) => existing.push_str(fragment),
                        None => *signature = Some(fragment.to_string()),
                    }
                }
            }
            "input_json_delta" => {
                let fragment = delta["partial_json"].as_str().unwrap_or_default();
                if let Some((_, buffer)) = self.json_buffers.iter_mut().find(|(i, _)| *i == index) {
                    buffer.push_str(fragment);
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn stop_block(&mut self, event: &Value) -> Result<()> {
        let index = event["index"].as_u64().unwrap_or(0) as usize;
        if let Some(position) = self.json_buffers.iter().position(|(i, _)| *i == index) {
            let (_, buffer) = self.json_buffers.swap_remove(position);
            if let Some(ContentBlock::ToolUse { input, .. }) = self.blocks.get_mut(index) {
                *input = if buffer.trim().is_empty() {
                    Value::Object(Default::default())
                } else {
                    serde_json::from_str(&buffer).with_context(|| format!("invalid tool input JSON: {buffer}"))?
                };
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<ApiResponse> {
        Ok(ApiResponse {
            content: self.blocks,
            stop_reason: self.stop_reason.context("stream ended without a stop_reason")?,
            usage: self.usage,
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(assembler: &mut MessageAssembler, events: &[&str], deltas: &mut String) {
        for event in events {
            let mut sink = |fragment: &str| deltas.push_str(fragment);
            assembler.handle(event, &mut sink).unwrap();
        }
    }

    #[test]
    fn assembles_text_and_tool_use_from_sse_events() {
        let mut assembler = MessageAssembler::default();
        let mut deltas = String::new();
        feed(
            &mut assembler,
            &[
                r#"{"type":"message_start","message":{"usage":{"input_tokens":42}}}"#,
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hola "}}"#,
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"mundo"}}"#,
                r#"{"type":"content_block_stop","index":0}"#,
                r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tu_1","name":"read_file","input":{}}}"#,
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}"#,
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"a.rs\"}"}}"#,
                r#"{"type":"content_block_stop","index":1}"#,
                r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}"#,
                r#"{"type":"message_stop"}"#,
            ],
            &mut deltas,
        );

        let response = assembler.finish().unwrap();
        assert_eq!(deltas, "Hola mundo");
        assert_eq!(response.stop_reason, "tool_use");
        assert_eq!(response.usage.input_tokens, 42);
        assert_eq!(response.usage.output_tokens, 7);
        assert_eq!(response.content.len(), 2);
        match &response.content[1] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "tu_1");
                assert_eq!(name, "read_file");
                assert_eq!(input["path"], "a.rs");
            }
            other => panic!("expected tool_use, got {other:?}"),
        }
    }

    #[test]
    fn stream_error_event_fails_the_turn() {
        let mut assembler = MessageAssembler::default();
        let mut sink = |_: &str| {};
        let result = assembler.handle(
            r#"{"type":"error","error":{"type":"overloaded_error","message":"try later"}}"#,
            &mut sink,
        );
        assert!(result.is_err());
    }

    #[test]
    fn finish_without_stop_reason_is_an_error() {
        assert!(MessageAssembler::default().finish().is_err());
    }
}
