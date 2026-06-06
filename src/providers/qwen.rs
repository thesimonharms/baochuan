use async_trait::async_trait;
use reqwest::{Client, header};
use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::sse::dashscope_sse_to_chunks;
use crate::types::{ChatChoice, ChatMessage, ChatRequest, ChatResponse, Role, Usage};

const BASE_URL: &str = "https://dashscope.aliyuncs.com/api/v1";
const CHAT_PATH: &str = "services/aigc/text-generation/generation";

// ── Native DashScope wire types ───────────────────────────────────────────────

#[derive(Serialize)]
struct DashScopeRequest<'a> {
    model: &'a str,
    input: DashScopeInput<'a>,
    parameters: DashScopeParameters,
}

#[derive(Serialize)]
struct DashScopeInput<'a> {
    messages: Vec<DashScopeMessage>,
    // used only as a lifetime anchor
    #[serde(skip)]
    _phantom: std::marker::PhantomData<&'a ()>,
}

#[derive(Serialize)]
struct DashScopeMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct DashScopeParameters {
    result_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    incremental_output: bool,
}

#[derive(Deserialize)]
struct DashScopeResponse {
    output: DashScopeOutput,
    usage: Option<DashScopeUsage>,
    request_id: Option<String>,
}

#[derive(Deserialize)]
struct DashScopeOutput {
    choices: Vec<DashScopeChoice>,
}

#[derive(Deserialize)]
struct DashScopeChoice {
    message: DashScopeResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct DashScopeResponseMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct DashScopeUsage {
    input_tokens: u32,
    output_tokens: u32,
    total_tokens: u32,
}

// ── Conversion helpers ────────────────────────────────────────────────────────

fn to_dashscope_messages(messages: &[ChatMessage]) -> Vec<DashScopeMessage> {
    messages
        .iter()
        .map(|m| DashScopeMessage {
            role: match m.role {
                Role::System => "system".to_string(),
                Role::User => "user".to_string(),
                Role::Assistant => "assistant".to_string(),
                Role::Tool => "tool".to_string(),
            },
            content: m.content.to_text_lossy(),
        })
        .collect()
}

fn from_dashscope_response(resp: DashScopeResponse, model: &str) -> ChatResponse {
    let choices = resp
        .output
        .choices
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            let role = if c.message.role == "assistant" {
                Role::Assistant
            } else {
                Role::User
            };
            ChatChoice {
                index: i as u32,
                message: ChatMessage {
                    role,
                    content: c.message.content.into(),
                    audio: None,
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: c.finish_reason,
            }
        })
        .collect();

    let usage = resp.usage.map(|u| Usage {
        prompt_tokens: u.input_tokens,
        completion_tokens: u.output_tokens,
        total_tokens: u.total_tokens,
    });

    ChatResponse {
        id: resp.request_id.unwrap_or_default(),
        model: model.to_string(),
        choices,
        usage,
        citations: None,
    }
}

fn build_request<'a>(request: &'a ChatRequest, streaming: bool) -> DashScopeRequest<'a> {
    DashScopeRequest {
        model: &request.model,
        input: DashScopeInput {
            messages: to_dashscope_messages(&request.messages),
            _phantom: std::marker::PhantomData,
        },
        parameters: DashScopeParameters {
            result_format: "message",
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            incremental_output: streaming,
        },
    }
}

// ── Provider ──────────────────────────────────────────────────────────────────

/// A provider that connects to [Alibaba Cloud DashScope](https://dashscope.aliyun.com/)
/// to access **Qwen** (通义千问) models using the **native DashScope API**.
///
/// DashScope uses a different request envelope from OpenAI: messages live under
/// `input.messages`, generation settings under `parameters`, and the response
/// is wrapped in `output.choices`. Streaming uses Server-Sent Events with the
/// `X-DashScope-SSE: enable` header and incremental content delivery.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::QwenProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = QwenProvider::new(std::env::var("DASHSCOPE_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("qwen-turbo")
///         .message(ChatMessage::user("你好！"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct QwenProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl QwenProvider {
    /// Create a new Qwen/DashScope provider.
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::QwenProvider::new(
    ///     std::env::var("DASHSCOPE_API_KEY").expect("DASHSCOPE_API_KEY not set"),
    /// );
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: BASE_URL.to_string(),
        }
    }

    /// Override the base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    fn chat_url(&self) -> String {
        format!("{}/{}", self.base_url, CHAT_PATH)
    }
}

#[async_trait]
impl Provider for QwenProvider {
    fn name(&self) -> &str {
        "qwen"
    }

    // DashScope does not expose a public model-listing REST endpoint in the
    // native API. Use the DashScope console or SDK to discover available models.
    // models() returns the default Ok(vec![]).

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BaochuanError> {
        debug!(model = %request.model, "sending chat request to DashScope");

        let body = build_request(request, false);
        let response = self
            .client
            .post(self.chat_url())
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            error!(status = %status, body = %text, "DashScope API error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: text,
            });
        }

        let ds_response: DashScopeResponse = response.json().await?;
        debug!(request_id = ?ds_response.request_id, "received DashScope response");
        Ok(from_dashscope_response(ds_response, &request.model))
    }

    async fn stream_chat(&self, request: &ChatRequest) -> Result<ChunkStream, BaochuanError> {
        debug!(model = %request.model, "starting streaming chat request to DashScope");

        let body = build_request(request, true);
        let response = self
            .client
            .post(self.chat_url())
            .bearer_auth(&self.api_key)
            .header("X-DashScope-SSE", "enable")
            .header(header::ACCEPT, "text/event-stream")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            error!(status = %status, body = %text, "DashScope stream error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: text,
            });
        }

        let model = request.model.clone();
        Ok(Box::pin(dashscope_sse_to_chunks(
            response.bytes_stream(),
            model,
        )))
    }
}
