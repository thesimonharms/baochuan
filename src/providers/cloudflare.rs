use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::sse::cf_sse_to_chunks;
use crate::types::{ChatChoice, ChatMessage, ChatRequest, ChatResponse, ModelInfo, Role};

const BASE_URL: &str = "https://api.cloudflare.com/client/v4";

// ── Native wire types ─────────────────────────────────────────────────────────

#[derive(Serialize)]
struct CfChatRequest {
    messages: Vec<CfMessage>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct CfMessage {
    role: String,
    content: String,
}

// Non-streaming response envelope
#[derive(Deserialize)]
struct CfResponse<T> {
    result: Option<T>,
    success: bool,
    errors: Vec<CfError>,
}

#[derive(Deserialize)]
struct CfError {
    message: String,
}

#[derive(Deserialize)]
struct CfChatResult {
    response: String,
}

// Model search response
#[derive(Deserialize)]
struct CfModelSearchResponse {
    result: Vec<CfModel>,
}

#[derive(Deserialize)]
struct CfModel {
    id: String,
    description: Option<String>,
    task: Option<CfModelTask>,
}

#[derive(Deserialize)]
struct CfModelTask {
    name: String,
}

// ── Conversion helpers ────────────────────────────────────────────────────────

fn to_cf_messages(messages: &[ChatMessage]) -> Vec<CfMessage> {
    messages
        .iter()
        .map(|m| CfMessage {
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

fn from_cf_response(result: CfChatResult, model: &str) -> ChatResponse {
    ChatResponse {
        id: String::new(),
        model: model.to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage::assistant(result.response),
            finish_reason: Some("stop".to_string()),
        }],
        usage: None,
        citations: None,
    }
}

// ── Provider ──────────────────────────────────────────────────────────────────

/// A provider that connects to [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/)
/// using the **native `/ai/run/` API**.
///
/// Cloudflare routes requests through your account: the `account_id` is
/// embedded in every URL. Models are identified by their full path string
/// (e.g. `@cf/meta/llama-3.1-8b-instruct`).
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::CloudflareProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = CloudflareProvider::new(
///         std::env::var("CLOUDFLARE_ACCOUNT_ID").unwrap(),
///         std::env::var("CLOUDFLARE_API_TOKEN").unwrap(),
///     );
///
///     // List Text Generation models
///     let models = provider.models().await.unwrap();
///
///     let request = ChatRequestBuilder::new("@cf/meta/llama-3.1-8b-instruct")
///         .message(ChatMessage::user("What is the capital of France?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct CloudflareProvider {
    client: Client,
    account_id: String,
    api_token: String,
    base_url: String,
}

impl CloudflareProvider {
    /// Create a new Cloudflare Workers AI provider.
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::CloudflareProvider::new(
    ///     std::env::var("CLOUDFLARE_ACCOUNT_ID").expect("CLOUDFLARE_ACCOUNT_ID not set"),
    ///     std::env::var("CLOUDFLARE_API_TOKEN").expect("CLOUDFLARE_API_TOKEN not set"),
    /// );
    /// ```
    pub fn new(account_id: impl Into<String>, api_token: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            account_id: account_id.into(),
            api_token: api_token.into(),
            base_url: BASE_URL.to_string(),
        }
    }

    /// Override the base URL (useful for testing or proxies).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    fn run_url(&self, model: &str) -> String {
        format!(
            "{}/accounts/{}/ai/run/{}",
            self.base_url, self.account_id, model
        )
    }

    fn models_url(&self) -> String {
        format!(
            "{}/accounts/{}/ai/models/search",
            self.base_url, self.account_id
        )
    }
}

#[async_trait]
impl Provider for CloudflareProvider {
    fn name(&self) -> &str {
        "cloudflare"
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, BaochuanError> {
        debug!("listing models from Cloudflare Workers AI");

        let response = self
            .client
            .get(self.models_url())
            .bearer_auth(&self.api_token)
            // Filter to text-generation models only
            .query(&[("task", "Text Generation")])
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "Cloudflare models error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let envelope: CfModelSearchResponse = response.json().await?;
        Ok(envelope
            .result
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id,
                owned_by: m.task.map(|t| t.name),
                context_length: None,
                display_name: m.description,
            })
            .collect())
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BaochuanError> {
        debug!(model = %request.model, "sending chat request to Cloudflare Workers AI");

        let body = CfChatRequest {
            messages: to_cf_messages(&request.messages),
            stream: false,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
        };

        let response = self
            .client
            .post(self.run_url(&request.model))
            .bearer_auth(&self.api_token)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "Cloudflare API error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let envelope: CfResponse<CfChatResult> = response.json().await?;
        if !envelope.success {
            let msg = envelope
                .errors
                .into_iter()
                .map(|e| e.message)
                .collect::<Vec<_>>()
                .join("; ");
            return Err(BaochuanError::Api {
                status: 200,
                message: msg,
            });
        }

        let result = envelope.result.ok_or_else(|| BaochuanError::Api {
            status: 200,
            message: "empty result from Cloudflare".to_string(),
        })?;
        debug!(model = %request.model, "received Cloudflare response");
        Ok(from_cf_response(result, &request.model))
    }

    async fn stream_chat(&self, request: &ChatRequest) -> Result<ChunkStream, BaochuanError> {
        debug!(model = %request.model, "starting streaming chat request to Cloudflare Workers AI");

        let body = CfChatRequest {
            messages: to_cf_messages(&request.messages),
            stream: true,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
        };

        let response = self
            .client
            .post(self.run_url(&request.model))
            .bearer_auth(&self.api_token)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "Cloudflare stream error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let model = request.model.clone();
        Ok(Box::pin(cf_sse_to_chunks(response.bytes_stream(), model)))
    }
}
