use async_trait::async_trait;

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::types::{ChatRequest, ChatResponse, ModelInfo, TtsRequest};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Remap `max_tokens` → `max_completion_tokens` for OpenAI's Chat Completions API.
///
/// OpenAI deprecated `max_tokens` in favour of `max_completion_tokens`, and
/// o-series / reasoning models reject `max_tokens` entirely.
fn to_openai_body(request: &ChatRequest) -> Result<serde_json::Value, BaochuanError> {
    let mut body = serde_json::to_value(request)?;
    if let Some(obj) = body.as_object_mut() {
        if obj.contains_key("max_completion_tokens") {
            obj.remove("max_tokens");
        } else if let Some(max) = obj.remove("max_tokens") {
            obj.insert("max_completion_tokens".to_string(), max);
        }
    }
    Ok(body)
}

/// A provider that connects to the [OpenAI](https://platform.openai.com/) API.
///
/// Compatible with any service that implements the OpenAI chat completions spec
/// (Azure OpenAI, local models via LM Studio, etc.) — use `with_base_url` to
/// point at an alternative endpoint.
///
/// On the wire, `max_tokens` is automatically sent as `max_completion_tokens`
/// so o-series and other reasoning models work without special casing.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::OpenAIProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = OpenAIProvider::new(std::env::var("OPENAI_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("gpt-4o")
///         .message(ChatMessage::user("What is the capital of France?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct OpenAIProvider {
    inner: OpenAICompatClient,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider.
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::OpenAIProvider::new(
    ///     std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set"),
    /// );
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAICompatClient::with_key(api_key, DEFAULT_BASE_URL),
        }
    }

    /// Override the base URL (e.g. for Azure OpenAI or local proxies).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner.base_url = base_url.into();
        self
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, BaochuanError> {
        self.inner.models().await
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BaochuanError> {
        self.inner.chat_json(&to_openai_body(request)?, self.name()).await
    }

    async fn stream_chat(&self, request: &ChatRequest) -> Result<ChunkStream, BaochuanError> {
        self.inner.stream_chat_json(&to_openai_body(request)?, self.name()).await
    }

    async fn tts(&self, request: &TtsRequest) -> Result<Vec<u8>, BaochuanError> {
        self.inner.tts(request, self.name()).await
    }
}
