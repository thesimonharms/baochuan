use async_trait::async_trait;

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

const DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";

/// A provider that connects to [xAI Grok](https://console.x.ai/).
///
/// The Grok API is OpenAI-compatible. Notable models include `grok-3` and
/// `grok-3-mini`.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::GrokProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = GrokProvider::new(std::env::var("XAI_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("grok-3")
///         .message(ChatMessage::user("What is the capital of France?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct GrokProvider {
    inner: OpenAICompatClient,
}

impl GrokProvider {
    /// Create a new Grok provider.
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::GrokProvider::new(
    ///     std::env::var("XAI_API_KEY").expect("XAI_API_KEY not set"),
    /// );
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self { inner: OpenAICompatClient::with_key(api_key, DEFAULT_BASE_URL) }
    }

    /// Override the base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner.base_url = base_url.into();
        self
    }
}

#[async_trait]
impl Provider for GrokProvider {
    fn name(&self) -> &str {
        "grok"
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, BaochuanError> {
        self.inner.models().await
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BaochuanError> {
        self.inner.chat(request, self.name()).await
    }

    async fn stream_chat(&self, request: &ChatRequest) -> Result<ChunkStream, BaochuanError> {
        self.inner.stream_chat(request, self.name()).await
    }
}
