use async_trait::async_trait;

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

const DEFAULT_BASE_URL: &str = "https://api.fireworks.ai/inference/v1";

/// A provider that connects to [Fireworks AI](https://fireworks.ai/).
///
/// Fireworks serves fast, production-grade inference for open-weight models
/// via an OpenAI-compatible API. Notable models include
/// `accounts/fireworks/models/llama-v3p3-70b-instruct` and
/// `accounts/fireworks/models/deepseek-v3p1`.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::FireworksProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = FireworksProvider::new(std::env::var("FIREWORKS_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("accounts/fireworks/models/llama-v3p3-70b-instruct")
///         .message(ChatMessage::user("What is the capital of France?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct FireworksProvider {
    inner: OpenAICompatClient,
}

impl FireworksProvider {
    /// Create a new Fireworks AI provider.
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::FireworksProvider::new(
    ///     std::env::var("FIREWORKS_API_KEY").expect("FIREWORKS_API_KEY not set"),
    /// );
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAICompatClient::with_key(api_key, DEFAULT_BASE_URL),
        }
    }

    /// Override the base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner.base_url = base_url.into();
        self
    }
}

#[async_trait]
impl Provider for FireworksProvider {
    fn name(&self) -> &str {
        "fireworks"
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
