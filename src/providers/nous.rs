use async_trait::async_trait;

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

const DEFAULT_BASE_URL: &str = "https://inference-api.nousresearch.com/v1";

/// A provider that connects to [Nous Portal](https://portal.nousresearch.com/).
///
/// Nous Portal exposes an OpenAI-compatible API at
/// `https://inference-api.nousresearch.com/v1`, so this provider delegates to
/// baochuan's shared OpenAI-compatible client.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::NousProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = NousProvider::new(std::env::var("NOUS_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("Hermes-4-405B")
///         .message(ChatMessage::user("Tell me about Zheng He's treasure fleet."))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct NousProvider {
    inner: OpenAICompatClient,
}

impl NousProvider {
    /// Create a new Nous Portal provider with the given API key.
    ///
    /// The API key should be provided via an environment variable rather than
    /// being hard-coded:
    /// ```rust,no_run
    /// let provider = baochuan::providers::NousProvider::new(
    ///     std::env::var("NOUS_API_KEY").expect("NOUS_API_KEY not set"),
    /// );
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAICompatClient::with_key(api_key, DEFAULT_BASE_URL),
        }
    }

    /// Override the base URL (useful for tests, proxies, or compatible gateways).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner.base_url = base_url.into();
        self
    }
}

#[async_trait]
impl Provider for NousProvider {
    fn name(&self) -> &str {
        "nous"
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
