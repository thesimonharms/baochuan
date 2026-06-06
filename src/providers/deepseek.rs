use async_trait::async_trait;

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

const DEFAULT_BASE_URL: &str = "https://api.deepseek.com/v1";

/// A provider that connects to the [DeepSeek](https://platform.deepseek.com/) API.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::DeepSeekProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = DeepSeekProvider::new(std::env::var("DEEPSEEK_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("deepseek-chat")
///         .message(ChatMessage::user("What is the capital of France?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct DeepSeekProvider {
    inner: OpenAICompatClient,
}

impl DeepSeekProvider {
    /// Create a new DeepSeek provider with the given API key.
    ///
    /// The API key should be provided via an environment variable rather than
    /// being hard-coded:
    /// ```rust,no_run
    /// let provider = baochuan::providers::DeepSeekProvider::new(
    ///     std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY not set"),
    /// );
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAICompatClient::with_key(api_key, DEFAULT_BASE_URL),
        }
    }

    /// Override the base URL (useful for proxies or self-hosted instances).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner.base_url = base_url.into();
        self
    }
}

#[async_trait]
impl Provider for DeepSeekProvider {
    fn name(&self) -> &str {
        "deepseek"
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
