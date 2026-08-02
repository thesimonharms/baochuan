use async_trait::async_trait;

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

const DEFAULT_BASE_URL: &str = "https://api.mistral.ai/v1";

/// A provider that connects to the [Mistral AI](https://mistral.ai/) API.
///
/// The Mistral API is OpenAI-compatible. Notable models include
/// `mistral-large-latest`, `mistral-medium-latest`, and `mistral-small-latest`.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::MistralProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = MistralProvider::new(std::env::var("MISTRAL_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("mistral-large-latest")
///         .message(ChatMessage::user("What is the capital of France?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct MistralProvider {
    inner: OpenAICompatClient,
}

impl MistralProvider {
    /// Create a new Mistral provider.
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::MistralProvider::new(
    ///     std::env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY not set"),
    /// );
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAICompatClient::with_key(api_key, DEFAULT_BASE_URL),
        }
    }

    /// Override the base URL (e.g. for self-hosted Mistral).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner.base_url = base_url.into();
        self
    }
}

#[async_trait]
impl Provider for MistralProvider {
    fn name(&self) -> &str {
        "mistral"
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
