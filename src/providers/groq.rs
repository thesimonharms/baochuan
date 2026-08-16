use async_trait::async_trait;

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

const DEFAULT_BASE_URL: &str = "https://api.groq.com/openai/v1";

/// A provider that connects to [Groq](https://groq.com/).
///
/// Groq serves ultra-low-latency inference on its LPU hardware via an
/// OpenAI-compatible API. Notable models include `llama-3.3-70b-versatile`
/// and `openai/gpt-oss-120b`.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::GroqProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = GroqProvider::new(std::env::var("GROQ_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("llama-3.3-70b-versatile")
///         .message(ChatMessage::user("What is the capital of France?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct GroqProvider {
    inner: OpenAICompatClient,
}

impl GroqProvider {
    /// Create a new Groq provider.
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::GroqProvider::new(
    ///     std::env::var("GROQ_API_KEY").expect("GROQ_API_KEY not set"),
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
impl Provider for GroqProvider {
    fn name(&self) -> &str {
        "groq"
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
