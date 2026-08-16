use async_trait::async_trait;

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

const DEFAULT_BASE_URL: &str = "https://api.cerebras.ai/v1";

/// A provider that connects to [Cerebras Inference](https://inference.cerebras.ai/).
///
/// Cerebras runs models on its wafer-scale engine for record-breaking
/// generation speed, exposed through an OpenAI-compatible API. Notable
/// models include `llama-3.3-70b` and `gpt-oss-120b`.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::CerebrasProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = CerebrasProvider::new(std::env::var("CEREBRAS_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("llama-3.3-70b")
///         .message(ChatMessage::user("What is the capital of France?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct CerebrasProvider {
    inner: OpenAICompatClient,
}

impl CerebrasProvider {
    /// Create a new Cerebras provider.
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::CerebrasProvider::new(
    ///     std::env::var("CEREBRAS_API_KEY").expect("CEREBRAS_API_KEY not set"),
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
impl Provider for CerebrasProvider {
    fn name(&self) -> &str {
        "cerebras"
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
