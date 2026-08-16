use async_trait::async_trait;

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

const DEFAULT_BASE_URL: &str = "https://api.deepinfra.com/v1/openai";

/// A provider that connects to [DeepInfra](https://deepinfra.com/).
///
/// DeepInfra offers low-cost inference for open-weight models via an
/// OpenAI-compatible API. Notable models include
/// `meta-llama/Meta-Llama-3.3-70B-Instruct` and `deepseek-ai/DeepSeek-V3.1`.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::DeepInfraProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = DeepInfraProvider::new(std::env::var("DEEPINFRA_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("meta-llama/Meta-Llama-3.3-70B-Instruct")
///         .message(ChatMessage::user("What is the capital of France?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct DeepInfraProvider {
    inner: OpenAICompatClient,
}

impl DeepInfraProvider {
    /// Create a new DeepInfra provider.
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::DeepInfraProvider::new(
    ///     std::env::var("DEEPINFRA_API_KEY").expect("DEEPINFRA_API_KEY not set"),
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
impl Provider for DeepInfraProvider {
    fn name(&self) -> &str {
        "deepinfra"
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
