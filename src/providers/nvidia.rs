use async_trait::async_trait;

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

const DEFAULT_BASE_URL: &str = "https://integrate.api.nvidia.com/v1";

/// A provider that connects to [NVIDIA NIM](https://build.nvidia.com/).
///
/// NVIDIA's hosted NIM microservices expose an OpenAI-compatible API.
/// Notable models include `meta/llama-3.3-70b-instruct` and
/// `nvidia/llama-3.3-nemotron-super-49b-v1`.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::NvidiaProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = NvidiaProvider::new(std::env::var("NVIDIA_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("meta/llama-3.3-70b-instruct")
///         .message(ChatMessage::user("What is the capital of France?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct NvidiaProvider {
    inner: OpenAICompatClient,
}

impl NvidiaProvider {
    /// Create a new NVIDIA NIM provider.
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::NvidiaProvider::new(
    ///     std::env::var("NVIDIA_API_KEY").expect("NVIDIA_API_KEY not set"),
    /// );
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAICompatClient::with_key(api_key, DEFAULT_BASE_URL),
        }
    }

    /// Override the base URL (e.g. for a self-hosted NIM endpoint).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner.base_url = base_url.into();
        self
    }
}

#[async_trait]
impl Provider for NvidiaProvider {
    fn name(&self) -> &str {
        "nvidia"
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
