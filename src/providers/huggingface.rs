use async_trait::async_trait;

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

const DEFAULT_BASE_URL: &str = "https://router.huggingface.co/v1";

/// A provider that connects to [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers).
///
/// The Hugging Face router exposes an OpenAI-compatible API that proxies to
/// the inference backend hosting each model. Model IDs use the familiar
/// `org/model` Hub naming, e.g. `meta-llama/Llama-3.3-70B-Instruct`.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::HuggingFaceProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = HuggingFaceProvider::new(std::env::var("HF_TOKEN").unwrap());
///
///     let request = ChatRequestBuilder::new("meta-llama/Llama-3.3-70B-Instruct")
///         .message(ChatMessage::user("What is the capital of France?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct HuggingFaceProvider {
    inner: OpenAICompatClient,
}

impl HuggingFaceProvider {
    /// Create a new Hugging Face provider.
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::HuggingFaceProvider::new(
    ///     std::env::var("HF_TOKEN").expect("HF_TOKEN not set"),
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
impl Provider for HuggingFaceProvider {
    fn name(&self) -> &str {
        "huggingface"
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
