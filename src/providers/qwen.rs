use async_trait::async_trait;

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

/// China / Beijing DashScope OpenAI-compatible endpoint (default).
const DEFAULT_BASE_URL: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1";

/// International (Singapore) DashScope OpenAI-compatible endpoint.
///
/// Use with [`QwenProvider::with_base_url`] when your key is from the
/// international Model Studio console.
pub const INTL_BASE_URL: &str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1";

/// A provider that connects to [Alibaba Cloud DashScope](https://dashscope.aliyun.com/)
/// to access **Qwen** (通义千问) models via the **OpenAI-compatible** API.
///
/// Defaults to the China endpoint. Pass [`INTL_BASE_URL`] to
/// [`with_base_url`](Self::with_base_url) for the international region.
/// Model listing (`GET /models`) is supported in compatible mode.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::QwenProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = QwenProvider::new(std::env::var("DASHSCOPE_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("qwen-plus")
///         .message(ChatMessage::user("你好！"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct QwenProvider {
    inner: OpenAICompatClient,
}

impl QwenProvider {
    /// Create a new Qwen / DashScope provider (China compatible-mode endpoint).
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::QwenProvider::new(
    ///     std::env::var("DASHSCOPE_API_KEY").expect("DASHSCOPE_API_KEY not set"),
    /// );
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAICompatClient::with_key(api_key, DEFAULT_BASE_URL),
        }
    }

    /// Override the base URL (e.g. [`INTL_BASE_URL`] for international).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner.base_url = base_url.into();
        self
    }
}

#[async_trait]
impl Provider for QwenProvider {
    fn name(&self) -> &str {
        "qwen"
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
