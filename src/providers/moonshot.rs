use async_trait::async_trait;

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

/// International (default) Moonshot / Kimi endpoint.
const DEFAULT_BASE_URL: &str = "https://api.moonshot.ai/v1";

/// China-mainland Moonshot endpoint — use with [`MoonshotProvider::with_base_url`]
/// when your API key was issued on platform.moonshot.cn.
pub const CHINA_BASE_URL: &str = "https://api.moonshot.cn/v1";

/// A provider that connects to [Moonshot AI / Kimi](https://platform.kimi.ai/).
///
/// The API is OpenAI-compatible. Current models include `kimi-k3`, `kimi-k2.6`,
/// and `kimi-k2.7-code`. The default base URL is the international endpoint
/// (`api.moonshot.ai`); China-mainland keys must use
/// [`CHINA_BASE_URL`] via [`with_base_url`](Self::with_base_url).
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::MoonshotProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = MoonshotProvider::new(std::env::var("MOONSHOT_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("kimi-k3")
///         .message(ChatMessage::user("你好！"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct MoonshotProvider {
    inner: OpenAICompatClient,
}

impl MoonshotProvider {
    /// Create a new Moonshot AI provider (international endpoint).
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::MoonshotProvider::new(
    ///     std::env::var("MOONSHOT_API_KEY").expect("MOONSHOT_API_KEY not set"),
    /// );
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAICompatClient::with_key(api_key, DEFAULT_BASE_URL),
        }
    }

    /// Override the base URL (e.g. [`CHINA_BASE_URL`] for mainland China keys).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner.base_url = base_url.into();
        self
    }
}

#[async_trait]
impl Provider for MoonshotProvider {
    fn name(&self) -> &str {
        "moonshot"
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
