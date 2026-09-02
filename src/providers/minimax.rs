use async_trait::async_trait;

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

/// International (default) MiniMax endpoint.
const DEFAULT_BASE_URL: &str = "https://api.minimax.io/v1";

/// China-mainland MiniMax endpoint — use with [`MiniMaxProvider::with_base_url`]
/// when your API key was issued on the mainland console.
pub const CHINA_BASE_URL: &str = "https://api.minimaxi.com/v1";

/// A provider that connects to [MiniMax](https://www.minimax.io/).
///
/// The API is OpenAI-compatible. Current models include `MiniMax-M3`.
/// The default base URL is the international endpoint (`api.minimax.io`);
/// China-mainland keys must use [`CHINA_BASE_URL`] via
/// [`with_base_url`](Self::with_base_url).
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::MiniMaxProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = MiniMaxProvider::new(std::env::var("MINIMAX_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("MiniMax-M3")
///         .message(ChatMessage::user("你好！"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct MiniMaxProvider {
    inner: OpenAICompatClient,
}

impl MiniMaxProvider {
    /// Create a new MiniMax provider (international endpoint).
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::MiniMaxProvider::new(
    ///     std::env::var("MINIMAX_API_KEY").expect("MINIMAX_API_KEY not set"),
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
impl Provider for MiniMaxProvider {
    fn name(&self) -> &str {
        "minimax"
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
