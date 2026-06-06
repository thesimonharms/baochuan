use async_trait::async_trait;

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

const DEFAULT_BASE_URL: &str = "http://localhost:8080/v1";

/// A provider that connects to a running
/// [llama.cpp server](https://github.com/ggerganov/llama.cpp/tree/master/tools/server)
/// (`llama-server`).
///
/// llama.cpp's server implements the OpenAI chat completions API at `/v1/`,
/// so this provider is a thin wrapper around that interface. No API key is
/// required.
///
/// Start the server with:
/// ```bash
/// llama-server -m your-model.gguf --port 8080
/// ```
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::LlamaCppProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = LlamaCppProvider::new();
///
///     let request = ChatRequestBuilder::new("llama") // model ID returned by /v1/models
///         .message(ChatMessage::user("What is 2 + 2?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct LlamaCppProvider {
    inner: OpenAICompatClient,
}

impl LlamaCppProvider {
    /// Create a provider pointing at the default llama.cpp server address
    /// (`http://localhost:8080`). The `/v1` path prefix is added automatically.
    pub fn new() -> Self {
        Self {
            inner: OpenAICompatClient::no_key(DEFAULT_BASE_URL),
        }
    }

    /// Override the server address. Pass the root URL (e.g.
    /// `"http://localhost:9090"`) — the `/v1` prefix is appended automatically.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let b = base_url.into();
        self.inner.base_url = format!("{}/v1", b.trim_end_matches('/'));
        self
    }
}

impl Default for LlamaCppProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for LlamaCppProvider {
    fn name(&self) -> &str {
        "llamacpp"
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
