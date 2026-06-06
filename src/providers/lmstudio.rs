use async_trait::async_trait;
use serde::Deserialize;
use tracing::{debug, error};

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

const DEFAULT_BASE_URL: &str = "http://localhost:1234/api/v0";

// ── Native /api/v0/ model list ────────────────────────────────────────────────
//
// LM Studio exposes richer metadata at /api/v0/models than a plain OpenAI shim,
// including quantization, architecture, context length, and load state.

#[derive(Deserialize)]
struct LmsModelList {
    data: Vec<LmsModel>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct LmsModel {
    id: String,
    publisher: Option<String>,
    arch: Option<String>,
    quantization: Option<String>,
    max_context_length: Option<u32>,
}

// ── Provider ──────────────────────────────────────────────────────────────────

/// A provider that connects to a local [LM Studio](https://lmstudio.ai/) server
/// using its **native `/api/v0/` API**.
///
/// The native API returns richer model metadata (quantization, architecture,
/// context length, load state) than the OpenAI-compatible `/v1/` shim.
/// Chat completions follow the OpenAI response format.
///
/// LM Studio must be running with the local server enabled. No API key is
/// required by default, though LM Studio optionally supports one.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::LmStudioProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     // Assumes LM Studio is running on the default port
///     let provider = LmStudioProvider::new();
///
///     // List locally available models
///     let models = provider.models().await.unwrap();
///     for m in &models {
///         println!("{} (context: {:?})", m.id, m.context_length);
///     }
///
///     // Chat with the first loaded model
///     if let Some(model) = models.first() {
///         let request = ChatRequestBuilder::new(&model.id)
///             .message(ChatMessage::user("Hello!"))
///             .build()
///             .unwrap();
///         let response = provider.chat(&request).await.unwrap();
///         println!("{}", response.content().unwrap_or(""));
///     }
/// }
/// ```
pub struct LmStudioProvider {
    inner: OpenAICompatClient,
}

impl LmStudioProvider {
    /// Create a provider pointing at the default LM Studio address
    /// (`http://localhost:1234`). The `/api/v0` path prefix is added automatically.
    pub fn new() -> Self {
        Self {
            inner: OpenAICompatClient::no_key(DEFAULT_BASE_URL),
        }
    }

    /// Override the server address. Pass the root URL (e.g.
    /// `"http://localhost:5678"`) — the `/api/v0` prefix is appended automatically.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let b = base_url.into();
        self.inner.base_url = format!("{}/api/v0", b.trim_end_matches('/'));
        self
    }

    /// Set an API key if LM Studio's authentication is enabled.
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.inner.api_key = Some(key.into());
        self
    }
}

impl Default for LmStudioProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for LmStudioProvider {
    fn name(&self) -> &str {
        "lmstudio"
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, BaochuanError> {
        debug!("listing models from LM Studio native API");

        let response = self
            .inner
            .auth(self.inner.client.get(self.inner.models_url()))
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "LM Studio models error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let list: LmsModelList = response.json().await?;
        Ok(list
            .data
            .into_iter()
            .map(|m| {
                let display = match (&m.arch, &m.quantization) {
                    (Some(arch), Some(quant)) => Some(format!("{arch} · {quant}")),
                    (Some(arch), None) => Some(arch.clone()),
                    _ => None,
                };
                ModelInfo {
                    id: m.id,
                    owned_by: m.publisher,
                    context_length: m.max_context_length,
                    display_name: display,
                }
            })
            .collect())
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BaochuanError> {
        self.inner.chat(request, self.name()).await
    }

    async fn stream_chat(&self, request: &ChatRequest) -> Result<ChunkStream, BaochuanError> {
        self.inner.stream_chat(request, self.name()).await
    }
}
