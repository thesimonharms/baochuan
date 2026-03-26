use async_trait::async_trait;
use tracing::{debug, error};

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::providers::sse::sse_to_chunks;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

const DEFAULT_BASE_URL: &str = "https://api.githubcopilot.com";

/// A provider that connects to the [GitHub Copilot](https://github.com/features/copilot) API.
///
/// The Copilot API is OpenAI-compatible and exposes a variety of models
/// including GPT-4o, Claude 3.5 Sonnet, o1, and others depending on your
/// Copilot subscription.
///
/// Authentication uses a GitHub personal access token (classic or fine-grained)
/// with Copilot access, or the `GITHUB_TOKEN` available in GitHub Actions.
///
/// Optionally set `editor_version` and `integration_id` to identify your
/// application in Copilot telemetry.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::CopilotProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = CopilotProvider::new(std::env::var("GITHUB_TOKEN").unwrap());
///
///     let request = ChatRequestBuilder::new("gpt-4o")
///         .message(ChatMessage::user("What is the capital of France?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct CopilotProvider {
    inner: OpenAICompatClient,
    editor_version: Option<String>,
    integration_id: Option<String>,
}

impl CopilotProvider {
    /// Create a new GitHub Copilot provider.
    ///
    /// `token` is a GitHub personal access token (classic or fine-grained) or
    /// the `GITHUB_TOKEN` from a GitHub Actions workflow.
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::CopilotProvider::new(
    ///     std::env::var("GITHUB_TOKEN").expect("GITHUB_TOKEN not set"),
    /// );
    /// ```
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            inner: OpenAICompatClient::with_key(token, DEFAULT_BASE_URL),
            editor_version: None,
            integration_id: None,
        }
    }

    /// Override the base URL (useful for proxies or enterprise GitHub instances).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner.base_url = base_url.into();
        self
    }

    /// Set the `Editor-Version` header to identify your editor in Copilot telemetry.
    ///
    /// Example: `"vscode/1.90.0"`, `"neovim/0.10.0"`.
    pub fn editor_version(mut self, version: impl Into<String>) -> Self {
        self.editor_version = Some(version.into());
        self
    }

    /// Set the `Copilot-Integration-Id` header to identify your integration.
    pub fn integration_id(mut self, id: impl Into<String>) -> Self {
        self.integration_id = Some(id.into());
        self
    }

    fn apply_headers(&self, mut builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref v) = self.editor_version {
            builder = builder.header("Editor-Version", v.as_str());
        }
        if let Some(ref id) = self.integration_id {
            builder = builder.header("Copilot-Integration-Id", id.as_str());
        }
        builder
    }
}

#[async_trait]
impl Provider for CopilotProvider {
    fn name(&self) -> &str {
        "copilot"
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, BaochuanError> {
        self.inner.models().await
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BaochuanError> {
        debug!(model = %request.model, "sending chat request to GitHub Copilot");

        let response = self
            .apply_headers(
                self.inner
                    .auth(self.inner.client.post(self.inner.chat_url()))
                    .json(request),
            )
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "Copilot API error");
            return Err(BaochuanError::Api { status: status.as_u16(), message: body });
        }

        let resp: ChatResponse = response.json().await?;
        debug!(id = %resp.id, "received Copilot response");
        Ok(resp)
    }

    async fn stream_chat(&self, request: &ChatRequest) -> Result<ChunkStream, BaochuanError> {
        debug!(model = %request.model, "starting streaming chat request to GitHub Copilot");

        let mut body = serde_json::to_value(request)?;
        body["stream"] = serde_json::Value::Bool(true);

        let response = self
            .apply_headers(
                self.inner
                    .auth(self.inner.client.post(self.inner.chat_url()))
                    .json(&body),
            )
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "Copilot stream error");
            return Err(BaochuanError::Api { status: status.as_u16(), message: body });
        }

        Ok(Box::pin(sse_to_chunks(response.bytes_stream())))
    }
}
