use async_trait::async_trait;
use reqwest::header;
use serde::Deserialize;
use tracing::{debug, error};

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::providers::sse::sse_to_chunks;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";

// OpenRouter returns richer model metadata than the plain OpenAI list.
#[derive(Deserialize)]
struct OpenRouterModelList {
    data: Vec<OpenRouterModel>,
}

#[derive(Deserialize)]
struct OpenRouterModel {
    id: String,
    name: Option<String>,
    context_length: Option<u32>,
}

/// A provider that connects to [OpenRouter](https://openrouter.ai/), giving
/// access to hundreds of models from many different AI providers through a
/// single unified API.
///
/// OpenRouter optionally accepts `HTTP-Referer` and `X-Title` headers to
/// identify your application — set these with the builder methods if you'd
/// like your app to appear in your OpenRouter dashboard.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::OpenRouterProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = OpenRouterProvider::new(std::env::var("OPENROUTER_API_KEY").unwrap())
///         .site_name("My App");
///
///     let request = ChatRequestBuilder::new("openai/gpt-4o")
///         .message(ChatMessage::user("What is the capital of France?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct OpenRouterProvider {
    inner: OpenAICompatClient,
    site_url: Option<String>,
    site_name: Option<String>,
}

impl OpenRouterProvider {
    /// Create a new OpenRouter provider with the given API key.
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::OpenRouterProvider::new(
    ///     std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY not set"),
    /// );
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAICompatClient::with_key(api_key, DEFAULT_BASE_URL),
            site_url: None,
            site_name: None,
        }
    }

    /// Override the base URL (useful for proxies).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner.base_url = base_url.into();
        self
    }

    /// Set the `HTTP-Referer` header — shown in OpenRouter analytics.
    pub fn site_url(mut self, url: impl Into<String>) -> Self {
        self.site_url = Some(url.into());
        self
    }

    /// Set the `X-Title` header — the display name of your app in OpenRouter.
    pub fn site_name(mut self, name: impl Into<String>) -> Self {
        self.site_name = Some(name.into());
        self
    }

    /// Apply the OpenRouter-specific headers to a request builder.
    fn apply_headers(&self, mut builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref url) = self.site_url {
            builder = builder.header("HTTP-Referer", url.as_str());
        }
        if let Some(ref name) = self.site_name {
            builder = builder.header("X-Title", name.as_str());
        }
        builder
    }
}

#[async_trait]
impl Provider for OpenRouterProvider {
    fn name(&self) -> &str {
        "openrouter"
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, BaochuanError> {
        let response = self
            .inner
            .auth(self.inner.client.get(self.inner.models_url()))
            .header(header::CONTENT_TYPE, "application/json")
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(BaochuanError::Api { status: status.as_u16(), message: body });
        }

        let list: OpenRouterModelList = response.json().await?;
        Ok(list.data.into_iter().map(|m| ModelInfo {
            id: m.id,
            owned_by: None,
            context_length: m.context_length,
            display_name: m.name,
        }).collect())
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BaochuanError> {
        debug!(model = %request.model, "sending chat request to OpenRouter");

        let response = self.apply_headers(
            self.inner
                .auth(self.inner.client.post(self.inner.chat_url()))
                .header(header::CONTENT_TYPE, "application/json")
                .json(request),
        )
        .send()
        .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "OpenRouter API error");
            return Err(BaochuanError::Api { status: status.as_u16(), message: body });
        }

        let chat_response: ChatResponse = response.json().await?;
        debug!(id = %chat_response.id, "received OpenRouter response");
        Ok(chat_response)
    }

    async fn stream_chat(&self, request: &ChatRequest) -> Result<ChunkStream, BaochuanError> {
        debug!(model = %request.model, "starting streaming chat request to OpenRouter");

        let mut body = serde_json::to_value(request)?;
        body["stream"] = serde_json::Value::Bool(true);

        let response = self.apply_headers(
            self.inner
                .auth(self.inner.client.post(self.inner.chat_url()))
                .header(header::CONTENT_TYPE, "application/json")
                .json(&body),
        )
        .send()
        .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "OpenRouter stream error");
            return Err(BaochuanError::Api { status: status.as_u16(), message: body });
        }

        Ok(Box::pin(sse_to_chunks(response.bytes_stream())))
    }
}
