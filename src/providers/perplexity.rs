use async_trait::async_trait;
use serde::Deserialize;
use tracing::{debug, error};

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::types::{ChatRequest, ChatResponse, ModelInfo};

const DEFAULT_BASE_URL: &str = "https://api.perplexity.ai";

// ── Native response type ──────────────────────────────────────────────────────
//
// Perplexity's API follows the OpenAI wire format but adds a `citations` array
// of source URLs for online/search-augmented models.

#[derive(Deserialize)]
struct PerplexityResponse {
    id: String,
    model: String,
    choices: Vec<crate::types::ChatChoice>,
    usage: Option<crate::types::Usage>,
    #[serde(default)]
    citations: Vec<String>,
}

#[derive(Deserialize)]
struct PerplexityModelList {
    data: Vec<PerplexityModel>,
}

#[derive(Deserialize)]
struct PerplexityModel {
    id: String,
    owned_by: Option<String>,
    context_length: Option<u32>,
}

// ── Provider ──────────────────────────────────────────────────────────────────

/// A provider that connects to the [Perplexity AI](https://www.perplexity.ai/) API.
///
/// Perplexity's Sonar API follows the OpenAI chat completions format (also
/// available at `POST /v1/sonar`), with one important addition: search-augmented
/// models such as `sonar` and `sonar-pro` include a `citations` array of source
/// URLs in the response. These are surfaced via [`ChatResponse::citations`].
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::PerplexityProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = PerplexityProvider::new(std::env::var("PERPLEXITY_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("sonar-pro")
///         .message(ChatMessage::user("What happened in the news today?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
///
///     if let Some(citations) = &response.citations {
///         for url in citations {
///             println!("  source: {url}");
///         }
///     }
/// }
/// ```
pub struct PerplexityProvider {
    inner: OpenAICompatClient,
}

impl PerplexityProvider {
    /// Create a new Perplexity provider.
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::PerplexityProvider::new(
    ///     std::env::var("PERPLEXITY_API_KEY").expect("PERPLEXITY_API_KEY not set"),
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
impl Provider for PerplexityProvider {
    fn name(&self) -> &str {
        "perplexity"
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, BaochuanError> {
        debug!("listing models from Perplexity");

        let response = self
            .inner
            .auth(self.inner.client.get(self.inner.models_url()))
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "Perplexity models error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let list: PerplexityModelList = response.json().await?;
        Ok(list
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id,
                owned_by: m.owned_by,
                context_length: m.context_length,
                display_name: None,
            })
            .collect())
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BaochuanError> {
        debug!(model = %request.model, "sending chat request to Perplexity");

        let response = self
            .inner
            .auth(self.inner.client.post(self.inner.chat_url()))
            .json(request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "Perplexity API error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let ppl: PerplexityResponse = response.json().await?;
        debug!(id = %ppl.id, "received Perplexity response");
        Ok(ChatResponse {
            id: ppl.id,
            model: ppl.model,
            choices: ppl.choices,
            usage: ppl.usage,
            citations: if ppl.citations.is_empty() {
                None
            } else {
                Some(ppl.citations)
            },
        })
    }

    async fn stream_chat(&self, request: &ChatRequest) -> Result<ChunkStream, BaochuanError> {
        // Citations are not included in streaming chunks; use chat() for grounded responses.
        self.inner.stream_chat(request, self.name()).await
    }
}
