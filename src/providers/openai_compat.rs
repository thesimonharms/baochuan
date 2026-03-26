//! Shared HTTP client for OpenAI-compatible providers.
//!
//! [`OpenAICompatClient`] handles request construction, auth, error handling,
//! and SSE streaming. Individual provider structs contain one of these and
//! delegate to it, only overriding what differs (provider name, default URL,
//! custom response parsing, extra headers, etc.).

use reqwest::Client;
use tracing::{debug, error};

use crate::error::BaochuanError;
use crate::provider::ChunkStream;
use crate::providers::helpers::fetch_openai_models;
use crate::providers::sse::sse_to_chunks;
use crate::types::{ChatRequest, ChatResponse, ModelInfo, TtsRequest};

/// Shared HTTP client for OpenAI-compatible providers.
///
/// Handles bearer auth, JSON serialization, error handling, and SSE streaming.
/// Provider structs wrap one of these and delegate common methods to it.
pub(crate) struct OpenAICompatClient {
    pub(crate) client: Client,
    pub(crate) api_key: Option<String>,
    pub(crate) base_url: String,
}

impl OpenAICompatClient {
    /// Create a client with a required API key.
    pub fn with_key(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self { client: Client::new(), api_key: Some(api_key.into()), base_url: base_url.into() }
    }

    /// Create a client with no API key (e.g. llama.cpp, local LM Studio without auth).
    pub fn no_key(base_url: impl Into<String>) -> Self {
        Self { client: Client::new(), api_key: None, base_url: base_url.into() }
    }

    /// Apply bearer auth to a request builder if an API key is set.
    pub(crate) fn auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => builder.bearer_auth(key),
            None => builder,
        }
    }

    pub fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    pub fn models_url(&self) -> String {
        format!("{}/models", self.base_url)
    }

    pub fn tts_url(&self) -> String {
        format!("{}/audio/speech", self.base_url)
    }

    /// Send a non-streaming chat request and deserialize the response.
    pub async fn chat(
        &self,
        request: &ChatRequest,
        provider_name: &str,
    ) -> Result<ChatResponse, BaochuanError> {
        debug!(model = %request.model, provider = %provider_name, "sending chat request");

        let response = self
            .auth(self.client.post(self.chat_url()))
            .json(request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, provider = %provider_name, "API error");
            return Err(BaochuanError::Api { status: status.as_u16(), message: body });
        }

        let resp: ChatResponse = response.json().await?;
        debug!(id = %resp.id, "received response");
        Ok(resp)
    }

    /// Send a streaming chat request and return an SSE chunk stream.
    pub async fn stream_chat(
        &self,
        request: &ChatRequest,
        provider_name: &str,
    ) -> Result<ChunkStream, BaochuanError> {
        debug!(model = %request.model, provider = %provider_name, "starting streaming chat");

        let mut body = serde_json::to_value(request)?;
        body["stream"] = serde_json::Value::Bool(true);

        let response = self
            .auth(self.client.post(self.chat_url()))
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, provider = %provider_name, "stream error");
            return Err(BaochuanError::Api { status: status.as_u16(), message: body });
        }

        Ok(Box::pin(sse_to_chunks(response.bytes_stream())))
    }

    /// Fetch models from the standard OpenAI-compatible `/models` endpoint.
    pub async fn models(&self) -> Result<Vec<ModelInfo>, BaochuanError> {
        fetch_openai_models(self.auth(self.client.get(self.models_url()))).await
    }

    /// Send a TTS request and return the raw audio bytes.
    pub async fn tts(
        &self,
        request: &TtsRequest,
        provider_name: &str,
    ) -> Result<Vec<u8>, BaochuanError> {
        debug!(
            model = %request.model, voice = %request.voice,
            provider = %provider_name, "sending TTS request"
        );

        let response = self
            .auth(self.client.post(self.tts_url()))
            .json(request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, provider = %provider_name, "TTS error");
            return Err(BaochuanError::Api { status: status.as_u16(), message: body });
        }

        Ok(response.bytes().await?.to_vec())
    }
}
