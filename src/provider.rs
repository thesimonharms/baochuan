use std::pin::Pin;

use async_trait::async_trait;
use futures_util::Stream;

use crate::error::BaochuanError;
use crate::types::{ChatRequest, ChatResponse, ModelInfo, StreamChunk, TtsRequest};

/// A boxed, send-safe stream of [`StreamChunk`] results.
pub type ChunkStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, BaochuanError>> + Send>>;

/// The core abstraction over any AI provider.
///
/// Implement this trait to add support for a new provider.
///
/// # Example — using a provider generically
/// ```rust,no_run
/// use baochuan::{Provider, ChatRequest, ChatMessage, ChatRequestBuilder};
///
/// async fn ask(provider: &dyn Provider, question: &str) -> String {
///     let req = ChatRequestBuilder::new("some-model")
///         .message(ChatMessage::user(question))
///         .build()
///         .unwrap();
///
///     provider.chat(&req).await.unwrap().content().unwrap_or("").to_string()
/// }
/// ```
#[async_trait]
pub trait Provider: Send + Sync {
    /// Send a blocking chat completion request and return the full response.
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BaochuanError>;

    /// Send a streaming chat completion request and return an async stream of chunks.
    ///
    /// The returned stream yields incremental [`StreamChunk`]s as the model
    /// generates its response. Callers can accumulate `delta_content()` from
    /// each chunk to build the full message.
    async fn stream_chat(&self, request: &ChatRequest) -> Result<ChunkStream, BaochuanError>;

    /// List all models available from this provider.
    ///
    /// Returns an empty [`Vec`] for providers that do not expose a model listing
    /// endpoint. Call this to discover what model IDs you can pass to
    /// [`ChatRequestBuilder::new`](crate::types::ChatRequestBuilder).
    async fn models(&self) -> Result<Vec<ModelInfo>, BaochuanError> {
        Ok(vec![])
    }

    /// Synthesise speech from text and return raw audio bytes.
    ///
    /// The audio format is determined by [`TtsRequest::format`]; the default
    /// when unset is provider-specific (usually MP3).
    ///
    /// Returns [`BaochuanError::InvalidRequest`] for providers that do not
    /// support text-to-speech. Currently implemented by [`OpenAIProvider`].
    ///
    /// [`OpenAIProvider`]: crate::providers::OpenAIProvider
    async fn tts(&self, _request: &TtsRequest) -> Result<Vec<u8>, BaochuanError> {
        Err(BaochuanError::InvalidRequest(format!(
            "{} does not support text-to-speech",
            self.name()
        )))
    }

    /// A human-readable identifier for this provider (e.g., `"deepseek"`, `"openrouter"`).
    fn name(&self) -> &str;
}
