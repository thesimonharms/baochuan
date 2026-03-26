use serde::{Deserialize, Serialize};

use super::message::{ChatMessage, Role};
use super::tools::ToolCallDelta;

// ── Anthropic streaming helper types ─────────────────────────────────────────

/// Top-level SSE event envelope from the Anthropic streaming API.
#[derive(Debug, Deserialize)]
pub struct AnthropicStreamEvent {
    pub message: Option<AnthropicStreamMessage>,
    pub delta: Option<AnthropicStreamDelta>,
}

/// The `message` field inside a `message_start` event.
#[derive(Debug, Deserialize)]
pub struct AnthropicStreamMessage {
    pub id: Option<String>,
    pub model: Option<String>,
}

/// The `delta` field inside a `content_block_delta` event.
#[derive(Debug, Deserialize)]
pub struct AnthropicStreamDelta {
    #[serde(rename = "type")]
    pub delta_type: Option<String>,
    pub text: Option<String>,
}

/// A complete (non-streaming) chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<Usage>,
    /// Source URLs returned by search-augmented providers (e.g. Perplexity).
    /// `None` for all other providers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citations: Option<Vec<String>>,
}

impl ChatResponse {
    /// Returns the text content of the first choice, if present.
    pub fn content(&self) -> Option<&str> {
        self.choices.first().and_then(|c| c.message.content.as_str())
    }
}

/// A single completion choice in a [`ChatResponse`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

/// Token usage statistics for a request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Streaming types ───────────────────────────────────────────────────────────

/// A single chunk received during a streaming response (SSE).
#[derive(Debug, Clone, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

impl StreamChunk {
    /// Returns the incremental text delta of the first choice, if any.
    pub fn delta_content(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.delta.content.as_deref())
    }

    /// Returns true if this chunk signals the end of the stream.
    pub fn is_finished(&self) -> bool {
        self.choices
            .first()
            .and_then(|c| c.finish_reason.as_deref())
            .is_some()
    }
}

/// A streaming choice containing a partial delta.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

/// The incremental content delta within a stream chunk.
#[derive(Debug, Clone, Deserialize)]
pub struct Delta {
    pub role: Option<Role>,
    pub content: Option<String>,
    /// Streaming tool call fragments, if the model is calling a function.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}
