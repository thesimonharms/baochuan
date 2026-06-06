use serde::{Deserialize, Serialize};

use super::tools::ToolCall;

// ── Role ──────────────────────────────────────────────────────────────────────

/// The role of a participant in a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

// ── MessageContent ────────────────────────────────────────────────────────────

/// The content of a [`ChatMessage`]: either plain text or a list of content
/// parts (text, images, audio, and/or documents).
///
/// Serialises as a JSON string for `Text` and as a JSON array for `Parts`,
/// matching the OpenAI multimodal wire format expected by most providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain-text content.
    Text(String),
    /// Multipart content containing text, images, audio, and/or document parts.
    Parts(Vec<ContentPart>),
}

impl Default for MessageContent {
    fn default() -> Self {
        MessageContent::Text(String::new())
    }
}

impl MessageContent {
    /// Returns a reference to the first text segment, if any.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s.as_str()),
            Self::Parts(parts) => parts.iter().find_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            }),
        }
    }

    /// Concatenates all text parts into a single `String`, skipping non-text
    /// parts. Useful as a fallback for providers that only support text input.
    pub fn to_text_lossy(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

// ── ContentPart ───────────────────────────────────────────────────────────────

/// A single content part within a multipart [`ChatMessage`].
///
/// Serialises with `"type"` as a discriminant tag, following the OpenAI
/// multimodal format. Providers that don't support a given modality will
/// silently drop unsupported parts and use only the text.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// A text segment.
    Text { text: String },

    /// An image referenced by URL or inline data URL (`data:image/jpeg;base64,...`).
    ///
    /// Supported by: OpenAI, Anthropic, Gemini, Grok, Mistral, DeepSeek,
    /// OpenRouter, Moonshot, Perplexity, llama.cpp, LM Studio, Ollama (data URLs only).
    ImageUrl { image_url: ImageUrl },

    /// Audio input for speech-capable models (`data:audio/wav;base64,...` or
    /// base64 data directly).
    ///
    /// Supported by: OpenAI (GPT-4o audio), Gemini.
    /// Other providers fall back to text-only.
    InputAudio { input_audio: AudioInput },

    /// A document (PDF or plain text) for document-capable models.
    ///
    /// Supported by: Anthropic, Gemini.
    /// Other providers fall back to text-only.
    Document { document: DocumentInput },
}

impl ContentPart {
    /// Create a text part.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create an image part from a URL or data URL.
    ///
    /// - HTTPS URL: `"https://example.com/photo.jpg"`
    /// - Data URL:  `"data:image/jpeg;base64,/9j/4AAQ..."`
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }

    /// Create an audio input part from base64-encoded audio data.
    ///
    /// `format` is the audio format string, e.g. `"wav"`, `"mp3"`, `"flac"`,
    /// `"opus"`, `"aac"`, `"pcm16"`.
    pub fn audio(data: impl Into<String>, format: impl Into<String>) -> Self {
        Self::InputAudio {
            input_audio: AudioInput {
                data: data.into(),
                format: format.into(),
            },
        }
    }

    /// Create a document part from base64-encoded document data.
    ///
    /// `media_type` is the MIME type, e.g. `"application/pdf"` or `"text/plain"`.
    pub fn document(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::Document {
            document: DocumentInput {
                data: data.into(),
                media_type: media_type.into(),
            },
        }
    }
}

// ── ImageUrl ──────────────────────────────────────────────────────────────────

/// An image reference used inside a [`ContentPart::ImageUrl`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// HTTPS image URL or a `data:image/...;base64,...` data URL.
    pub url: String,
    /// Optional resolution hint (`"low"`, `"high"`, or `"auto"`).
    /// Supported by OpenAI vision models; ignored by other providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

// ── AudioInput ────────────────────────────────────────────────────────────────

/// Base64-encoded audio for use as a [`ContentPart::InputAudio`].
///
/// OpenAI wire format: `{"type":"input_audio","input_audio":{"data":"...","format":"wav"}}`.
/// Gemini maps this to `inlineData` with the appropriate audio MIME type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInput {
    /// Base64-encoded audio bytes (no data-URL prefix needed).
    pub data: String,
    /// Audio codec / container: `"wav"`, `"mp3"`, `"flac"`, `"opus"`, `"aac"`, `"pcm16"`.
    pub format: String,
}

impl AudioInput {
    /// Returns the MIME type corresponding to this audio format, e.g. `"audio/wav"`.
    pub fn mime_type(&self) -> String {
        match self.format.as_str() {
            "mp3" => "audio/mpeg",
            "flac" => "audio/flac",
            "opus" => "audio/ogg; codecs=opus",
            "aac" => "audio/aac",
            "pcm16" => "audio/pcm",
            _ => "audio/wav", // default
        }
        .to_string()
    }
}

// ── DocumentInput ─────────────────────────────────────────────────────────────

/// Base64-encoded document for use as a [`ContentPart::Document`].
///
/// Supported by Anthropic (PDF, plain text) and Gemini (PDF and more).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentInput {
    /// Base64-encoded document bytes.
    pub data: String,
    /// MIME type of the document, e.g. `"application/pdf"` or `"text/plain"`.
    pub media_type: String,
}

// ── AudioOutput ───────────────────────────────────────────────────────────────

/// Audio data returned by a model (e.g. GPT-4o with `modalities: ["audio"]`).
///
/// Appears as `ChatMessage::audio` when a model produces audio output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioOutput {
    /// Provider-assigned audio clip identifier.
    pub id: Option<String>,
    /// Base64-encoded audio bytes.
    pub data: String,
    /// Unix timestamp at which the audio data expires from the provider's servers.
    pub expires_at: Option<u64>,
    /// Text transcript of the audio, if included by the provider.
    pub transcript: Option<String>,
}

// ── ChatMessage ───────────────────────────────────────────────────────────────

/// A single message in a conversation thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    /// Text or multipart content. Deserialised as empty string when the
    /// provider returns `null` (e.g. GPT-4o audio-only responses).
    #[serde(default, deserialize_with = "deser_nullable_content")]
    pub content: MessageContent,
    /// Audio output, present only when a model produces audio (e.g. GPT-4o
    /// with `modalities: ["audio"]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioOutput>,
    /// Tool calls requested by the model in an assistant message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID this message is responding to (used in `Role::Tool` messages).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

fn deser_nullable_content<'de, D>(d: D) -> Result<MessageContent, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt = Option::<MessageContent>::deserialize(d)?;
    Ok(opt.unwrap_or_default())
}

impl ChatMessage {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: MessageContent::Text(content.into()),
            audio: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a user text message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(content.into()),
            audio: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create an assistant text message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content.into()),
            audio: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a tool result message to send back after a model requested a tool call.
    ///
    /// `tool_call_id` must match the [`ToolCall::id`](crate::types::ToolCall::id) from the
    /// assistant message. `content` is the serialised result of calling the function.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: MessageContent::Text(content.into()),
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
            audio: None,
        }
    }

    /// Create a user message containing text followed by a single image.
    ///
    /// `image_url` can be an HTTPS URL or a `data:image/...;base64,...` URL.
    pub fn user_with_image(text: impl Into<String>, image_url: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Parts(vec![
                ContentPart::text(text),
                ContentPart::image_url(image_url),
            ]),
            audio: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a message with arbitrary content parts.
    pub fn with_parts(role: Role, parts: Vec<ContentPart>) -> Self {
        Self {
            role,
            content: MessageContent::Parts(parts),
            audio: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}
