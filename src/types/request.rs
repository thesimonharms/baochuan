use serde::{Deserialize, Serialize};

use super::message::ChatMessage;
use super::tools::{Tool, ToolChoice};
use crate::error::BaochuanError;

// ── ThinkingConfig ────────────────────────────────────────────────────────────

/// Controls whether a reasoning/thinking model should produce chain-of-thought
/// tokens (DeepSeek V4, Kimi, and similar OpenAI-compatible APIs).
///
/// # Example
/// ```rust
/// use baochuan::types::{ChatMessage, ChatRequestBuilder, ThinkingConfig};
///
/// let request = ChatRequestBuilder::new("deepseek-v4-flash")
///     .message(ChatMessage::user("Solve 2+2."))
///     .thinking(ThinkingConfig::enabled())
///     .reasoning_effort("high")
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// `"enabled"` or `"disabled"`.
    #[serde(rename = "type")]
    pub thinking_type: String,
}

impl ThinkingConfig {
    /// Enable thinking / chain-of-thought generation.
    pub fn enabled() -> Self {
        Self {
            thinking_type: "enabled".to_string(),
        }
    }

    /// Disable thinking (non-reasoning chat mode).
    pub fn disabled() -> Self {
        Self {
            thinking_type: "disabled".to_string(),
        }
    }
}

// ── ChatRequest ───────────────────────────────────────────────────────────────

/// A fully constructed chat completion request ready to send.
#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Preferred token limit for OpenAI (including o-series / reasoning models).
    /// When set, [`OpenAIProvider`](crate::providers::OpenAIProvider) sends this
    /// instead of `max_tokens`. Other OpenAI-compatible providers that understand
    /// the field will receive it as-is.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Output modalities to request. Use `["text", "audio"]` to enable audio
    /// output on models that support it (e.g. GPT-4o audio).
    /// `None` → provider default (text only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    /// Audio output configuration required when `modalities` includes `"audio"`.
    #[serde(rename = "audio", skip_serializing_if = "Option::is_none")]
    pub audio_output: Option<AudioOutputConfig>,
    /// Tools (functions) the model may call. Supported by most providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Controls whether/how the model uses tools. Defaults to `"auto"` when
    /// tools are provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Enable or disable thinking mode (DeepSeek V4, Kimi, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    /// Reasoning effort hint (`"low"`, `"medium"`, `"high"`, `"max"`) for
    /// providers that support it (DeepSeek, xAI, OpenAI reasoning models, …).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

/// Configuration for audio output in a chat completion (GPT-4o audio).
#[derive(Debug, Clone, Serialize)]
pub struct AudioOutputConfig {
    /// Voice to use (e.g. `"alloy"`, `"echo"`, `"fable"`, `"onyx"`, `"nova"`, `"shimmer"`).
    pub voice: String,
    /// Output format: `"wav"`, `"mp3"`, `"flac"`, `"opus"`, `"pcm16"`.
    pub format: String,
}

// ── ChatRequestBuilder ────────────────────────────────────────────────────────

/// Builder for [`ChatRequest`].
///
/// # Example
/// ```rust
/// use baochuan::types::{ChatMessage, ChatRequestBuilder};
///
/// let request = ChatRequestBuilder::new("deepseek-v4-flash")
///     .message(ChatMessage::user("Hello!"))
///     .max_tokens(512)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Default)]
pub struct ChatRequestBuilder {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    stream: bool,
    max_tokens: Option<u32>,
    max_completion_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    modalities: Option<Vec<String>>,
    audio_output: Option<AudioOutputConfig>,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
    thinking: Option<ThinkingConfig>,
    reasoning_effort: Option<String>,
}

impl ChatRequestBuilder {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
            ..Default::default()
        }
    }

    pub fn message(mut self, message: ChatMessage) -> Self {
        self.messages.push(message);
        self
    }

    pub fn messages(mut self, messages: Vec<ChatMessage>) -> Self {
        self.messages = messages;
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set `max_completion_tokens` (preferred by OpenAI, including o-series).
    pub fn max_completion_tokens(mut self, max_completion_tokens: u32) -> Self {
        self.max_completion_tokens = Some(max_completion_tokens);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Request specific output modalities (e.g. `["text", "audio"]`).
    /// Required to enable audio output on GPT-4o audio models.
    pub fn modalities(mut self, modalities: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.modalities = Some(modalities.into_iter().map(Into::into).collect());
        self
    }

    /// Configure audio output for models that support it.
    /// Must be combined with `.modalities(["text", "audio"])`.
    pub fn audio_output(mut self, voice: impl Into<String>, format: impl Into<String>) -> Self {
        self.audio_output = Some(AudioOutputConfig {
            voice: voice.into(),
            format: format.into(),
        });
        self
    }

    /// Add a single tool (function) the model may call.
    pub fn tool(mut self, tool: Tool) -> Self {
        self.tools.get_or_insert_with(Vec::new).push(tool);
        self
    }

    /// Set all tools at once.
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Control whether/how the model uses tools.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Enable or disable thinking mode for reasoning models (DeepSeek V4, Kimi, …).
    pub fn thinking(mut self, thinking: ThinkingConfig) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Set reasoning effort (`"low"`, `"medium"`, `"high"`, `"max"`).
    pub fn reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    pub fn build(self) -> Result<ChatRequest, BaochuanError> {
        let model = self
            .model
            .ok_or_else(|| BaochuanError::InvalidRequest("model must be specified".to_string()))?;

        if self.messages.is_empty() {
            return Err(BaochuanError::InvalidRequest(
                "at least one message is required".to_string(),
            ));
        }

        Ok(ChatRequest {
            model,
            messages: self.messages,
            stream: self.stream,
            max_tokens: self.max_tokens,
            max_completion_tokens: self.max_completion_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            modalities: self.modalities,
            audio_output: self.audio_output,
            tools: self.tools,
            tool_choice: self.tool_choice,
            thinking: self.thinking,
            reasoning_effort: self.reasoning_effort,
        })
    }
}

// ── TtsRequest ────────────────────────────────────────────────────────────────

/// A text-to-speech request, used with [`Provider::tts`](crate::provider::Provider::tts).
///
/// Currently implemented by [`OpenAIProvider`](crate::providers::OpenAIProvider),
/// which calls the `/v1/audio/speech` endpoint.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::OpenAIProvider, Provider};
/// use baochuan::types::TtsRequestBuilder;
///
/// #[tokio::main]
/// async fn main() {
///     let provider = OpenAIProvider::new(std::env::var("OPENAI_API_KEY").unwrap());
///
///     let req = TtsRequestBuilder::new("tts-1", "Hello from baochuan!")
///         .voice("nova")
///         .format("mp3")
///         .build()
///         .unwrap();
///
///     let audio_bytes = provider.tts(&req).await.unwrap();
///     std::fs::write("output.mp3", audio_bytes).unwrap();
/// }
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct TtsRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    #[serde(rename = "response_format", skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
}

/// Builder for [`TtsRequest`].
#[derive(Debug, Default)]
pub struct TtsRequestBuilder {
    model: Option<String>,
    input: Option<String>,
    voice: Option<String>,
    format: Option<String>,
    speed: Option<f32>,
}

impl TtsRequestBuilder {
    /// Create a new builder with the required model and input text.
    pub fn new(model: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            model: Some(model.into()),
            input: Some(input.into()),
            ..Default::default()
        }
    }

    /// Set the voice (required before [`build`](Self::build)).
    ///
    /// OpenAI voices: `"alloy"`, `"echo"`, `"fable"`, `"onyx"`, `"nova"`, `"shimmer"`.
    pub fn voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }

    /// Set the output audio format.
    ///
    /// OpenAI formats: `"mp3"`, `"opus"`, `"aac"`, `"flac"`, `"wav"`, `"pcm"`.
    pub fn format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    /// Set the speaking speed multiplier (0.25–4.0, default 1.0).
    pub fn speed(mut self, speed: f32) -> Self {
        self.speed = Some(speed);
        self
    }

    pub fn build(self) -> Result<TtsRequest, BaochuanError> {
        Ok(TtsRequest {
            model: self
                .model
                .ok_or_else(|| BaochuanError::InvalidRequest("model required".to_string()))?,
            input: self
                .input
                .ok_or_else(|| BaochuanError::InvalidRequest("input required".to_string()))?,
            voice: self
                .voice
                .ok_or_else(|| BaochuanError::InvalidRequest("voice required".to_string()))?,
            format: self.format,
            speed: self.speed,
        })
    }
}
