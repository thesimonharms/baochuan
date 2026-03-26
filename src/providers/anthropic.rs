use async_trait::async_trait;
use reqwest::{Client, header};
use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::helpers::parse_data_url;
use crate::providers::sse::anthropic_sse_to_chunks;
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatChoice, ContentPart, DocumentInput, FunctionCall,
    MessageContent, ModelInfo, Role, ToolCall, ToolChoice, Usage,
};

const BASE_URL: &str = "https://api.anthropic.com/v1";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;

// ── Model list wire types ─────────────────────────────────────────────────────

#[derive(Deserialize)]
struct AnthropicModelList {
    data: Vec<AnthropicModelEntry>,
}

#[derive(Deserialize)]
struct AnthropicModelEntry {
    id: String,
    display_name: Option<String>,
}

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice>,
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    input_schema: serde_json::Value,
}

#[derive(Serialize)]
struct AnthropicToolChoice {
    #[serde(rename = "type")]
    choice_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

// Anthropic messages can have either plain-text content or an array of typed
// content blocks. We use an untagged enum so serde picks the right variant.
#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicMessageContent,
}

#[derive(Serialize)]
#[serde(untagged)]
enum AnthropicMessageContent {
    Text(String),
    Parts(Vec<AnthropicContentBlock>),
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentBlock {
    Text { text: String },
    Image { source: AnthropicImageSource },
    Document { source: AnthropicDocumentSource },
    ToolUse { id: String, name: String, input: serde_json::Value },
    ToolResult { tool_use_id: String, content: String },
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

#[derive(Serialize)]
#[serde(tag = "type")]
enum AnthropicDocumentSource {
    #[serde(rename = "base64")]
    Base64 { media_type: String, data: String },
}

#[derive(Deserialize)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<AnthropicContent>,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: Option<String>,
    /// Present on `tool_use` content blocks.
    id: Option<String>,
    name: Option<String>,
    input: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// ── Conversion helpers ────────────────────────────────────────────────────────

fn content_parts_to_anthropic(parts: &[ContentPart]) -> Vec<AnthropicContentBlock> {
    parts.iter().filter_map(|p| match p {
        ContentPart::Text { text } => {
            Some(AnthropicContentBlock::Text { text: text.clone() })
        }
        ContentPart::ImageUrl { image_url } => {
            if let Some((media_type, data)) = parse_data_url(&image_url.url) {
                Some(AnthropicContentBlock::Image {
                    source: AnthropicImageSource::Base64 { media_type, data },
                })
            } else {
                Some(AnthropicContentBlock::Image {
                    source: AnthropicImageSource::Url { url: image_url.url.clone() },
                })
            }
        }
        ContentPart::Document { document: DocumentInput { data, media_type } } => {
            Some(AnthropicContentBlock::Document {
                source: AnthropicDocumentSource::Base64 {
                    media_type: media_type.clone(),
                    data: data.clone(),
                },
            })
        }
        // Anthropic does not support audio input — drop the part.
        ContentPart::InputAudio { .. } => None,
    }).collect()
}

fn to_anthropic_message(m: &ChatMessage) -> AnthropicMessage {
    // Tool result messages: Role::Tool → user role with tool_result block
    if m.role == Role::Tool {
        let tool_use_id = m.tool_call_id.clone().unwrap_or_default();
        return AnthropicMessage {
            role: "user".to_string(),
            content: AnthropicMessageContent::Parts(vec![
                AnthropicContentBlock::ToolResult {
                    tool_use_id,
                    content: m.content.to_text_lossy(),
                }
            ]),
        };
    }

    // Assistant messages with tool calls: emit text block(s) + tool_use blocks
    if let Some(tool_calls) = &m.tool_calls {
        let mut blocks: Vec<AnthropicContentBlock> = match &m.content {
            MessageContent::Text(s) if !s.is_empty() => {
                vec![AnthropicContentBlock::Text { text: s.clone() }]
            }
            MessageContent::Parts(parts) => content_parts_to_anthropic(parts),
            _ => vec![],
        };
        for tc in tool_calls {
            let input: serde_json::Value =
                serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::Value::Null);
            blocks.push(AnthropicContentBlock::ToolUse {
                id: tc.id.clone(),
                name: tc.function.name.clone(),
                input,
            });
        }
        return AnthropicMessage {
            role: "assistant".to_string(),
            content: AnthropicMessageContent::Parts(blocks),
        };
    }

    // Standard message
    let content = match &m.content {
        MessageContent::Text(s) => AnthropicMessageContent::Text(s.clone()),
        MessageContent::Parts(parts) => {
            AnthropicMessageContent::Parts(content_parts_to_anthropic(parts))
        }
    };

    AnthropicMessage {
        role: match m.role {
            Role::User => "user".to_string(),
            Role::Assistant => "assistant".to_string(),
            _ => "user".to_string(),
        },
        content,
    }
}

fn to_anthropic_tool_choice(tc: &ToolChoice) -> AnthropicToolChoice {
    use crate::types::tools::{ToolChoicePreset};
    match tc {
        ToolChoice::Preset(ToolChoicePreset::Auto) => {
            AnthropicToolChoice { choice_type: "auto".to_string(), name: None }
        }
        ToolChoice::Preset(ToolChoicePreset::Required) => {
            // Anthropic uses "any" to mean "must use at least one tool"
            AnthropicToolChoice { choice_type: "any".to_string(), name: None }
        }
        ToolChoice::Preset(ToolChoicePreset::None) => {
            AnthropicToolChoice { choice_type: "none".to_string(), name: None }
        }
        ToolChoice::Function(f) => {
            AnthropicToolChoice { choice_type: "tool".to_string(), name: Some(f.function.name.clone()) }
        }
    }
}

fn to_anthropic_request(request: &ChatRequest, stream: bool) -> AnthropicRequest<'_> {
    // Extract system messages into the top-level `system` field
    let system: Option<String> = {
        let parts: Vec<String> = request
            .messages
            .iter()
            .filter(|m| m.role == Role::System)
            .map(|m| m.content.to_text_lossy())
            .collect();
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n"))
        }
    };

    let messages = request
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(to_anthropic_message)
        .collect();

    let tools = request.tools.as_ref().map(|tools| {
        tools.iter().map(|t| AnthropicTool {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            input_schema: t.function.parameters.clone(),
        }).collect()
    });

    let tool_choice = request.tool_choice.as_ref().map(to_anthropic_tool_choice);

    AnthropicRequest {
        model: &request.model,
        max_tokens: request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        messages,
        system,
        stream,
        temperature: request.temperature,
        tools,
        tool_choice,
    }
}

fn from_anthropic_response(resp: AnthropicResponse) -> ChatResponse {
    let mut text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for block in resp.content {
        match block.content_type.as_str() {
            "text" => {
                if let Some(t) = block.text {
                    text.push_str(&t);
                }
            }
            "tool_use" => {
                if let (Some(id), Some(name)) = (block.id, block.name) {
                    let arguments = block.input
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "{}".to_string());
                    tool_calls.push(ToolCall {
                        id,
                        call_type: "function".to_string(),
                        function: FunctionCall { name, arguments },
                    });
                }
            }
            _ => {}
        }
    }

    let mut message = ChatMessage::assistant(text);
    if !tool_calls.is_empty() {
        message.tool_calls = Some(tool_calls);
    }

    ChatResponse {
        id: resp.id,
        model: resp.model,
        choices: vec![ChatChoice {
            index: 0,
            message,
            finish_reason: resp.stop_reason,
        }],
        usage: Some(Usage {
            prompt_tokens: resp.usage.input_tokens,
            completion_tokens: resp.usage.output_tokens,
            total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
        }),
        citations: None,
    }
}

// ── Provider ──────────────────────────────────────────────────────────────────

/// A provider that connects to the [Anthropic](https://www.anthropic.com/) API (Claude models).
///
/// The Anthropic API uses a different request/response format from the
/// OpenAI-compatible providers. baochuan handles the conversion automatically —
/// system prompts, message roles, and usage statistics are all mapped to the
/// common types.
///
/// `max_tokens` is required by Anthropic; if you don't set it on the request,
/// baochuan defaults to 4096.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::AnthropicProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = AnthropicProvider::new(std::env::var("ANTHROPIC_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("claude-opus-4-6")
///         .message(ChatMessage::system("You are a helpful assistant."))
///         .message(ChatMessage::user("What is the capital of France?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider.
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::AnthropicProvider::new(
    ///     std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set"),
    /// );
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: BASE_URL.to_string(),
        }
    }

    /// Override the base URL (useful for proxies).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    fn messages_url(&self) -> String {
        format!("{}/messages", self.base_url)
    }

    fn base_request(&self, body: &impl Serialize) -> reqwest::RequestBuilder {
        self.client
            .post(self.messages_url())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header(header::CONTENT_TYPE, "application/json")
            .json(body)
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, BaochuanError> {
        let url = format!("{}/models", self.base_url);
        let response = self
            .client
            .get(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(BaochuanError::Api { status: status.as_u16(), message: body });
        }

        let list: AnthropicModelList = response.json().await?;
        Ok(list.data.into_iter().map(|m| ModelInfo {
            id: m.id,
            owned_by: Some("anthropic".to_string()),
            context_length: None,
            display_name: m.display_name,
        }).collect())
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BaochuanError> {
        debug!(model = %request.model, "sending chat request to Anthropic");

        let body = to_anthropic_request(request, false);
        let response = self.base_request(&body).send().await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            error!(status = %status, body = %text, "Anthropic API error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: text,
            });
        }

        let anthropic_response: AnthropicResponse = response.json().await?;
        debug!(id = %anthropic_response.id, "received Anthropic response");
        Ok(from_anthropic_response(anthropic_response))
    }

    async fn stream_chat(&self, request: &ChatRequest) -> Result<ChunkStream, BaochuanError> {
        debug!(model = %request.model, "starting streaming chat request to Anthropic");

        let body = to_anthropic_request(request, true);
        let response = self.base_request(&body).send().await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            error!(status = %status, body = %text, "Anthropic stream error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: text,
            });
        }

        Ok(Box::pin(anthropic_sse_to_chunks(response.bytes_stream())))
    }
}
