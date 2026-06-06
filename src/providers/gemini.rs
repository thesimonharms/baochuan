use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::helpers::{guess_image_mime_type, parse_data_url};
use crate::types::{
    ChatChoice, ChatMessage, ChatRequest, ChatResponse, ContentPart, Delta, DocumentInput,
    FunctionCall, MessageContent, ModelInfo, Role, StreamChoice, StreamChunk, ToolCall, ToolChoice,
    Usage,
};

const BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

// ── Model list wire types ─────────────────────────────────────────────────────

#[derive(Deserialize)]
struct GeminiModelList {
    models: Vec<GeminiModelEntry>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiModelEntry {
    /// e.g. "models/gemini-1.5-flash"
    name: String,
    display_name: Option<String>,
    input_token_limit: Option<u32>,
}

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTools>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<GeminiToolConfig>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiTools {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: serde_json::Value,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiToolConfig {
    function_calling_config: GeminiFunctionCallingConfig,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiFunctionCallingConfig {
    mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_function_names: Option<Vec<String>>,
}

#[derive(Serialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

/// Gemini content part — text, inline base64 data, file URI, or function call/response.
#[derive(Serialize)]
#[serde(untagged)]
enum GeminiPart {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: GeminiInlineData,
    },
    FileData {
        #[serde(rename = "fileData")]
        file_data: GeminiFileData,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCallPart,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GeminiFunctionResponsePart,
    },
}

#[derive(Serialize, Deserialize)]
struct GeminiFunctionCallPart {
    name: String,
    args: serde_json::Value,
}

#[derive(Serialize)]
struct GeminiFunctionResponsePart {
    name: String,
    response: serde_json::Value,
}

#[derive(Serialize)]
struct GeminiInlineData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

#[derive(Serialize)]
struct GeminiFileData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    #[serde(rename = "fileUri")]
    file_uri: String,
}

#[derive(Serialize)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: GeminiResponseContent,
    finish_reason: Option<String>,
    index: Option<u32>,
}

#[derive(Deserialize)]
struct GeminiResponseContent {
    parts: Vec<GeminiResponsePart>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponsePart {
    text: Option<String>,
    function_call: Option<GeminiFunctionCallPart>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: Option<u32>,
    candidates_token_count: Option<u32>,
    total_token_count: Option<u32>,
}

// ── Conversion helpers ────────────────────────────────────────────────────────

fn content_to_gemini_parts(content: &MessageContent) -> Vec<GeminiPart> {
    match content {
        MessageContent::Text(s) => vec![GeminiPart::Text { text: s.clone() }],
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|p| match p {
                ContentPart::Text { text } => GeminiPart::Text { text: text.clone() },
                ContentPart::ImageUrl { image_url } => {
                    if let Some((mime_type, data)) = parse_data_url(&image_url.url) {
                        GeminiPart::InlineData {
                            inline_data: GeminiInlineData { mime_type, data },
                        }
                    } else {
                        GeminiPart::FileData {
                            file_data: GeminiFileData {
                                mime_type: guess_image_mime_type(&image_url.url).to_string(),
                                file_uri: image_url.url.clone(),
                            },
                        }
                    }
                }
                ContentPart::InputAudio { input_audio } => GeminiPart::InlineData {
                    inline_data: GeminiInlineData {
                        mime_type: input_audio.mime_type(),
                        data: input_audio.data.clone(),
                    },
                },
                ContentPart::Document {
                    document: DocumentInput { data, media_type },
                } => {
                    // PDFs and other documents use inlineData on Gemini
                    GeminiPart::InlineData {
                        inline_data: GeminiInlineData {
                            mime_type: media_type.clone(),
                            data: data.clone(),
                        },
                    }
                }
            })
            .collect(),
    }
}

fn to_gemini_content(m: &ChatMessage) -> GeminiContent {
    // Tool result messages (Role::Tool) → user role with functionResponse part
    if m.role == Role::Tool {
        let name = m.tool_call_id.clone().unwrap_or_default();
        let result_text = m.content.to_text_lossy();
        // Gemini expects response as a JSON object
        let response = serde_json::json!({ "result": result_text });
        return GeminiContent {
            role: "user".to_string(),
            parts: vec![GeminiPart::FunctionResponse {
                function_response: GeminiFunctionResponsePart { name, response },
            }],
        };
    }

    // Assistant messages with tool_calls → model role with functionCall parts
    if let Some(tool_calls) = &m.tool_calls {
        let mut parts: Vec<GeminiPart> = match &m.content {
            MessageContent::Text(s) if !s.is_empty() => {
                vec![GeminiPart::Text { text: s.clone() }]
            }
            MessageContent::Parts(_) => content_to_gemini_parts(&m.content),
            _ => vec![],
        };
        for tc in tool_calls {
            let args: serde_json::Value =
                serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::Value::Null);
            parts.push(GeminiPart::FunctionCall {
                function_call: GeminiFunctionCallPart {
                    name: tc.function.name.clone(),
                    args,
                },
            });
        }
        return GeminiContent {
            role: "model".to_string(),
            parts,
        };
    }

    GeminiContent {
        role: match m.role {
            Role::User => "user".to_string(),
            Role::Assistant => "model".to_string(),
            _ => "user".to_string(),
        },
        parts: content_to_gemini_parts(&m.content),
    }
}

fn to_gemini_tool_config(tc: &ToolChoice) -> GeminiToolConfig {
    use crate::types::tools::ToolChoicePreset;
    let (mode, names) = match tc {
        ToolChoice::Preset(ToolChoicePreset::Auto) => ("AUTO".to_string(), None),
        ToolChoice::Preset(ToolChoicePreset::Required) => ("ANY".to_string(), None),
        ToolChoice::Preset(ToolChoicePreset::None) => ("NONE".to_string(), None),
        ToolChoice::Function(f) => ("ANY".to_string(), Some(vec![f.function.name.clone()])),
    };
    GeminiToolConfig {
        function_calling_config: GeminiFunctionCallingConfig {
            mode,
            allowed_function_names: names,
        },
    }
}

fn to_gemini_request(request: &ChatRequest) -> GeminiRequest {
    let system_instruction: Option<GeminiSystemInstruction> = {
        let parts: Vec<GeminiPart> = request
            .messages
            .iter()
            .filter(|m| m.role == Role::System)
            .map(|m| GeminiPart::Text {
                text: m.content.to_text_lossy(),
            })
            .collect();
        if parts.is_empty() {
            None
        } else {
            Some(GeminiSystemInstruction { parts })
        }
    };

    let contents = request
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(to_gemini_content)
        .collect();

    let generation_config = if request.max_tokens.is_some() || request.temperature.is_some() {
        Some(GeminiGenerationConfig {
            max_output_tokens: request.max_tokens,
            temperature: request.temperature,
        })
    } else {
        None
    };

    let tools = request.tools.as_ref().map(|tools| {
        vec![GeminiTools {
            function_declarations: tools
                .iter()
                .map(|t| GeminiFunctionDeclaration {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    parameters: t.function.parameters.clone(),
                })
                .collect(),
        }]
    });

    let tool_config = request.tool_choice.as_ref().map(to_gemini_tool_config);

    GeminiRequest {
        contents,
        system_instruction,
        generation_config,
        tools,
        tool_config,
    }
}

fn from_gemini_response(resp: GeminiResponse, model: &str) -> ChatResponse {
    let choices = resp
        .candidates
        .into_iter()
        .map(|c| {
            let mut text = String::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();

            for part in c.content.parts {
                if let Some(t) = part.text {
                    text.push_str(&t);
                }
                if let Some(fc) = part.function_call {
                    // Gemini has no call ID — use function name as ID for round-tripping
                    tool_calls.push(ToolCall {
                        id: fc.name.clone(),
                        call_type: "function".to_string(),
                        function: FunctionCall {
                            name: fc.name,
                            arguments: fc.args.to_string(),
                        },
                    });
                }
            }

            let mut message = ChatMessage::assistant(text);
            if !tool_calls.is_empty() {
                message.tool_calls = Some(tool_calls);
            }

            ChatChoice {
                index: c.index.unwrap_or(0),
                message,
                finish_reason: c.finish_reason,
            }
        })
        .collect();

    let usage = resp.usage_metadata.map(|u| Usage {
        prompt_tokens: u.prompt_token_count.unwrap_or(0),
        completion_tokens: u.candidates_token_count.unwrap_or(0),
        total_tokens: u.total_token_count.unwrap_or(0),
    });

    ChatResponse {
        id: String::new(), // Gemini does not return a top-level request ID
        model: model.to_string(),
        choices,
        usage,
        citations: None,
    }
}

/// Parse a Gemini SSE stream where each `data:` line is a `GeminiResponse`.
fn gemini_sse_to_chunks(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
    model: String,
) -> impl Stream<Item = Result<StreamChunk, BaochuanError>> + Send {
    let mut buffer = String::new();
    let mut chunk_index: u64 = 0;

    stream.flat_map(move |result| {
        let items: Vec<Result<StreamChunk, BaochuanError>> = match result {
            Err(e) => vec![Err(BaochuanError::Http(e))],
            Ok(bytes) => {
                buffer.push_str(&String::from_utf8_lossy(&bytes));
                let mut chunks = Vec::new();

                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].trim().to_string();
                    buffer.drain(..=newline_pos);

                    if line.is_empty() {
                        continue;
                    }

                    let data = match line.strip_prefix("data:") {
                        Some(rest) => rest.trim(),
                        None => continue,
                    };

                    match serde_json::from_str::<GeminiResponse>(data) {
                        Ok(resp) => {
                            let text = resp
                                .candidates
                                .first()
                                .and_then(|c| c.content.parts.first())
                                .and_then(|p| p.text.clone())
                                .unwrap_or_default();

                            let finish_reason = resp
                                .candidates
                                .first()
                                .and_then(|c| c.finish_reason.clone())
                                .filter(|r| r != "UNSPECIFIED" && !r.is_empty());

                            chunk_index += 1;
                            chunks.push(Ok(StreamChunk {
                                id: format!("gemini-chunk-{chunk_index}"),
                                model: model.clone(),
                                choices: vec![StreamChoice {
                                    index: 0,
                                    delta: Delta {
                                        role: None,
                                        content: if text.is_empty() { None } else { Some(text) },
                                        tool_calls: None,
                                    },
                                    finish_reason,
                                }],
                            }));
                        }
                        Err(e) => {
                            error!(data = %data, error = %e, "failed to parse Gemini SSE chunk");
                            chunks.push(Err(BaochuanError::Stream(format!(
                                "failed to parse Gemini chunk: {e}"
                            ))));
                        }
                    }
                }

                chunks
            }
        };

        futures_util::stream::iter(items)
    })
}

// ── Provider ──────────────────────────────────────────────────────────────────

/// A provider that connects to the [Google Gemini](https://ai.google.dev/) API.
///
/// The Gemini API uses a different request/response format from OpenAI-compatible
/// providers. baochuan handles the conversion automatically. Authentication uses
/// an API key passed as a query parameter rather than a Bearer token.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::GeminiProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = GeminiProvider::new(std::env::var("GEMINI_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("gemini-1.5-flash")
///         .message(ChatMessage::user("What is the capital of France?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct GeminiProvider {
    client: Client,
    api_key: String,
    base_url: String,
}

impl GeminiProvider {
    /// Create a new Gemini provider.
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::GeminiProvider::new(
    ///     std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY not set"),
    /// );
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            base_url: BASE_URL.to_string(),
        }
    }

    /// Override the base URL (useful for Vertex AI or proxies).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    fn generate_url(&self, model: &str) -> String {
        format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url, model, self.api_key
        )
    }

    fn stream_url(&self, model: &str) -> String {
        format!(
            "{}/models/{}:streamGenerateContent?alt=sse&key={}",
            self.base_url, model, self.api_key
        )
    }
}

#[async_trait]
impl Provider for GeminiProvider {
    fn name(&self) -> &str {
        "gemini"
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, BaochuanError> {
        let url = format!("{}/models?key={}", self.base_url, self.api_key);
        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let list: GeminiModelList = response.json().await?;
        Ok(list
            .models
            .into_iter()
            .map(|m| ModelInfo {
                // Strip "models/" prefix → "gemini-1.5-flash"
                id: m
                    .name
                    .strip_prefix("models/")
                    .unwrap_or(&m.name)
                    .to_string(),
                owned_by: Some("google".to_string()),
                context_length: m.input_token_limit,
                display_name: m.display_name,
            })
            .collect())
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BaochuanError> {
        debug!(model = %request.model, "sending chat request to Gemini");

        let body = to_gemini_request(request);
        let response = self
            .client
            .post(self.generate_url(&request.model))
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            error!(status = %status, body = %text, "Gemini API error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: text,
            });
        }

        let gemini_response: GeminiResponse = response.json().await?;
        debug!(model = %request.model, "received Gemini response");
        Ok(from_gemini_response(gemini_response, &request.model))
    }

    async fn stream_chat(&self, request: &ChatRequest) -> Result<ChunkStream, BaochuanError> {
        debug!(model = %request.model, "starting streaming chat request to Gemini");

        let body = to_gemini_request(request);
        let response = self
            .client
            .post(self.stream_url(&request.model))
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            error!(status = %status, body = %text, "Gemini stream error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: text,
            });
        }

        let model = request.model.clone();
        Ok(Box::pin(gemini_sse_to_chunks(
            response.bytes_stream(),
            model,
        )))
    }
}
