use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::helpers::parse_data_url;
use crate::types::{
    ChatChoice, ChatMessage, ChatRequest, ChatResponse, ContentPart, Delta, MessageContent,
    ModelInfo, Role, StreamChoice, StreamChunk, Usage,
};

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

// ── Native wire types ─────────────────────────────────────────────────────────

#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
}

#[derive(Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
}

#[derive(Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    model: String,
    message: OllamaResponseMessage,
    done: bool,
    prompt_eval_count: Option<u32>,
    eval_count: Option<u32>,
}

#[derive(Deserialize)]
struct OllamaStreamChunk {
    model: String,
    message: OllamaResponseMessage,
    done: bool,
}

#[derive(Deserialize)]
struct OllamaResponseMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OllamaModelList {
    models: Vec<OllamaModelEntry>,
}

#[derive(Deserialize)]
struct OllamaModelEntry {
    name: String,
    details: Option<OllamaModelDetails>,
}

#[derive(Deserialize)]
struct OllamaModelDetails {
    family: Option<String>,
    parameter_size: Option<String>,
    quantization_level: Option<String>,
}

// ── Conversion helpers ────────────────────────────────────────────────────────

fn to_ollama_messages(messages: &[ChatMessage]) -> Vec<OllamaMessage> {
    messages
        .iter()
        .map(|m| {
            // Ollama passes images as a separate base64 array, extracted from
            // data-URL image parts. HTTP-URL images are not supported natively.
            let images: Vec<String> = match &m.content {
                MessageContent::Parts(parts) => parts
                    .iter()
                    .filter_map(|p| {
                        if let ContentPart::ImageUrl { image_url } = p {
                            parse_data_url(&image_url.url).map(|(_mime, data)| data)
                        } else {
                            None
                        }
                    })
                    .collect(),
                _ => vec![],
            };

            OllamaMessage {
                role: match m.role {
                    Role::System => "system".to_string(),
                    Role::User => "user".to_string(),
                    Role::Assistant => "assistant".to_string(),
                    Role::Tool => "tool".to_string(),
                },
                content: m.content.to_text_lossy(),
                images: if images.is_empty() {
                    None
                } else {
                    Some(images)
                },
            }
        })
        .collect()
}

fn from_ollama_response(resp: OllamaChatResponse) -> ChatResponse {
    let prompt_tokens = resp.prompt_eval_count.unwrap_or(0);
    let completion_tokens = resp.eval_count.unwrap_or(0);

    ChatResponse {
        id: String::new(), // Ollama does not return a request ID
        model: resp.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: if resp.message.role == "assistant" {
                    Role::Assistant
                } else {
                    Role::User
                },
                content: MessageContent::Text(resp.message.content),
                reasoning_content: None,
                audio: None,
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: if resp.done {
                Some("stop".to_string())
            } else {
                None
            },
        }],
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }),
        citations: None,
    }
}

/// Parse Ollama's NDJSON streaming response into [`StreamChunk`]s.
///
/// Ollama streams newline-delimited JSON objects rather than SSE. Each line is
/// a complete JSON object; the final one has `"done": true`.
fn ollama_ndjson_to_chunks(
    stream: impl Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
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

                    match serde_json::from_str::<OllamaStreamChunk>(&line) {
                        Ok(chunk) => {
                            chunk_index += 1;
                            let finish_reason = if chunk.done {
                                Some("stop".to_string())
                            } else {
                                None
                            };
                            let content = if chunk.message.content.is_empty() {
                                None
                            } else {
                                Some(chunk.message.content)
                            };
                            chunks.push(Ok(StreamChunk {
                                id: format!("ollama-chunk-{chunk_index}"),
                                model: chunk.model,
                                choices: vec![StreamChoice {
                                    index: 0,
                                    delta: Delta {
                                        role: None,
                                        content,
                                        tool_calls: None,
                                    },
                                    finish_reason,
                                }],
                            }));
                        }
                        Err(e) => {
                            error!(line = %line, error = %e, "failed to parse Ollama NDJSON chunk");
                            chunks.push(Err(BaochuanError::Stream(format!(
                                "failed to parse Ollama chunk: {e}"
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

/// A provider that connects to a local [Ollama](https://ollama.com/) server
/// using Ollama's **native `/api/` API**.
///
/// Ollama's native API uses NDJSON for streaming (not SSE), and returns
/// richer metadata in model listings including family, parameter size, and
/// quantization. No API key is needed.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::OllamaProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = OllamaProvider::new();
///
///     // See what models you have pulled
///     let models = provider.models().await.unwrap();
///     for m in &models {
///         println!("{}", m.id);
///     }
///
///     let request = ChatRequestBuilder::new("llama3.2")
///         .message(ChatMessage::user("Why is the sky blue?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
/// }
/// ```
pub struct OllamaProvider {
    client: Client,
    base_url: String,
}

impl OllamaProvider {
    /// Create a provider pointing at the default Ollama address
    /// (`http://localhost:11434`).
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Override the server address.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

impl Default for OllamaProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, BaochuanError> {
        debug!("listing models from Ollama");

        let url = format!("{}/api/tags", self.base_url);
        let response = self.client.get(&url).send().await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "Ollama models error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let list: OllamaModelList = response.json().await?;
        Ok(list
            .models
            .into_iter()
            .map(|m| {
                let display = m.details.as_ref().and_then(|d| {
                    match (&d.parameter_size, &d.quantization_level) {
                        (Some(p), Some(q)) => Some(format!("{p} · {q}")),
                        (Some(p), None) => Some(p.clone()),
                        _ => None,
                    }
                });
                let owned_by = m.details.as_ref().and_then(|d| d.family.clone());
                ModelInfo {
                    id: m.name,
                    owned_by,
                    context_length: None, // not in /api/tags; use /api/show for details
                    display_name: display,
                }
            })
            .collect())
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BaochuanError> {
        debug!(model = %request.model, "sending chat request to Ollama");

        let options = if request.max_tokens.is_some()
            || request.temperature.is_some()
            || request.top_p.is_some()
        {
            Some(OllamaOptions {
                num_predict: request.max_tokens,
                temperature: request.temperature,
                top_p: request.top_p,
            })
        } else {
            None
        };

        let body = OllamaChatRequest {
            model: &request.model,
            messages: to_ollama_messages(&request.messages),
            stream: false,
            options,
        };

        let url = format!("{}/api/chat", self.base_url);
        let response = self.client.post(&url).json(&body).send().await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "Ollama API error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let ollama_response: OllamaChatResponse = response.json().await?;
        Ok(from_ollama_response(ollama_response))
    }

    async fn stream_chat(&self, request: &ChatRequest) -> Result<ChunkStream, BaochuanError> {
        debug!(model = %request.model, "starting streaming chat request to Ollama");

        let options = if request.max_tokens.is_some()
            || request.temperature.is_some()
            || request.top_p.is_some()
        {
            Some(OllamaOptions {
                num_predict: request.max_tokens,
                temperature: request.temperature,
                top_p: request.top_p,
            })
        } else {
            None
        };

        let body = OllamaChatRequest {
            model: &request.model,
            messages: to_ollama_messages(&request.messages),
            stream: true,
            options,
        };

        let url = format!("{}/api/chat", self.base_url);
        let response = self.client.post(&url).json(&body).send().await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "Ollama stream error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        Ok(Box::pin(ollama_ndjson_to_chunks(response.bytes_stream())))
    }
}
