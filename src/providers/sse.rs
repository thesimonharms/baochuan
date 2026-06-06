use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use tracing::error;

use crate::error::BaochuanError;
use crate::types::response::{AnthropicStreamDelta, AnthropicStreamEvent};
use crate::types::{Delta, StreamChoice, StreamChunk};

/// Parse a raw byte stream of OpenAI-compatible SSE `data:` lines into [`StreamChunk`]s.
///
/// Handles multi-byte network chunks, skips empty lines, stops at `[DONE]`.
pub fn sse_to_chunks<S>(stream: S) -> impl Stream<Item = Result<StreamChunk, BaochuanError>> + Send
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let mut buffer = String::new();

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

                    if data == "[DONE]" {
                        break;
                    }

                    match serde_json::from_str::<StreamChunk>(data) {
                        Ok(chunk) => chunks.push(Ok(chunk)),
                        Err(e) => {
                            error!(data = %data, error = %e, "failed to parse SSE chunk");
                            chunks.push(Err(BaochuanError::Stream(format!(
                                "failed to parse chunk: {e}"
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

/// Parse Anthropic's named-event SSE stream into [`StreamChunk`]s.
///
/// Anthropic uses `event:` + `data:` pairs with typed events. Only
/// `content_block_delta` events with `text_delta` payloads produce chunks.
pub fn anthropic_sse_to_chunks<S>(
    stream: S,
) -> impl Stream<Item = Result<StreamChunk, BaochuanError>> + Send
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let mut buffer = String::new();
    let mut current_event = String::new();
    let mut message_id = String::new();
    let mut model = String::new();

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
                        current_event.clear();
                        continue;
                    }

                    if let Some(event_type) = line.strip_prefix("event:") {
                        current_event = event_type.trim().to_string();
                        continue;
                    }

                    let data = match line.strip_prefix("data:") {
                        Some(rest) => rest.trim(),
                        None => continue,
                    };

                    match current_event.as_str() {
                        "message_start" => {
                            if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data) {
                                if let Some(msg) = event.message {
                                    message_id = msg.id.unwrap_or_default();
                                    model = msg.model.unwrap_or_default();
                                }
                            }
                        }
                        "content_block_delta" => {
                            if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data) {
                                if let Some(AnthropicStreamDelta {
                                    delta_type: Some(ref t),
                                    text: Some(ref text),
                                }) = event.delta
                                {
                                    if t == "text_delta" {
                                        chunks.push(Ok(StreamChunk {
                                            id: message_id.clone(),
                                            model: model.clone(),
                                            choices: vec![StreamChoice {
                                                index: 0,
                                                delta: Delta {
                                                    role: None,
                                                    content: Some(text.clone()),
                                                    tool_calls: None,
                                                },
                                                finish_reason: None,
                                            }],
                                        }));
                                    }
                                }
                            }
                        }
                        "message_stop" => {
                            chunks.push(Ok(StreamChunk {
                                id: message_id.clone(),
                                model: model.clone(),
                                choices: vec![StreamChoice {
                                    index: 0,
                                    delta: Delta {
                                        role: None,
                                        content: None,
                                        tool_calls: None,
                                    },
                                    finish_reason: Some("stop".to_string()),
                                }],
                            }));
                        }
                        _ => {}
                    }
                }

                chunks
            }
        };

        futures_util::stream::iter(items)
    })
}

// ── Cloudflare Workers AI streaming ──────────────────────────────────────────
//
// CF native streaming chunks are `data: {"response":"text","p":"..."}`.
// Unlike OpenAI-compat SSE the payload is a flat object, not a choices array.

#[derive(serde::Deserialize)]
struct CfStreamChunkData {
    response: Option<String>,
}

/// Parse Cloudflare Workers AI native SSE chunks into [`StreamChunk`]s.
pub fn cf_sse_to_chunks<S>(
    stream: S,
    model: String,
) -> impl Stream<Item = Result<StreamChunk, BaochuanError>> + Send
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let mut buffer = String::new();
    let mut index: u64 = 0;

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

                    if data == "[DONE]" {
                        break;
                    }

                    match serde_json::from_str::<CfStreamChunkData>(data) {
                        Ok(cf) => {
                            index += 1;
                            chunks.push(Ok(StreamChunk {
                                id: format!("cf-chunk-{index}"),
                                model: model.clone(),
                                choices: vec![StreamChoice {
                                    index: 0,
                                    delta: Delta {
                                        role: None,
                                        content: cf.response,
                                        tool_calls: None,
                                    },
                                    finish_reason: None,
                                }],
                            }));
                        }
                        Err(e) => {
                            error!(data = %data, error = %e, "failed to parse CF SSE chunk");
                            chunks.push(Err(BaochuanError::Stream(format!(
                                "failed to parse CF chunk: {e}"
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

// ── DashScope (Alibaba Qwen) streaming ────────────────────────────────────────
//
// DashScope SSE events are triplets:
//   id:N
//   event:result
//   :HTTP_STATUS/200        ← SSE comment, skip
//   data:{json}
//
// Each `data:` payload is a full DashScope response. With `incremental_output:true`
// the content field carries only the new tokens. `finish_reason` is the string
// "null" (not JSON null) while generating, "stop" on the final chunk.

#[derive(serde::Deserialize)]
pub(crate) struct DashScopeStreamPayload {
    pub output: DashScopeStreamOutput,
    pub request_id: Option<String>,
}

#[derive(serde::Deserialize)]
pub(crate) struct DashScopeStreamOutput {
    pub choices: Vec<DashScopeStreamChoice>,
}

#[derive(serde::Deserialize)]
pub(crate) struct DashScopeStreamChoice {
    pub message: DashScopeStreamMessage,
    pub finish_reason: Option<String>,
}

#[derive(serde::Deserialize)]
pub(crate) struct DashScopeStreamMessage {
    pub content: String,
}

/// Parse DashScope SSE chunks into [`StreamChunk`]s.
pub fn dashscope_sse_to_chunks<S>(
    stream: S,
    model: String,
) -> impl Stream<Item = Result<StreamChunk, BaochuanError>> + Send
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let mut buffer = String::new();

    stream.flat_map(move |result| {
        let items: Vec<Result<StreamChunk, BaochuanError>> = match result {
            Err(e) => vec![Err(BaochuanError::Http(e))],
            Ok(bytes) => {
                buffer.push_str(&String::from_utf8_lossy(&bytes));
                let mut chunks = Vec::new();

                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].trim().to_string();
                    buffer.drain(..=newline_pos);

                    if line.is_empty()
                        || line.starts_with("id:")
                        || line.starts_with("event:")
                        || line.starts_with(':')
                    {
                        continue;
                    }

                    let data = match line.strip_prefix("data:") {
                        Some(rest) => rest.trim(),
                        None => continue,
                    };

                    match serde_json::from_str::<DashScopeStreamPayload>(data) {
                        Ok(payload) => {
                            if let Some(choice) = payload.output.choices.into_iter().next() {
                                let done = choice
                                    .finish_reason
                                    .as_deref()
                                    .map(|r| r != "null")
                                    .unwrap_or(false);
                                let finish_reason = if done {
                                    choice.finish_reason.filter(|r| r != "null")
                                } else {
                                    None
                                };
                                let content = choice.message.content;
                                chunks.push(Ok(StreamChunk {
                                    id: payload.request_id.clone().unwrap_or_default(),
                                    model: model.clone(),
                                    choices: vec![StreamChoice {
                                        index: 0,
                                        delta: Delta {
                                            role: None,
                                            content: if content.is_empty() {
                                                None
                                            } else {
                                                Some(content)
                                            },
                                            tool_calls: None,
                                        },
                                        finish_reason,
                                    }],
                                }));
                            }
                        }
                        Err(e) => {
                            error!(data = %data, error = %e, "failed to parse DashScope SSE chunk");
                            chunks.push(Err(BaochuanError::Stream(format!(
                                "failed to parse DashScope chunk: {e}"
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    fn make_stream(data: &'static str) -> impl Stream<Item = Result<Bytes, reqwest::Error>> {
        futures_util::stream::iter(vec![Ok(Bytes::from(data))])
    }

    #[tokio::test]
    async fn test_sse_basic_parsing() {
        let data = concat!(
            "data: {\"id\":\"t1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null}]}\n",
            "\n",
            "data: {\"id\":\"t1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" World\"},\"finish_reason\":null}]}\n",
            "\n",
            "data: [DONE]\n",
        );

        let chunks: Vec<_> = sse_to_chunks(make_stream(data)).collect().await;
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].as_ref().unwrap().delta_content(), Some("Hello"));
        assert_eq!(chunks[1].as_ref().unwrap().delta_content(), Some(" World"));
    }

    #[tokio::test]
    async fn test_sse_stops_at_done() {
        let data = concat!(
            "data: {\"id\":\"t1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"A\"},\"finish_reason\":null}]}\n",
            "data: [DONE]\n",
            "data: {\"id\":\"t1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"B\"},\"finish_reason\":null}]}\n",
        );

        let chunks: Vec<_> = sse_to_chunks(make_stream(data)).collect().await;
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].as_ref().unwrap().delta_content(), Some("A"));
    }

    #[tokio::test]
    async fn test_anthropic_sse_parsing() {
        let data = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01\",\"model\":\"claude-3-5-sonnet-20241022\"}}\n",
            "\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n",
            "\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" there\"}}\n",
            "\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n",
            "\n",
        );

        let chunks: Vec<_> = anthropic_sse_to_chunks(make_stream(data)).collect().await;
        // 2 content deltas + 1 message_stop
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].as_ref().unwrap().delta_content(), Some("Hi"));
        assert_eq!(chunks[1].as_ref().unwrap().delta_content(), Some(" there"));
        assert!(chunks[2].as_ref().unwrap().is_finished());
    }
}
