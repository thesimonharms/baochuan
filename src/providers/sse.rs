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
                            if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data)
                                && let Some(msg) = event.message
                            {
                                message_id = msg.id.unwrap_or_default();
                                model = msg.model.unwrap_or_default();
                            }
                        }
                        "content_block_delta" => {
                            if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data)
                                && let Some(AnthropicStreamDelta {
                                    delta_type: Some(ref t),
                                    text: Some(ref text),
                                }) = event.delta
                                && t == "text_delta"
                            {
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

/// Parse OpenAI Responses / Perplexity Agent SSE into [`StreamChunk`]s.
///
/// These APIs emit named events (`event:`) plus a JSON `data:` payload.
/// Text arrives as `response.output_text.delta` with a `delta` string.
pub fn responses_sse_to_chunks<S>(
    stream: S,
    model: String,
) -> impl Stream<Item = Result<StreamChunk, BaochuanError>> + Send
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let mut buffer = String::new();
    let mut current_event = String::new();
    let mut index: u64 = 0;
    let mut response_id = String::new();

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

                    if data == "[DONE]" {
                        break;
                    }

                    let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
                        continue;
                    };

                    if let Some(id) = value.get("id").and_then(|v| v.as_str()) {
                        response_id = id.to_string();
                    } else if let Some(id) = value
                        .get("response")
                        .and_then(|r| r.get("id"))
                        .and_then(|v| v.as_str())
                    {
                        response_id = id.to_string();
                    }

                    let event_name = if current_event.is_empty() {
                        value
                            .get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string()
                    } else {
                        current_event.clone()
                    };

                    match event_name.as_str() {
                        "response.output_text.delta" => {
                            if let Some(text) = value.get("delta").and_then(|v| v.as_str())
                                && !text.is_empty()
                            {
                                index += 1;
                                chunks.push(Ok(StreamChunk {
                                    id: if response_id.is_empty() {
                                        format!("agent-chunk-{index}")
                                    } else {
                                        response_id.clone()
                                    },
                                    model: model.clone(),
                                    choices: vec![StreamChoice {
                                        index: 0,
                                        delta: Delta {
                                            role: None,
                                            content: Some(text.to_string()),
                                            tool_calls: None,
                                        },
                                        finish_reason: None,
                                    }],
                                }));
                            }
                        }
                        "response.completed" | "response.done" => {
                            chunks.push(Ok(StreamChunk {
                                id: response_id.clone(),
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

    #[tokio::test]
    async fn test_responses_sse_parsing() {
        let data = concat!(
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Par\"}\n",
            "\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"is.\"}\n",
            "\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"id\":\"resp_1\"}\n",
            "\n",
        );

        let chunks: Vec<_> = responses_sse_to_chunks(make_stream(data), "sonar".into())
            .collect()
            .await;
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].as_ref().unwrap().delta_content(), Some("Par"));
        assert_eq!(chunks[1].as_ref().unwrap().delta_content(), Some("is."));
        assert!(chunks[2].as_ref().unwrap().is_finished());
    }
}
