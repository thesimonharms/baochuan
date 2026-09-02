use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::{debug, error};

use crate::error::BaochuanError;
use crate::provider::{ChunkStream, Provider};
use crate::providers::openai_compat::OpenAICompatClient;
use crate::providers::sse::responses_sse_to_chunks;
use crate::types::{
    ChatChoice, ChatMessage, ChatRequest, ChatResponse, MessageContent, ModelInfo, Role, Usage,
};

const DEFAULT_BASE_URL: &str = "https://api.perplexity.ai/v1";

/// Sonar Chat Completions retired in favour of the Agent API (`POST /v1/agent`).
/// Legacy model slugs map onto Agent presets so existing call sites keep working.
fn map_sonar_model(model: &str) -> (&'static str, Option<&str>) {
    match model {
        "sonar" => ("preset", Some("fast")),
        "sonar-pro" => ("preset", Some("low")),
        "sonar-reasoning" | "sonar-reasoning-pro" => ("preset", Some("medium")),
        "sonar-deep-research" => ("preset", Some("high")),
        "fast" | "low" | "medium" | "high" | "xhigh" | "wide-research" => ("preset", Some(model)),
        _ => ("model", None),
    }
}

fn to_agent_body(request: &ChatRequest, stream: bool) -> Value {
    let mut instructions = Vec::new();
    let mut input = Vec::new();

    for message in &request.messages {
        let text = message.content.to_text_lossy();
        match message.role {
            Role::System => {
                if !text.is_empty() {
                    instructions.push(text);
                }
            }
            Role::User | Role::Assistant | Role::Tool => {
                input.push(json!({
                    "type": "message",
                    "role": message.role.to_string(),
                    "content": text,
                }));
            }
        }
    }

    let mut body = json!({
        "input": input,
        "tools": [{ "type": "web_search" }],
    });

    let (kind, preset) = map_sonar_model(&request.model);
    if let Some(preset) = preset {
        body["preset"] = json!(preset);
    } else if kind == "model" {
        body["model"] = json!(&request.model);
    }

    if !instructions.is_empty() {
        body["instructions"] = json!(instructions.join("\n"));
    }

    let max_output = request.max_completion_tokens.or(request.max_tokens);
    if let Some(max) = max_output {
        body["max_output_tokens"] = json!(max);
    }
    if let Some(temp) = request.temperature {
        body["temperature"] = json!(temp);
    }
    if stream {
        body["stream"] = json!(true);
    }

    body
}

#[derive(Deserialize)]
struct PerplexityAgentResponse {
    id: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    output: Vec<Value>,
    #[serde(default)]
    usage: Option<PerplexityUsage>,
}

#[derive(Deserialize)]
struct PerplexityUsage {
    #[serde(default)]
    input_tokens: Option<u32>,
    #[serde(default)]
    output_tokens: Option<u32>,
    #[serde(default)]
    total_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct PerplexityModelList {
    data: Vec<PerplexityModel>,
}

#[derive(Deserialize)]
struct PerplexityModel {
    id: String,
    owned_by: Option<String>,
    context_length: Option<u32>,
}

fn extract_output_text(output: &[Value]) -> String {
    let mut text = String::new();
    for item in output {
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
        if item_type != "message" {
            continue;
        }
        let Some(parts) = item.get("content").and_then(Value::as_array) else {
            continue;
        };
        for part in parts {
            if part.get("type").and_then(Value::as_str) == Some("output_text")
                && let Some(chunk) = part.get("text").and_then(Value::as_str)
            {
                text.push_str(chunk);
            }
        }
    }
    text
}

fn extract_citations(output: &[Value]) -> Option<Vec<String>> {
    let mut citations = Vec::new();
    for item in output {
        if item.get("type").and_then(Value::as_str) != Some("search_results") {
            continue;
        }
        let Some(results) = item.get("results").and_then(Value::as_array) else {
            continue;
        };
        for result in results {
            if let Some(url) = result.get("url").and_then(Value::as_str)
                && !url.is_empty()
                && !citations.iter().any(|existing| existing == url)
            {
                citations.push(url.to_string());
            }
        }
    }
    if citations.is_empty() {
        None
    } else {
        Some(citations)
    }
}

fn from_agent_response(resp: PerplexityAgentResponse) -> ChatResponse {
    let text = extract_output_text(&resp.output);
    let citations = extract_citations(&resp.output);
    let usage = resp.usage.map(|u| {
        let prompt = u.input_tokens.unwrap_or(0);
        let completion = u.output_tokens.unwrap_or(0);
        Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: u.total_tokens.unwrap_or(prompt + completion),
        }
    });

    ChatResponse {
        id: resp.id,
        model: resp.model.unwrap_or_default(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: Role::Assistant,
                content: MessageContent::Text(text),
                reasoning_content: None,
                audio: None,
                tool_calls: None,
                tool_call_id: None,
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage,
        citations,
    }
}

/// A provider that connects to the [Perplexity](https://docs.perplexity.ai/) Agent API.
///
/// Perplexity retired Sonar Chat Completions (`POST /chat/completions`) on
/// 27 September 2026 in favour of `POST /v1/agent` (Responses-compatible).
/// This provider maps the unified [`ChatRequest`] onto that Agent endpoint:
///
/// - Legacy slugs `sonar` / `sonar-pro` / `sonar-reasoning-pro` /
///   `sonar-deep-research` become Agent presets `fast` / `low` / `medium` / `high`.
/// - System messages become `instructions`; other turns become `input` items.
/// - `max_tokens` is sent as `max_output_tokens`.
/// - `web_search` is enabled so responses stay grounded.
/// - Source URLs are read from `search_results` output items and surfaced as
///   [`ChatResponse::citations`].
///
/// You can also pass a preset name (`fast`, `low`, `medium`, `high`) or a
/// provider-qualified model such as `perplexity/sonar` directly.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::PerplexityProvider, ChatMessage, ChatRequestBuilder, Provider};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = PerplexityProvider::new(std::env::var("PERPLEXITY_API_KEY").unwrap());
///
///     let request = ChatRequestBuilder::new("sonar-pro")
///         .message(ChatMessage::user("What happened in the news today?"))
///         .build()
///         .unwrap();
///
///     let response = provider.chat(&request).await.unwrap();
///     println!("{}", response.content().unwrap_or(""));
///
///     if let Some(citations) = &response.citations {
///         for url in citations {
///             println!("  source: {url}");
///         }
///     }
/// }
/// ```
pub struct PerplexityProvider {
    inner: OpenAICompatClient,
}

impl PerplexityProvider {
    /// Create a new Perplexity provider.
    ///
    /// ```rust,no_run
    /// let provider = baochuan::providers::PerplexityProvider::new(
    ///     std::env::var("PERPLEXITY_API_KEY").expect("PERPLEXITY_API_KEY not set"),
    /// );
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            inner: OpenAICompatClient::with_key(api_key, DEFAULT_BASE_URL),
        }
    }

    /// Override the base URL. Pass the `/v1` root (for example
    /// `https://api.perplexity.ai/v1`); chat is sent to `{base}/agent`.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner.base_url = base_url.into();
        self
    }

    fn agent_url(&self) -> String {
        format!("{}/agent", self.inner.base_url.trim_end_matches('/'))
    }
}

#[async_trait]
impl Provider for PerplexityProvider {
    fn name(&self) -> &str {
        "perplexity"
    }

    async fn models(&self) -> Result<Vec<ModelInfo>, BaochuanError> {
        debug!("listing models from Perplexity");

        let response = self
            .inner
            .auth(self.inner.client.get(self.inner.models_url()))
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "Perplexity models error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let list: PerplexityModelList = response.json().await?;
        Ok(list
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id,
                owned_by: m.owned_by,
                context_length: m.context_length,
                display_name: None,
            })
            .collect())
    }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BaochuanError> {
        debug!(model = %request.model, "sending Agent API request to Perplexity");

        let body = to_agent_body(request, false);
        let response = self
            .inner
            .auth(self.inner.client.post(self.agent_url()))
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "Perplexity Agent API error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let agent: PerplexityAgentResponse = response.json().await?;
        debug!(id = %agent.id, "received Perplexity Agent response");
        Ok(from_agent_response(agent))
    }

    async fn stream_chat(&self, request: &ChatRequest) -> Result<ChunkStream, BaochuanError> {
        debug!(model = %request.model, "starting Perplexity Agent stream");

        let body = to_agent_body(request, true);
        let response = self
            .inner
            .auth(self.inner.client.post(self.agent_url()))
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "Perplexity Agent stream error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        Ok(Box::pin(responses_sse_to_chunks(
            response.bytes_stream(),
            request.model.clone(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChatRequestBuilder;

    #[test]
    fn maps_sonar_slugs_to_presets() {
        let req = ChatRequestBuilder::new("sonar-pro")
            .message(ChatMessage::system("Be concise."))
            .message(ChatMessage::user("News?"))
            .max_tokens(256)
            .build()
            .unwrap();

        let body = to_agent_body(&req, false);
        assert_eq!(body["preset"], "low");
        assert!(body.get("model").is_none());
        assert_eq!(body["max_output_tokens"], 256);
        assert_eq!(body["instructions"], "Be concise.");
        assert_eq!(body["tools"][0]["type"], "web_search");
        assert_eq!(body["input"][0]["role"], "user");
        assert_eq!(body["input"][0]["content"], "News?");
    }

    #[test]
    fn passes_through_qualified_models() {
        let req = ChatRequestBuilder::new("perplexity/sonar")
            .message(ChatMessage::user("Hello"))
            .build()
            .unwrap();
        let body = to_agent_body(&req, true);
        assert_eq!(body["model"], "perplexity/sonar");
        assert!(body.get("preset").is_none());
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn extracts_text_and_citations_from_agent_output() {
        let resp = PerplexityAgentResponse {
            id: "resp_1".into(),
            model: Some("openai/gpt-5.6-sol".into()),
            output: vec![
                json!({
                    "type": "search_results",
                    "results": [
                        {"id": 1, "url": "https://example.com/a"},
                        {"id": 2, "url": "https://example.com/b"}
                    ]
                }),
                json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Paris is the capital."}
                    ]
                }),
            ],
            usage: Some(PerplexityUsage {
                input_tokens: Some(10),
                output_tokens: Some(8),
                total_tokens: Some(18),
            }),
        };

        let chat = from_agent_response(resp);
        assert_eq!(chat.content(), Some("Paris is the capital."));
        assert_eq!(
            chat.citations,
            Some(vec![
                "https://example.com/a".to_string(),
                "https://example.com/b".to_string()
            ])
        );
        assert_eq!(chat.usage.unwrap().total_tokens, 18);
    }
}
