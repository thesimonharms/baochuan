use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::BaochuanError;

/// A boxed, send-safe stream of agent events.
pub type AgentEventStream = Pin<Box<dyn Stream<Item = Result<AgentEvent, BaochuanError>> + Send>>;

/// Higher-level abstraction over stateful agent runtimes.
///
/// [`AgentProvider`] is intentionally separate from [`Provider`](crate::Provider):
/// model providers answer chat completions, while agent providers can run
/// multi-step tasks, preserve server-side state, emit tool progress, and return
/// structured agent lifecycle events.
#[async_trait]
pub trait AgentProvider: Send + Sync {
    /// Run an agent request to completion and return the final response.
    async fn run(&self, request: &AgentRunRequest) -> Result<AgentResponse, BaochuanError>;

    /// Run an agent request as a server-sent event stream.
    async fn stream_run(
        &self,
        request: &AgentRunRequest,
    ) -> Result<AgentEventStream, BaochuanError>;

    /// A human-readable identifier for this agent provider.
    fn name(&self) -> &str;
}

/// Input accepted by the OpenAI Responses-compatible agent endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AgentInput {
    /// Plain text input for the next turn.
    Text(String),
    /// Structured Responses API input items.
    Items(Vec<AgentInputItem>),
}

impl AgentInput {
    fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.trim().is_empty(),
            Self::Items(items) => items.is_empty(),
        }
    }
}

impl From<String> for AgentInput {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for AgentInput {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

/// A structured Responses API input item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInputItem {
    pub role: String,
    pub content: AgentInputContent,
}

/// Content for a structured agent input item.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AgentInputContent {
    Text(String),
    Parts(Vec<AgentInputContentPart>),
}

/// One content part in a structured agent input item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInputContentPart {
    #[serde(rename = "type")]
    pub part_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

/// A request to a stateful agent runtime.
#[derive(Debug, Clone, Serialize)]
pub struct AgentRunRequest {
    pub model: String,
    pub input: AgentInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    pub stream: bool,
}

/// Builder for [`AgentRunRequest`].
#[derive(Debug, Default)]
pub struct AgentRunRequestBuilder {
    model: Option<String>,
    input: Option<AgentInput>,
    instructions: Option<String>,
    previous_response_id: Option<String>,
    conversation: Option<String>,
    store: Option<bool>,
    stream: bool,
}

impl AgentRunRequestBuilder {
    pub fn new(model: impl Into<String>, input: impl Into<AgentInput>) -> Self {
        Self {
            model: Some(model.into()),
            input: Some(input.into()),
            ..Default::default()
        }
    }

    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    pub fn previous_response_id(mut self, previous_response_id: impl Into<String>) -> Self {
        self.previous_response_id = Some(previous_response_id.into());
        self
    }

    pub fn conversation(mut self, conversation: impl Into<String>) -> Self {
        self.conversation = Some(conversation.into());
        self
    }

    pub fn store(mut self, store: bool) -> Self {
        self.store = Some(store);
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    pub fn build(self) -> Result<AgentRunRequest, BaochuanError> {
        let model = self
            .model
            .ok_or_else(|| BaochuanError::InvalidRequest("model must be specified".to_string()))?;
        let input = self
            .input
            .ok_or_else(|| BaochuanError::InvalidRequest("input must be specified".to_string()))?;

        if input.is_empty() {
            return Err(BaochuanError::InvalidRequest(
                "input must not be empty".to_string(),
            ));
        }

        Ok(AgentRunRequest {
            model,
            input,
            instructions: self.instructions,
            previous_response_id: self.previous_response_id,
            conversation: self.conversation,
            store: self.store,
            stream: self.stream,
        })
    }
}

/// A completed response from an agent runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    pub id: String,
    pub object: Option<String>,
    pub status: String,
    pub model: Option<String>,
    #[serde(default)]
    pub output: Vec<AgentOutputItem>,
    pub usage: Option<AgentUsage>,
}

impl AgentResponse {
    /// Concatenate every `output_text` content block in the response.
    pub fn output_text(&self) -> String {
        self.output
            .iter()
            .flat_map(|item| item.content.iter().flatten())
            .filter(|content| content.content_type == "output_text")
            .filter_map(|content| content.text.as_deref())
            .collect::<Vec<_>>()
            .join("")
    }
}

/// One output item in a Responses API response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutputItem {
    #[serde(rename = "type")]
    pub item_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<AgentOutputContent>>,
}

/// A content block in an agent message output item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutputContent {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Token usage reported by a Responses-compatible agent endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

/// One server-sent event from an agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentEvent {
    /// The SSE event name, such as `response.output_text.delta` or
    /// `hermes.tool.progress`.
    pub event: Option<String>,
    /// Raw JSON payload for the event.
    pub data: Value,
}

impl AgentEvent {
    /// Return the OpenAI Responses `delta` field for text delta events.
    pub fn output_text_delta(&self) -> Option<&str> {
        self.data.get("delta").and_then(Value::as_str)
    }
}

pub(crate) fn sse_to_agent_events<S>(
    stream: S,
) -> impl Stream<Item = Result<AgentEvent, BaochuanError>> + Send
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    let mut buffer = String::new();
    let mut current_event: Option<String> = None;
    let mut done = false;

    stream.flat_map(move |result| {
        let items: Vec<Result<AgentEvent, BaochuanError>> = match result {
            Err(e) => vec![Err(BaochuanError::Http(e))],
            Ok(bytes) => {
                if done {
                    return futures_util::stream::iter(Vec::new());
                }

                buffer.push_str(&String::from_utf8_lossy(&bytes));
                let mut events = Vec::new();

                while let Some(newline_pos) = buffer.find('\n') {
                    let line = buffer[..newline_pos].trim().to_string();
                    buffer.drain(..=newline_pos);

                    if line.is_empty() {
                        current_event = None;
                        continue;
                    }

                    if let Some(event) = line.strip_prefix("event:") {
                        current_event = Some(event.trim().to_string());
                        continue;
                    }

                    let data = match line.strip_prefix("data:") {
                        Some(rest) => rest.trim(),
                        None => continue,
                    };

                    if data == "[DONE]" {
                        done = true;
                        break;
                    }

                    match serde_json::from_str::<Value>(data) {
                        Ok(value) => events.push(Ok(AgentEvent {
                            event: current_event.clone(),
                            data: value,
                        })),
                        Err(e) => events.push(Err(BaochuanError::Stream(format!(
                            "failed to parse agent event: {e}"
                        )))),
                    }
                }

                events
            }
        };

        futures_util::stream::iter(items)
    })
}
