use async_trait::async_trait;
use reqwest::Client;
use tracing::{debug, error};

use crate::agent::{
    sse_to_agent_events, AgentEventStream, AgentProvider, AgentResponse, AgentRunRequest,
};
use crate::error::BaochuanError;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8642/v1";

/// A provider that connects to the Hermes Agent API server.
///
/// Hermes exposes an OpenAI-compatible API server with a stateful Responses API
/// endpoint at `/v1/responses`. This provider targets that higher-level agent
/// endpoint rather than plain chat completions.
///
/// # Example
/// ```rust,no_run
/// use baochuan::{providers::HermesAgentProvider, AgentProvider, AgentRunRequestBuilder};
///
/// #[tokio::main]
/// async fn main() {
///     let provider = HermesAgentProvider::new(
///         std::env::var("API_SERVER_KEY").expect("API_SERVER_KEY not set"),
///     );
///
///     let request = AgentRunRequestBuilder::new("hermes-agent", "What files are in this project?")
///         .instructions("You are a helpful coding assistant.")
///         .store(true)
///         .build()
///         .unwrap();
///
///     let response = provider.run(&request).await.unwrap();
///     println!("{}", response.output_text());
/// }
/// ```
pub struct HermesAgentProvider {
    client: Client,
    api_key: Option<String>,
    base_url: String,
}

impl HermesAgentProvider {
    /// Create a Hermes Agent provider using the default local API server URL.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: Some(api_key.into()),
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Create a Hermes Agent provider without bearer auth.
    pub fn no_key() -> Self {
        Self {
            client: Client::new(),
            api_key: None,
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Override the base URL (for example, `http://localhost:8642/v1`).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    fn responses_url(&self) -> String {
        format!("{}/responses", self.base_url.trim_end_matches('/'))
    }

    fn auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) if !key.is_empty() => builder.bearer_auth(key),
            _ => builder,
        }
    }
}

#[async_trait]
impl AgentProvider for HermesAgentProvider {
    fn name(&self) -> &str {
        "hermes-agent"
    }

    async fn run(&self, request: &AgentRunRequest) -> Result<AgentResponse, BaochuanError> {
        debug!(model = %request.model, "sending Hermes Agent responses request");

        let response = self
            .auth(self.client.post(self.responses_url()))
            .json(request)
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "Hermes Agent API error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        Ok(response.json().await?)
    }

    async fn stream_run(
        &self,
        request: &AgentRunRequest,
    ) -> Result<AgentEventStream, BaochuanError> {
        debug!(model = %request.model, "starting Hermes Agent responses stream");

        let mut body = serde_json::to_value(request)?;
        body["stream"] = serde_json::Value::Bool(true);

        let response = self
            .auth(self.client.post(self.responses_url()))
            .json(&body)
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            error!(status = %status, body = %body, "Hermes Agent stream error");
            return Err(BaochuanError::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        Ok(Box::pin(sse_to_agent_events(response.bytes_stream())))
    }
}
