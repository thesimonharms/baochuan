use baochuan::{AgentInput, AgentProvider, AgentRunRequestBuilder, providers::HermesAgentProvider};
use futures_util::StreamExt;

fn hermes_response_body() -> &'static str {
    r#"{
        "id": "resp_abc123",
        "object": "response",
        "status": "completed",
        "model": "hermes-agent",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "README.md src/ tests/"}]
        }],
        "usage": {"input_tokens": 50, "output_tokens": 12, "total_tokens": 62}
    }"#
}

#[test]
fn test_agent_run_request_builder() {
    let req = AgentRunRequestBuilder::new("hermes-agent", "What files are here?")
        .instructions("You are a coding assistant.")
        .previous_response_id("resp_previous")
        .conversation("baochuan")
        .store(true)
        .build()
        .unwrap();

    assert_eq!(req.model, "hermes-agent");
    assert!(matches!(req.input, AgentInput::Text(ref text) if text == "What files are here?"));
    assert_eq!(
        req.instructions.as_deref(),
        Some("You are a coding assistant.")
    );
    assert_eq!(req.previous_response_id.as_deref(), Some("resp_previous"));
    assert_eq!(req.conversation.as_deref(), Some("baochuan"));
    assert_eq!(req.store, Some(true));
    assert!(!req.stream);
}

#[test]
fn test_agent_run_request_requires_input() {
    let err = AgentRunRequestBuilder::new("hermes-agent", "")
        .build()
        .unwrap_err();
    assert!(err.to_string().contains("input"));
}

#[tokio::test]
async fn test_hermes_agent_run_uses_responses_api() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/responses")
        .match_header("authorization", "Bearer test-api-key")
        .match_body(mockito::Matcher::Regex("previous_response_id".to_string()))
        .match_body(mockito::Matcher::Regex("resp_previous".to_string()))
        .match_body(mockito::Matcher::Regex("What files are here".to_string()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(hermes_response_body())
        .create_async()
        .await;

    let provider = HermesAgentProvider::new("test-api-key").with_base_url(server.url() + "/v1");
    let request = AgentRunRequestBuilder::new("hermes-agent", "What files are here?")
        .instructions("You are a coding assistant.")
        .previous_response_id("resp_previous")
        .store(true)
        .build()
        .unwrap();

    let response = provider.run(&request).await.unwrap();

    assert_eq!(response.id, "resp_abc123");
    assert_eq!(response.status, "completed");
    assert_eq!(response.output_text(), "README.md src/ tests/");
    assert_eq!(response.usage.unwrap().total_tokens, 62);
    mock.assert_async().await;
}

#[tokio::test]
async fn test_hermes_agent_stream_run_yields_response_events() {
    let sse_body = concat!(
        "event: response.created\n",
        "data: {\"id\":\"resp_abc123\",\"type\":\"response.created\"}\n",
        "\n",
        "event: hermes.tool.progress\n",
        "data: {\"tool\":\"terminal\",\"status\":\"started\"}\n",
        "\n",
        "event: response.output_text.delta\n",
        "data: {\"delta\":\"README.md\"}\n",
        "\n",
        "event: response.completed\n",
        "data: {\"response\":{\"id\":\"resp_abc123\",\"status\":\"completed\"}}\n",
        "\n",
        "data: [DONE]\n",
    );

    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/responses")
        .match_header("authorization", "Bearer test-api-key")
        .match_body(mockito::Matcher::Regex("\"stream\":true".to_string()))
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse_body)
        .create_async()
        .await;

    let provider = HermesAgentProvider::new("test-api-key").with_base_url(server.url() + "/v1");
    let request = AgentRunRequestBuilder::new("hermes-agent", "List files")
        .stream(true)
        .build()
        .unwrap();

    let mut stream = provider.stream_run(&request).await.unwrap();
    let mut event_names = Vec::new();
    let mut text_delta = String::new();

    while let Some(event) = stream.next().await {
        let event = event.unwrap();
        if let Some(name) = event.event.as_deref() {
            event_names.push(name.to_string());
        }
        if let Some(delta) = event.output_text_delta() {
            text_delta.push_str(delta);
        }
    }

    assert_eq!(
        event_names,
        vec![
            "response.created",
            "hermes.tool.progress",
            "response.output_text.delta",
            "response.completed"
        ]
    );
    assert_eq!(text_delta, "README.md");
}

#[test]
fn test_hermes_agent_provider_name() {
    assert_eq!(HermesAgentProvider::new("key").name(), "hermes-agent");
}
