/// Integration tests for all baochuan providers.
///
/// Each test uses a `mockito` HTTP server so no real API keys are required.
/// Tests verify that providers:
///   - send requests to the correct path
///   - set the correct auth headers
///   - correctly deserialize successful responses into `ChatResponse`
///   - return `BaochuanError::Api` for non-2xx status codes
///   - stream SSE correctly
use baochuan::{
    providers::{
        AnthropicProvider, CloudflareProvider, DeepSeekProvider, GeminiProvider, GrokProvider,
        LlamaCppProvider, LmStudioProvider, MistralProvider, MoonshotProvider, NousProvider,
        OllamaProvider, OpenAIProvider, OpenRouterProvider, PerplexityProvider, QwenProvider,
    },
    AudioInput, ChatMessage, ChatRequestBuilder, ContentPart, FunctionDefinition, MessageContent,
    Provider, Tool, ToolCall, ToolChoice, TtsRequestBuilder,
};
use futures_util::StreamExt;

// ── Shared fixtures ───────────────────────────────────────────────────────────

fn openai_success_body() -> &'static str {
    r#"{
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Paris."},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    }"#
}

fn openai_stream_body() -> &'static str {
    concat!(
        "data: {\"id\":\"s1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Par\"},\"finish_reason\":null}]}\n",
        "\n",
        "data: {\"id\":\"s1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"is.\"},\"finish_reason\":\"stop\"}]}\n",
        "\n",
        "data: [DONE]\n",
    )
}

fn simple_request() -> baochuan::ChatRequest {
    ChatRequestBuilder::new("test-model")
        .message(ChatMessage::user("Capital of France?"))
        .build()
        .unwrap()
}

// ── Request builder unit tests ────────────────────────────────────────────────

#[test]
fn test_builder_produces_valid_request() {
    let req = ChatRequestBuilder::new("gpt-4o")
        .message(ChatMessage::system("Be concise."))
        .message(ChatMessage::user("Hello"))
        .max_tokens(256)
        .temperature(0.5)
        .build()
        .unwrap();

    assert_eq!(req.model, "gpt-4o");
    assert_eq!(req.messages.len(), 2);
    assert_eq!(req.max_tokens, Some(256));
    assert_eq!(req.temperature, Some(0.5));
    assert!(!req.stream);
}

#[test]
fn test_builder_requires_messages() {
    let err = ChatRequestBuilder::new("gpt-4o").build().unwrap_err();
    assert!(err.to_string().contains("message"));
}

#[test]
fn test_chat_message_constructors() {
    let u = ChatMessage::user("hi");
    let s = ChatMessage::system("sys");
    let a = ChatMessage::assistant("ok");

    assert!(matches!(u.role, baochuan::Role::User));
    assert!(matches!(s.role, baochuan::Role::System));
    assert!(matches!(a.role, baochuan::Role::Assistant));
}

// ── OpenAI ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_openai_chat_success() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(openai_success_body())
        .create_async()
        .await;

    let provider = OpenAIProvider::new("sk-test").with_base_url(server.url() + "/v1");
    let response = provider.chat(&simple_request()).await.unwrap();

    assert_eq!(response.content(), Some("Paris."));
    assert_eq!(response.id, "chatcmpl-test");
    mock.assert_async().await;
}

#[tokio::test]
async fn test_openai_chat_api_error() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/chat/completions")
        .with_status(401)
        .with_body(r#"{"error": {"message": "Invalid API key"}}"#)
        .create_async()
        .await;

    let provider = OpenAIProvider::new("bad-key").with_base_url(server.url() + "/v1");
    let err = provider.chat(&simple_request()).await.unwrap_err();

    assert!(matches!(
        err,
        baochuan::BaochuanError::Api { status: 401, .. }
    ));
}

#[tokio::test]
async fn test_openai_stream_chat() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(openai_stream_body())
        .create_async()
        .await;

    let provider = OpenAIProvider::new("sk-test").with_base_url(server.url() + "/v1");
    let req = ChatRequestBuilder::new("test-model")
        .message(ChatMessage::user("test"))
        .stream(true)
        .build()
        .unwrap();

    let mut stream = provider.stream_chat(&req).await.unwrap();
    let mut full = String::new();
    while let Some(chunk) = stream.next().await {
        if let Some(text) = chunk.unwrap().delta_content() {
            full.push_str(text);
        }
    }
    assert_eq!(full, "Paris.");
}

#[test]
fn test_openai_provider_name() {
    let p = OpenAIProvider::new("key");
    assert_eq!(p.name(), "openai");
}

// ── DeepSeek ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_deepseek_chat_success() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(openai_success_body())
        .create_async()
        .await;

    let provider = DeepSeekProvider::new("sk-test").with_base_url(server.url() + "/v1");
    let response = provider.chat(&simple_request()).await.unwrap();

    assert_eq!(response.content(), Some("Paris."));
    mock.assert_async().await;
}

#[test]
fn test_deepseek_provider_name() {
    assert_eq!(DeepSeekProvider::new("k").name(), "deepseek");
}

// ── Grok ──────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_grok_chat_success() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(openai_success_body())
        .create_async()
        .await;

    let provider = GrokProvider::new("xai-test").with_base_url(server.url() + "/v1");
    let response = provider.chat(&simple_request()).await.unwrap();

    assert_eq!(response.content(), Some("Paris."));
}

#[test]
fn test_grok_provider_name() {
    assert_eq!(GrokProvider::new("k").name(), "grok");
}

// ── Mistral ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_mistral_chat_success() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(openai_success_body())
        .create_async()
        .await;

    let provider = MistralProvider::new("msk-test").with_base_url(server.url() + "/v1");
    let response = provider.chat(&simple_request()).await.unwrap();

    assert_eq!(response.content(), Some("Paris."));
}

#[test]
fn test_mistral_provider_name() {
    assert_eq!(MistralProvider::new("k").name(), "mistral");
}

// ── OpenRouter ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_openrouter_chat_success() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/api/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(openai_success_body())
        .create_async()
        .await;

    let provider = OpenRouterProvider::new("or-test")
        .with_base_url(format!("{}/api/v1", server.url()))
        .site_name("tests");

    let response = provider.chat(&simple_request()).await.unwrap();
    assert_eq!(response.content(), Some("Paris."));
    mock.assert_async().await;
}

#[test]
fn test_openrouter_provider_name() {
    assert_eq!(OpenRouterProvider::new("k").name(), "openrouter");
}

// ── Anthropic ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_anthropic_chat_success() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/messages")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Paris."}],
                "model": "claude-opus-4-6",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 2}
            }"#,
        )
        .create_async()
        .await;

    let provider = AnthropicProvider::new("sk-ant-test").with_base_url(server.url() + "/v1");
    let response = provider.chat(&simple_request()).await.unwrap();

    assert_eq!(response.content(), Some("Paris."));
    assert_eq!(response.id, "msg_test");
    // Usage should be summed correctly
    let usage = response.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 2);
    assert_eq!(usage.total_tokens, 12);
    mock.assert_async().await;
}

#[tokio::test]
async fn test_anthropic_system_prompt_extraction() {
    let mut server = mockito::Server::new_async().await;
    // Just verify the request is sent (not the body contents); 200 response
    server
        .mock("POST", "/v1/messages")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{
                "id": "msg_2",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": "claude-opus-4-6",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 1}
            }"#,
        )
        .create_async()
        .await;

    let provider = AnthropicProvider::new("sk-ant-test").with_base_url(server.url() + "/v1");

    let req = ChatRequestBuilder::new("claude-opus-4-6")
        .message(ChatMessage::system("You are a helpful assistant."))
        .message(ChatMessage::user("Hello"))
        .build()
        .unwrap();

    let response = provider.chat(&req).await.unwrap();
    assert_eq!(response.content(), Some("ok"));
}

#[tokio::test]
async fn test_anthropic_api_error() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/messages")
        .with_status(403)
        .with_body(r#"{"type":"error","error":{"type":"permission_error","message":"Forbidden"}}"#)
        .create_async()
        .await;

    let provider = AnthropicProvider::new("bad-key").with_base_url(server.url() + "/v1");
    let err = provider.chat(&simple_request()).await.unwrap_err();

    assert!(matches!(
        err,
        baochuan::BaochuanError::Api { status: 403, .. }
    ));
}

#[tokio::test]
async fn test_anthropic_stream_chat() {
    let sse_body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_stream\",\"model\":\"claude-opus-4-6\"}}\n",
        "\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Par\"}}\n",
        "\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"is.\"}}\n",
        "\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n",
        "\n",
    );

    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/messages")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse_body)
        .create_async()
        .await;

    let provider = AnthropicProvider::new("sk-ant-test").with_base_url(server.url() + "/v1");
    let req = ChatRequestBuilder::new("claude-opus-4-6")
        .message(ChatMessage::user("test"))
        .build()
        .unwrap();

    let mut stream = provider.stream_chat(&req).await.unwrap();
    let mut full = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        if let Some(text) = chunk.delta_content() {
            full.push_str(text);
        }
    }
    assert_eq!(full, "Paris.");
}

#[test]
fn test_anthropic_provider_name() {
    assert_eq!(AnthropicProvider::new("k").name(), "anthropic");
}

// ── Gemini ────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_gemini_chat_success() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1beta/models/gemini-1.5-flash:generateContent")
        .match_query(mockito::Matcher::Any)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{
                "candidates": [{
                    "content": {"parts": [{"text": "Paris."}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0
                }],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 12
                }
            }"#,
        )
        .create_async()
        .await;

    let provider = GeminiProvider::new("gemini-test-key").with_base_url(server.url() + "/v1beta");

    let req = ChatRequestBuilder::new("gemini-1.5-flash")
        .message(ChatMessage::user("Capital of France?"))
        .build()
        .unwrap();

    let response = provider.chat(&req).await.unwrap();
    assert_eq!(response.content(), Some("Paris."));
    let usage = response.usage.unwrap();
    assert_eq!(usage.total_tokens, 12);
    mock.assert_async().await;
}

#[tokio::test]
async fn test_gemini_chat_api_error() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1beta/models/gemini-1.5-flash:generateContent")
        .match_query(mockito::Matcher::Any)
        .with_status(400)
        .with_body(r#"{"error": {"message": "Invalid request"}}"#)
        .create_async()
        .await;

    let provider = GeminiProvider::new("bad-key").with_base_url(server.url() + "/v1beta");

    let req = ChatRequestBuilder::new("gemini-1.5-flash")
        .message(ChatMessage::user("test"))
        .build()
        .unwrap();

    let err = provider.chat(&req).await.unwrap_err();
    assert!(matches!(
        err,
        baochuan::BaochuanError::Api { status: 400, .. }
    ));
}

#[tokio::test]
async fn test_gemini_stream_chat() {
    let sse_body = concat!(
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Par\"}],\"role\":\"model\"},\"finishReason\":\"UNSPECIFIED\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":5}}\n",
        "\n",
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"is.\"}],\"role\":\"model\"},\"finishReason\":\"STOP\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":3}}\n",
        "\n",
    );

    let mut server = mockito::Server::new_async().await;
    server
        .mock(
            "POST",
            "/v1beta/models/gemini-1.5-flash:streamGenerateContent",
        )
        .match_query(mockito::Matcher::Any)
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse_body)
        .create_async()
        .await;

    let provider = GeminiProvider::new("gemini-key").with_base_url(server.url() + "/v1beta");

    let req = ChatRequestBuilder::new("gemini-1.5-flash")
        .message(ChatMessage::user("test"))
        .build()
        .unwrap();

    let mut stream = provider.stream_chat(&req).await.unwrap();
    let mut full = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        if let Some(text) = chunk.delta_content() {
            full.push_str(text);
        }
    }
    assert_eq!(full, "Paris.");
}

#[test]
fn test_gemini_provider_name() {
    assert_eq!(GeminiProvider::new("k").name(), "gemini");
}

// ── models() — OpenAI-compatible providers ────────────────────────────────────

fn openai_models_body() -> &'static str {
    r#"{"object":"list","data":[
        {"id":"gpt-4o","object":"model","owned_by":"openai"},
        {"id":"gpt-4o-mini","object":"model","owned_by":"openai"}
    ]}"#
}

#[tokio::test]
async fn test_openai_models() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("GET", "/v1/models")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(openai_models_body())
        .create_async()
        .await;

    let provider = OpenAIProvider::new("sk-test").with_base_url(server.url() + "/v1");
    let models = provider.models().await.unwrap();

    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "gpt-4o");
    assert_eq!(models[0].owned_by.as_deref(), Some("openai"));
    assert_eq!(models[1].id, "gpt-4o-mini");
}

#[tokio::test]
async fn test_openrouter_models() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("GET", "/api/v1/models")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{"data":[
            {"id":"openai/gpt-4o","name":"GPT-4o","context_length":128000},
            {"id":"anthropic/claude-3-5-sonnet","name":"Claude 3.5 Sonnet","context_length":200000}
        ]}"#,
        )
        .create_async()
        .await;

    let provider =
        OpenRouterProvider::new("or-test").with_base_url(format!("{}/api/v1", server.url()));
    let models = provider.models().await.unwrap();

    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "openai/gpt-4o");
    assert_eq!(models[0].context_length, Some(128000));
    assert_eq!(models[0].display_name.as_deref(), Some("GPT-4o"));
}

#[tokio::test]
async fn test_anthropic_models() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("GET", "/v1/models")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{"data":[
            {"id":"claude-opus-4-6","display_name":"Claude Opus 4.6","type":"model"},
            {"id":"claude-sonnet-4-6","display_name":"Claude Sonnet 4.6","type":"model"}
        ]}"#,
        )
        .create_async()
        .await;

    let provider = AnthropicProvider::new("sk-ant-test").with_base_url(server.url() + "/v1");
    let models = provider.models().await.unwrap();

    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "claude-opus-4-6");
    assert_eq!(models[0].display_name.as_deref(), Some("Claude Opus 4.6"));
    assert_eq!(models[0].owned_by.as_deref(), Some("anthropic"));
}

#[tokio::test]
async fn test_gemini_models() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("GET", "/v1beta/models")
        .match_query(mockito::Matcher::Any)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"models":[
            {"name":"models/gemini-1.5-flash","displayName":"Gemini 1.5 Flash","inputTokenLimit":1000000},
            {"name":"models/gemini-1.5-pro","displayName":"Gemini 1.5 Pro","inputTokenLimit":2000000}
        ]}"#)
        .create_async()
        .await;

    let provider = GeminiProvider::new("gemini-key").with_base_url(server.url() + "/v1beta");
    let models = provider.models().await.unwrap();

    assert_eq!(models.len(), 2);
    // "models/" prefix should be stripped
    assert_eq!(models[0].id, "gemini-1.5-flash");
    assert_eq!(models[0].context_length, Some(1_000_000));
    assert_eq!(models[0].owned_by.as_deref(), Some("google"));
}

// ── LM Studio ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_lmstudio_models() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("GET", "/api/v0/models")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{"data":[
            {
                "id": "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                "type": "llm",
                "publisher": "lmstudio-community",
                "arch": "llama",
                "quantization": "Q4_K_M",
                "state": "loaded",
                "maxContextLength": 131072
            }
        ]}"#,
        )
        .create_async()
        .await;

    let provider = LmStudioProvider::new().with_base_url(server.url());
    let models = provider.models().await.unwrap();

    assert_eq!(models.len(), 1);
    assert_eq!(
        models[0].id,
        "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
    );
    assert_eq!(models[0].owned_by.as_deref(), Some("lmstudio-community"));
    assert_eq!(models[0].context_length, Some(131_072));
    assert_eq!(models[0].display_name.as_deref(), Some("llama · Q4_K_M"));
}

#[tokio::test]
async fn test_lmstudio_chat_success() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/api/v0/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(openai_success_body())
        .create_async()
        .await;

    let provider = LmStudioProvider::new().with_base_url(server.url());
    let response = provider.chat(&simple_request()).await.unwrap();

    assert_eq!(response.content(), Some("Paris."));
}

#[test]
fn test_lmstudio_provider_name() {
    assert_eq!(LmStudioProvider::new().name(), "lmstudio");
}

// ── Ollama ────────────────────────────────────────────────────────────────────

fn ollama_chat_response_body() -> &'static str {
    r#"{
        "model": "llama3.2",
        "created_at": "2024-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": "Paris."},
        "done": true,
        "prompt_eval_count": 10,
        "eval_count": 2
    }"#
}

#[tokio::test]
async fn test_ollama_models() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("GET", "/api/tags")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{"models":[
            {
                "name": "llama3.2:latest",
                "size": 2019393189,
                "details": {
                    "family": "llama",
                    "parameter_size": "3.2B",
                    "quantization_level": "Q4_K_M"
                }
            },
            {
                "name": "mistral:latest",
                "size": 4109000000,
                "details": {
                    "family": "mistral",
                    "parameter_size": "7B",
                    "quantization_level": "Q4_0"
                }
            }
        ]}"#,
        )
        .create_async()
        .await;

    let provider = OllamaProvider::new().with_base_url(server.url());
    let models = provider.models().await.unwrap();

    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "llama3.2:latest");
    assert_eq!(models[0].owned_by.as_deref(), Some("llama"));
    assert_eq!(models[0].display_name.as_deref(), Some("3.2B · Q4_K_M"));
    assert_eq!(models[1].id, "mistral:latest");
}

#[tokio::test]
async fn test_ollama_chat_success() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/api/chat")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(ollama_chat_response_body())
        .create_async()
        .await;

    let provider = OllamaProvider::new().with_base_url(server.url());
    let response = provider.chat(&simple_request()).await.unwrap();

    assert_eq!(response.content(), Some("Paris."));
    let usage = response.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 2);
    assert_eq!(usage.total_tokens, 12);
}

#[tokio::test]
async fn test_ollama_stream_chat() {
    let ndjson = concat!(
        "{\"model\":\"llama3.2\",\"created_at\":\"2024-01-01T00:00:00Z\",\"message\":{\"role\":\"assistant\",\"content\":\"Par\"},\"done\":false}\n",
        "{\"model\":\"llama3.2\",\"created_at\":\"2024-01-01T00:00:00Z\",\"message\":{\"role\":\"assistant\",\"content\":\"is.\"},\"done\":false}\n",
        "{\"model\":\"llama3.2\",\"created_at\":\"2024-01-01T00:00:00Z\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true}\n",
    );

    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/api/chat")
        .with_status(200)
        .with_body(ndjson)
        .create_async()
        .await;

    let provider = OllamaProvider::new().with_base_url(server.url());
    let req = ChatRequestBuilder::new("llama3.2")
        .message(ChatMessage::user("test"))
        .build()
        .unwrap();

    let mut stream = provider.stream_chat(&req).await.unwrap();
    let mut full = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        if let Some(text) = chunk.delta_content() {
            full.push_str(text);
        }
    }
    assert_eq!(full, "Paris.");
}

#[test]
fn test_ollama_provider_name() {
    assert_eq!(OllamaProvider::new().name(), "ollama");
}

// ── llama.cpp ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_llamacpp_models() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("GET", "/v1/models")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{"object":"list","data":[{"id":"llama","object":"model","owned_by":"llamacpp"}]}"#,
        )
        .create_async()
        .await;

    let provider = LlamaCppProvider::new().with_base_url(server.url());
    let models = provider.models().await.unwrap();

    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "llama");
    assert_eq!(models[0].owned_by.as_deref(), Some("llamacpp"));
}

#[tokio::test]
async fn test_llamacpp_chat_success() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(openai_success_body())
        .create_async()
        .await;

    let provider = LlamaCppProvider::new().with_base_url(server.url());
    let response = provider.chat(&simple_request()).await.unwrap();

    assert_eq!(response.content(), Some("Paris."));
}

#[test]
fn test_llamacpp_provider_name() {
    assert_eq!(LlamaCppProvider::new().name(), "llamacpp");
}

// ── Moonshot AI ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_moonshot_chat_success() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(openai_success_body())
        .create_async()
        .await;

    let provider = MoonshotProvider::new("sk-moon").with_base_url(server.url() + "/v1");
    let response = provider.chat(&simple_request()).await.unwrap();
    assert_eq!(response.content(), Some("Paris."));
}

#[tokio::test]
async fn test_moonshot_models() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("GET", "/v1/models")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{"object":"list","data":[
            {"id":"moonshot-v1-8k","object":"model","owned_by":"moonshot"},
            {"id":"moonshot-v1-128k","object":"model","owned_by":"moonshot"}
        ]}"#,
        )
        .create_async()
        .await;

    let provider = MoonshotProvider::new("sk-moon").with_base_url(server.url() + "/v1");
    let models = provider.models().await.unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "moonshot-v1-8k");
}

#[test]
fn test_moonshot_provider_name() {
    assert_eq!(MoonshotProvider::new("k").name(), "moonshot");
}

// ── Nous Portal ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_nous_chat_success() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/chat/completions")
        .match_header("authorization", "Bearer nous-test")
        .match_body(mockito::Matcher::Regex("Hermes-4-405B".to_string()))
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(openai_success_body())
        .create_async()
        .await;

    let provider = NousProvider::new("nous-test").with_base_url(server.url() + "/v1");
    let req = ChatRequestBuilder::new("Hermes-4-405B")
        .message(ChatMessage::user("Capital of France?"))
        .build()
        .unwrap();

    let response = provider.chat(&req).await.unwrap();
    assert_eq!(response.content(), Some("Paris."));
    mock.assert_async().await;
}

#[tokio::test]
async fn test_nous_models() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("GET", "/v1/models")
        .match_header("authorization", "Bearer nous-test")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{"object":"list","data":[
            {"id":"Hermes-4-405B","object":"model","owned_by":"nousresearch"},
            {"id":"Hermes-4-70B","object":"model","owned_by":"nousresearch"}
        ]}"#,
        )
        .create_async()
        .await;

    let provider = NousProvider::new("nous-test").with_base_url(server.url() + "/v1");
    let models = provider.models().await.unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "Hermes-4-405B");
}

#[test]
fn test_nous_provider_name() {
    assert_eq!(NousProvider::new("k").name(), "nous");
}

// ── Cloudflare Workers AI ─────────────────────────────────────────────────────

#[tokio::test]
async fn test_cloudflare_chat_success() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock(
            "POST",
            "/client/v4/accounts/acct123/ai/run/@cf/meta/llama-3.1-8b-instruct",
        )
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"result":{"response":"Paris."},"success":true,"errors":[],"messages":[]}"#)
        .create_async()
        .await;

    let provider =
        CloudflareProvider::new("acct123", "token123").with_base_url(server.url() + "/client/v4");

    let req = ChatRequestBuilder::new("@cf/meta/llama-3.1-8b-instruct")
        .message(ChatMessage::user("Capital of France?"))
        .build()
        .unwrap();

    let response = provider.chat(&req).await.unwrap();
    assert_eq!(response.content(), Some("Paris."));
}

#[tokio::test]
async fn test_cloudflare_stream_chat() {
    let sse_body = concat!(
        "data: {\"response\":\"Par\",\"p\":\"abc\"}\n",
        "\n",
        "data: {\"response\":\"is.\",\"p\":\"def\"}\n",
        "\n",
        "data: [DONE]\n",
    );

    let mut server = mockito::Server::new_async().await;
    server
        .mock(
            "POST",
            "/client/v4/accounts/acct123/ai/run/@cf/meta/llama-3.1-8b-instruct",
        )
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse_body)
        .create_async()
        .await;

    let provider =
        CloudflareProvider::new("acct123", "token123").with_base_url(server.url() + "/client/v4");

    let req = ChatRequestBuilder::new("@cf/meta/llama-3.1-8b-instruct")
        .message(ChatMessage::user("test"))
        .build()
        .unwrap();

    let mut stream = provider.stream_chat(&req).await.unwrap();
    let mut full = String::new();
    while let Some(chunk) = stream.next().await {
        if let Some(text) = chunk.unwrap().delta_content() {
            full.push_str(text);
        }
    }
    assert_eq!(full, "Paris.");
}

#[tokio::test]
async fn test_cloudflare_models() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("GET", "/client/v4/accounts/acct123/ai/models/search")
        .match_query(mockito::Matcher::Any)
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"result":[
            {"id":"@cf/meta/llama-3.1-8b-instruct","description":"Llama 3.1 8B","task":{"id":"t1","name":"Text Generation"},"tags":[]},
            {"id":"@cf/mistral/mistral-7b-instruct-v0.1","description":"Mistral 7B","task":{"id":"t1","name":"Text Generation"},"tags":[]}
        ],"success":true,"result_info":{"page":1,"per_page":50,"count":2,"total_count":2}}"#)
        .create_async()
        .await;

    let provider =
        CloudflareProvider::new("acct123", "token123").with_base_url(server.url() + "/client/v4");
    let models = provider.models().await.unwrap();

    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "@cf/meta/llama-3.1-8b-instruct");
    assert_eq!(models[0].display_name.as_deref(), Some("Llama 3.1 8B"));
    assert_eq!(models[0].owned_by.as_deref(), Some("Text Generation"));
}

#[tokio::test]
async fn test_cloudflare_api_error_in_envelope() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/client/v4/accounts/acct123/ai/run/@cf/meta/llama-3.1-8b-instruct")
        .with_status(200)
        .with_body(r#"{"result":null,"success":false,"errors":[{"code":1000,"message":"Invalid model"}],"messages":[]}"#)
        .create_async()
        .await;

    let provider =
        CloudflareProvider::new("acct123", "token123").with_base_url(server.url() + "/client/v4");

    let req = ChatRequestBuilder::new("@cf/meta/llama-3.1-8b-instruct")
        .message(ChatMessage::user("test"))
        .build()
        .unwrap();

    let err = provider.chat(&req).await.unwrap_err();
    assert!(matches!(err, baochuan::BaochuanError::Api { .. }));
}

#[test]
fn test_cloudflare_provider_name() {
    assert_eq!(CloudflareProvider::new("a", "t").name(), "cloudflare");
}

// ── Perplexity ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_perplexity_chat_with_citations() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            r#"{
            "id": "ppl-test",
            "model": "llama-3.1-sonar-small-128k-online",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Paris is the capital of France."},
                "finish_reason": "stop"
            }],
            "citations": ["https://en.wikipedia.org/wiki/Paris"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
        }"#,
        )
        .create_async()
        .await;

    let provider = PerplexityProvider::new("ppl-key").with_base_url(server.url());
    let response = provider.chat(&simple_request()).await.unwrap();

    assert_eq!(response.content(), Some("Paris is the capital of France."));
    let citations = response.citations.unwrap();
    assert_eq!(citations.len(), 1);
    assert_eq!(citations[0], "https://en.wikipedia.org/wiki/Paris");
}

#[tokio::test]
async fn test_perplexity_models() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("GET", "/models")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"data":[
            {"id":"llama-3.1-sonar-small-128k-online","owned_by":"perplexity","context_length":127072},
            {"id":"llama-3.1-sonar-large-128k-online","owned_by":"perplexity","context_length":127072}
        ]}"#)
        .create_async()
        .await;

    let provider = PerplexityProvider::new("ppl-key").with_base_url(server.url());
    let models = provider.models().await.unwrap();

    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "llama-3.1-sonar-small-128k-online");
    assert_eq!(models[0].context_length, Some(127_072));
}

#[tokio::test]
async fn test_perplexity_no_citations_when_absent() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/chat/completions")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{
            "id": "ppl-test2",
            "model": "llama-3.1-sonar-small-128k-chat",
            "choices": [{"index":0,"message":{"role":"assistant","content":"Paris."},"finish_reason":"stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
        }"#)
        .create_async()
        .await;

    let provider = PerplexityProvider::new("ppl-key").with_base_url(server.url());
    let response = provider.chat(&simple_request()).await.unwrap();

    assert!(response.citations.is_none());
}

#[test]
fn test_perplexity_provider_name() {
    assert_eq!(PerplexityProvider::new("k").name(), "perplexity");
}

// ── Qwen / DashScope ──────────────────────────────────────────────────────────

fn qwen_chat_response_body() -> &'static str {
    r#"{
        "output": {
            "choices": [{
                "message": {"role": "assistant", "content": "Paris."},
                "finish_reason": "stop"
            }]
        },
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        "request_id": "qwen-req-1"
    }"#
}

#[tokio::test]
async fn test_qwen_chat_success() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/api/v1/services/aigc/text-generation/generation")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(qwen_chat_response_body())
        .create_async()
        .await;

    let provider = QwenProvider::new("sk-dashscope").with_base_url(server.url() + "/api/v1");

    let response = provider.chat(&simple_request()).await.unwrap();
    assert_eq!(response.content(), Some("Paris."));
    assert_eq!(response.id, "qwen-req-1");
    let usage = response.usage.unwrap();
    assert_eq!(usage.total_tokens, 12);
}

#[tokio::test]
async fn test_qwen_stream_chat() {
    let sse_body = concat!(
        "id:1\n",
        "event:result\n",
        ":HTTP_STATUS/200\n",
        "data:{\"output\":{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"Par\"},\"finish_reason\":\"null\"}]},\"usage\":{\"input_tokens\":10,\"output_tokens\":1,\"total_tokens\":11},\"request_id\":\"r1\"}\n",
        "\n",
        "id:2\n",
        "event:result\n",
        ":HTTP_STATUS/200\n",
        "data:{\"output\":{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"is.\"},\"finish_reason\":\"stop\"}]},\"usage\":{\"input_tokens\":10,\"output_tokens\":3,\"total_tokens\":13},\"request_id\":\"r1\"}\n",
        "\n",
    );

    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/api/v1/services/aigc/text-generation/generation")
        .with_status(200)
        .with_header("content-type", "text/event-stream")
        .with_body(sse_body)
        .create_async()
        .await;

    let provider = QwenProvider::new("sk-dashscope").with_base_url(server.url() + "/api/v1");

    let req = ChatRequestBuilder::new("qwen-turbo")
        .message(ChatMessage::user("test"))
        .build()
        .unwrap();

    let mut stream = provider.stream_chat(&req).await.unwrap();
    let mut full = String::new();
    while let Some(chunk) = stream.next().await {
        if let Some(text) = chunk.unwrap().delta_content() {
            full.push_str(text);
        }
    }
    assert_eq!(full, "Paris.");
}

#[test]
fn test_qwen_provider_name() {
    assert_eq!(QwenProvider::new("k").name(), "qwen");
}

// ── Provider trait object ─────────────────────────────────────────────────────

#[test]
fn test_provider_is_object_safe() {
    let providers: Vec<Box<dyn baochuan::Provider>> = vec![
        Box::new(OpenAIProvider::new("k")),
        Box::new(DeepSeekProvider::new("k")),
        Box::new(GrokProvider::new("k")),
        Box::new(MistralProvider::new("k")),
        Box::new(AnthropicProvider::new("k")),
        Box::new(GeminiProvider::new("k")),
        Box::new(OpenRouterProvider::new("k")),
        Box::new(MoonshotProvider::new("k")),
        Box::new(PerplexityProvider::new("k")),
        Box::new(QwenProvider::new("k")),
        Box::new(CloudflareProvider::new("a", "t")),
        Box::new(LmStudioProvider::new()),
        Box::new(OllamaProvider::new()),
        Box::new(LlamaCppProvider::new()),
    ];

    let names: Vec<&str> = providers.iter().map(|p| p.name()).collect();
    assert_eq!(
        names,
        [
            "openai",
            "deepseek",
            "grok",
            "mistral",
            "anthropic",
            "gemini",
            "openrouter",
            "moonshot",
            "perplexity",
            "qwen",
            "cloudflare",
            "lmstudio",
            "ollama",
            "llamacpp"
        ]
    );
}

// ── Multimodal ────────────────────────────────────────────────────────────────

#[test]
fn test_message_content_text_serializes_as_string() {
    let msg = ChatMessage::user("hello");
    let json = serde_json::to_value(&msg).unwrap();
    assert_eq!(json["content"], serde_json::json!("hello"));
}

#[test]
fn test_message_content_parts_serializes_as_array() {
    let msg = ChatMessage::user_with_image("describe this", "https://example.com/img.jpg");
    let json = serde_json::to_value(&msg).unwrap();
    let content = &json["content"];
    assert!(content.is_array());
    let parts = content.as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[0]["text"], "describe this");
    assert_eq!(parts[1]["type"], "image_url");
    assert_eq!(parts[1]["image_url"]["url"], "https://example.com/img.jpg");
}

#[test]
fn test_message_content_with_parts_constructor() {
    let msg = ChatMessage::with_parts(
        baochuan::Role::User,
        vec![
            ContentPart::text("what is in this image?"),
            ContentPart::image_url("data:image/png;base64,abc123"),
        ],
    );
    let json = serde_json::to_value(&msg).unwrap();
    let parts = json["content"].as_array().unwrap();
    assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,abc123");
}

#[test]
fn test_message_content_to_text_lossy_skips_images() {
    let content = MessageContent::Parts(vec![
        ContentPart::text("hello "),
        ContentPart::image_url("https://example.com/img.jpg"),
        ContentPart::text("world"),
    ]);
    assert_eq!(content.to_text_lossy(), "hello world");
}

#[test]
fn test_message_content_as_str_finds_first_text() {
    let content = MessageContent::Parts(vec![
        ContentPart::image_url("https://example.com/img.jpg"),
        ContentPart::text("caption"),
    ]);
    assert_eq!(content.as_str(), Some("caption"));
}

#[tokio::test]
async fn test_openai_multimodal_chat() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_body(openai_success_body())
        .create_async()
        .await;

    let provider = OpenAIProvider::new("test-key").with_base_url(server.url() + "/v1");

    let req = ChatRequestBuilder::new("gpt-4o")
        .message(ChatMessage::user_with_image(
            "What is in this image?",
            "https://example.com/photo.jpg",
        ))
        .build()
        .unwrap();

    let resp = provider.chat(&req).await.unwrap();
    assert_eq!(resp.content(), Some("Paris."));
}

#[tokio::test]
async fn test_anthropic_multimodal_chat() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/messages")
        .with_status(200)
        .with_body(
            r#"{
            "id": "msg-mm",
            "type": "message",
            "model": "claude-opus-4-6",
            "content": [{"type": "text", "text": "A cat."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 3}
        }"#,
        )
        .create_async()
        .await;

    let provider = AnthropicProvider::new("test-key").with_base_url(server.url() + "/v1");

    let req = ChatRequestBuilder::new("claude-opus-4-6")
        .message(ChatMessage::user_with_image(
            "What is in this image?",
            "data:image/jpeg;base64,/9j/4AAQSk==",
        ))
        .build()
        .unwrap();

    let resp = provider.chat(&req).await.unwrap();
    assert_eq!(resp.content(), Some("A cat."));
}

#[tokio::test]
async fn test_gemini_multimodal_chat() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock(
            "POST",
            mockito::Matcher::Regex(
                r"/v1beta/models/gemini-1\.5-flash:generateContent".to_string(),
            ),
        )
        .match_query(mockito::Matcher::Any)
        .with_status(200)
        .with_body(
            r#"{
            "candidates": [{
                "content": {"parts": [{"text": "A sunset."}]},
                "finishReason": "STOP",
                "index": 0
            }]
        }"#,
        )
        .create_async()
        .await;

    let provider = GeminiProvider::new("test-key").with_base_url(server.url() + "/v1beta");

    let req = ChatRequestBuilder::new("gemini-1.5-flash")
        .message(ChatMessage::user_with_image(
            "Describe this image.",
            "data:image/jpeg;base64,/9j/4AAQSk==",
        ))
        .build()
        .unwrap();

    let resp = provider.chat(&req).await.unwrap();
    assert_eq!(resp.content(), Some("A sunset."));
}

// ── Audio input / output / TTS ────────────────────────────────────────────────

#[test]
fn test_audio_input_part_serializes_correctly() {
    let part = ContentPart::audio("base64audiodata==", "wav");
    let json = serde_json::to_value(&part).unwrap();
    assert_eq!(json["type"], "input_audio");
    assert_eq!(json["input_audio"]["data"], "base64audiodata==");
    assert_eq!(json["input_audio"]["format"], "wav");
}

#[test]
fn test_document_part_serializes_correctly() {
    let part = ContentPart::document("base64pdfdata==", "application/pdf");
    let json = serde_json::to_value(&part).unwrap();
    assert_eq!(json["type"], "document");
    assert_eq!(json["document"]["media_type"], "application/pdf");
}

#[test]
fn test_audio_input_mime_type() {
    let a = AudioInput {
        data: "x".into(),
        format: "mp3".into(),
    };
    assert_eq!(a.mime_type(), "audio/mpeg");
    let b = AudioInput {
        data: "x".into(),
        format: "wav".into(),
    };
    assert_eq!(b.mime_type(), "audio/wav");
    let c = AudioInput {
        data: "x".into(),
        format: "flac".into(),
    };
    assert_eq!(c.mime_type(), "audio/flac");
}

#[test]
fn test_audio_output_deserializes_on_message() {
    // GPT-4o audio responses have null content and an `audio` field.
    let json = r#"{
        "role": "assistant",
        "content": null,
        "audio": {"id": "aud-1", "data": "base64audio==", "expires_at": 9999, "transcript": "Hi"}
    }"#;
    let msg: ChatMessage = serde_json::from_str(json).unwrap();
    assert!(msg.audio.is_some());
    let audio = msg.audio.unwrap();
    assert_eq!(audio.transcript.as_deref(), Some("Hi"));
    assert_eq!(audio.data, "base64audio==");
    // content should default to empty string
    assert_eq!(msg.content.to_text_lossy(), "");
}

#[test]
fn test_chat_request_modalities_serialized() {
    use baochuan::ChatRequestBuilder;
    let req = ChatRequestBuilder::new("gpt-4o-audio-preview")
        .message(ChatMessage::user("Say hello."))
        .modalities(["text", "audio"])
        .audio_output("alloy", "wav")
        .build()
        .unwrap();

    let json = serde_json::to_value(&req).unwrap();
    assert_eq!(json["modalities"], serde_json::json!(["text", "audio"]));
    assert_eq!(json["audio"]["voice"], "alloy");
    assert_eq!(json["audio"]["format"], "wav");
}

#[tokio::test]
async fn test_openai_tts() {
    let mut server = mockito::Server::new_async().await;
    let fake_audio = b"RIFF....WAVEfmt ";
    server
        .mock("POST", "/v1/audio/speech")
        .with_status(200)
        .with_header("content-type", "audio/wav")
        .with_body(fake_audio.as_ref())
        .create_async()
        .await;

    let provider = OpenAIProvider::new("test-key").with_base_url(server.url() + "/v1");

    let req = TtsRequestBuilder::new("tts-1", "Hello world")
        .voice("nova")
        .format("wav")
        .build()
        .unwrap();

    let audio_bytes = provider.tts(&req).await.unwrap();
    assert_eq!(audio_bytes, fake_audio);
}

#[tokio::test]
async fn test_tts_not_supported_by_default() {
    // DeepSeek doesn't implement TTS — should return InvalidRequest.
    use baochuan::providers::DeepSeekProvider;
    let provider = DeepSeekProvider::new("key");
    let req = TtsRequestBuilder::new("tts-1", "test")
        .voice("alloy")
        .build()
        .unwrap();
    let err = provider.tts(&req).await.unwrap_err();
    assert!(matches!(err, baochuan::BaochuanError::InvalidRequest(_)));
}

#[tokio::test]
async fn test_anthropic_document_input() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/messages")
        .with_status(200)
        .with_body(
            r#"{
            "id": "msg-doc",
            "type": "message",
            "model": "claude-opus-4-6",
            "content": [{"type": "text", "text": "Three pages."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 500, "output_tokens": 5}
        }"#,
        )
        .create_async()
        .await;

    let provider = AnthropicProvider::new("test-key").with_base_url(server.url() + "/v1");

    let req = ChatRequestBuilder::new("claude-opus-4-6")
        .message(ChatMessage::with_parts(
            baochuan::Role::User,
            vec![
                ContentPart::document("base64pdfdata==", "application/pdf"),
                ContentPart::text("How many pages is this document?"),
            ],
        ))
        .build()
        .unwrap();

    let resp = provider.chat(&req).await.unwrap();
    assert_eq!(resp.content(), Some("Three pages."));
}

// ── Tool use tests ────────────────────────────────────────────────────────────

#[test]
fn test_tool_serialises_in_openai_format() {
    let tool = Tool::function(
        "get_weather",
        "Get the weather for a city",
        serde_json::json!({
            "type": "object",
            "properties": { "location": {"type": "string"} },
            "required": ["location"]
        }),
    );
    let json = serde_json::to_value(&tool).unwrap();
    assert_eq!(json["type"], "function");
    assert_eq!(json["function"]["name"], "get_weather");
    assert_eq!(
        json["function"]["description"],
        "Get the weather for a city"
    );
    assert!(json["function"]["parameters"].is_object());
}

#[test]
fn test_tool_choice_presets_serialise() {
    assert_eq!(serde_json::to_value(ToolChoice::auto()).unwrap(), "auto");
    assert_eq!(
        serde_json::to_value(ToolChoice::required()).unwrap(),
        "required"
    );
    assert_eq!(serde_json::to_value(ToolChoice::none()).unwrap(), "none");
}

#[test]
fn test_tool_choice_function_serialises() {
    let choice = ToolChoice::function("get_weather");
    let json = serde_json::to_value(&choice).unwrap();
    assert_eq!(json["type"], "function");
    assert_eq!(json["function"]["name"], "get_weather");
}

#[test]
fn test_tool_call_deserialises() {
    let raw = r#"{
        "id": "call_abc123",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}
    }"#;
    let tc: ToolCall = serde_json::from_str(raw).unwrap();
    assert_eq!(tc.id, "call_abc123");
    assert_eq!(tc.function.name, "get_weather");
    assert_eq!(tc.function.arguments, r#"{"location":"Paris"}"#);
}

#[test]
fn test_chat_message_tool_result_constructor() {
    let msg = ChatMessage::tool_result("call_abc123", "Sunny, 22C");
    assert_eq!(msg.role, baochuan::Role::Tool);
    assert_eq!(msg.tool_call_id, Some("call_abc123".to_string()));
    assert_eq!(msg.content.as_str(), Some("Sunny, 22C"));
    assert!(msg.tool_calls.is_none());
}

#[test]
fn test_chat_request_with_tools_serialises() {
    let req = ChatRequestBuilder::new("gpt-4o")
        .message(ChatMessage::user("Weather in Paris?"))
        .tool(Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({"type": "object", "properties": {"location": {"type": "string"}}}),
        ))
        .tool_choice(ToolChoice::auto())
        .build()
        .unwrap();

    let json = serde_json::to_value(&req).unwrap();
    assert!(json["tools"].is_array());
    assert_eq!(json["tools"][0]["function"]["name"], "get_weather");
    assert_eq!(json["tool_choice"], "auto");
}

#[test]
fn test_function_definition_no_description_omitted() {
    // description is optional — when None it should not appear in JSON
    let fd = FunctionDefinition {
        name: "noop".to_string(),
        description: None,
        parameters: serde_json::json!({"type": "object", "properties": {}}),
    };
    let json = serde_json::to_value(&fd).unwrap();
    assert!(json.get("description").is_none());
}

#[tokio::test]
async fn test_openai_tool_call_response() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/chat/completions")
        .with_status(200)
        .with_body(
            r#"{
            "id": "chatcmpl-tool",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"Paris\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35}
        }"#,
        )
        .create_async()
        .await;

    let provider = OpenAIProvider::new("test-key").with_base_url(server.url() + "/v1");
    let req = ChatRequestBuilder::new("gpt-4o")
        .message(ChatMessage::user("Weather in Paris?"))
        .tool(Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({"type": "object", "properties": {}}),
        ))
        .build()
        .unwrap();

    let resp = provider.chat(&req).await.unwrap();
    let tool_calls = resp.choices[0].message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "call_abc123");
    assert_eq!(tool_calls[0].function.name, "get_weather");
    assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("tool_calls"));
}

#[tokio::test]
async fn test_anthropic_tool_call_response() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/messages")
        .with_status(200)
        .with_body(
            r#"{
            "id": "msg-tool",
            "type": "message",
            "model": "claude-opus-4-6",
            "content": [
                {"type": "text", "text": "I will check the weather."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"location": "Paris"}
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 30, "output_tokens": 20}
        }"#,
        )
        .create_async()
        .await;

    let provider = AnthropicProvider::new("test-key").with_base_url(server.url() + "/v1");
    let req = ChatRequestBuilder::new("claude-opus-4-6")
        .message(ChatMessage::user("Weather in Paris?"))
        .tool(Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({"type": "object", "properties": {}}),
        ))
        .build()
        .unwrap();

    let resp = provider.chat(&req).await.unwrap();
    let msg = &resp.choices[0].message;
    assert_eq!(msg.content.as_str(), Some("I will check the weather."));
    let tool_calls = msg.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "toolu_01");
    assert_eq!(tool_calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Paris");
}

#[tokio::test]
async fn test_gemini_tool_call_response() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock(
            "POST",
            mockito::Matcher::Regex(".*generateContent.*".to_string()),
        )
        .with_status(200)
        .with_body(
            r#"{
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"location": "Paris"}
                        }
                    }]
                },
                "finishReason": "STOP",
                "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 10,
                "totalTokenCount": 30
            }
        }"#,
        )
        .create_async()
        .await;

    let provider = GeminiProvider::new("test-key").with_base_url(server.url());
    let req = ChatRequestBuilder::new("gemini-1.5-flash")
        .message(ChatMessage::user("Weather in Paris?"))
        .tool(Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({"type": "object", "properties": {}}),
        ))
        .build()
        .unwrap();

    let resp = provider.chat(&req).await.unwrap();
    let tool_calls = resp.choices[0].message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Paris");
}

#[tokio::test]
async fn test_anthropic_tool_result_round_trip() {
    // Verify the tool result is sent with the correct Anthropic wire format.
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/messages")
        .with_status(200)
        .match_body(mockito::Matcher::Regex("toolu_01".to_string()))
        .with_body(
            r#"{
            "id": "msg-2",
            "type": "message",
            "model": "claude-opus-4-6",
            "content": [{"type": "text", "text": "It is sunny in Paris."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 10}
        }"#,
        )
        .create_async()
        .await;

    let provider = AnthropicProvider::new("test-key").with_base_url(server.url() + "/v1");

    let mut assistant_msg = ChatMessage::assistant("I will check the weather.");
    assistant_msg.tool_calls = Some(vec![ToolCall {
        id: "toolu_01".to_string(),
        call_type: "function".to_string(),
        function: baochuan::FunctionCall {
            name: "get_weather".to_string(),
            arguments: r#"{"location":"Paris"}"#.to_string(),
        },
    }]);

    let req = ChatRequestBuilder::new("claude-opus-4-6")
        .message(ChatMessage::user("Weather in Paris?"))
        .message(assistant_msg)
        .message(ChatMessage::tool_result("toolu_01", "Sunny, 22C"))
        .build()
        .unwrap();

    let resp = provider.chat(&req).await.unwrap();
    assert_eq!(resp.content(), Some("It is sunny in Paris."));
}
