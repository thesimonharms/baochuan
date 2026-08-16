# baochuan · 宝船

**A multi-provider AI API client for Rust.**

[![Crates.io](https://img.shields.io/crates/v/baochuan.svg)](https://crates.io/crates/baochuan)
[![docs.rs](https://img.shields.io/docsrs/baochuan)](https://docs.rs/baochuan)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## The Name

> *"宝船" (Bǎochuán) — Treasure Ship*

In the early 15th century, Admiral **Zheng He** commanded the largest wooden fleet the world had ever seen. His colossal **treasure ships** — some reportedly over 400 feet long — sailed from China to the Persian Gulf, East Africa, and beyond, connecting civilizations that had never met. They carried silk, porcelain, and ideas across the Indian Ocean, opening the world to each other.

**baochuan** aspires to the same role: a vessel that carries your Rust code to every major AI provider through a single unified interface, regardless of where those providers sail.

> The sister library [**ZhengHe**](https://github.com/simonharms/zhenghe) does the same for Java — named after the explorer who famously introduced China to Java (the island). ZhengHe connects Java to the DeepSeek API; baochuan carries the voyage forward in Rust, with a whole fleet of providers.

---

## Features

- **Fully async** — built on [tokio](https://tokio.rs/) and [reqwest](https://docs.rs/reqwest)
- **Multi-provider** — swap providers without changing your business logic
- **Streaming** — native SSE streaming support for real-time token delivery
- **Builder pattern** — ergonomic, validated request construction
- **Own implementation** — no third-party SDK wrappers; direct HTTP to each provider
- **Extensible** — implement the `Provider` trait to add any model provider
- **Agent runtimes** — optional `AgentProvider` trait for stateful agent APIs such as Hermes Agent
- **Reasoning support** — `thinking` / `reasoning_effort` for DeepSeek V4, Kimi, and similar APIs
- **OpenAI-ready** — `max_tokens` is remapped to `max_completion_tokens` for o-series models

---

## Supported Providers

| Provider | Chat | Streaming | Model List | API | Env Var |
|---|:---:|:---:|:---:|---|---|
| [OpenAI](https://platform.openai.com/) | ✅ | ✅ | ✅ | OpenAI Chat Completions | `OPENAI_API_KEY` |
| [Anthropic](https://www.anthropic.com/) | ✅ | ✅ | ✅ | Anthropic Messages | `ANTHROPIC_API_KEY` |
| [Google Gemini](https://ai.google.dev/) | ✅ | ✅ | ✅ | Gemini `generateContent` | `GEMINI_API_KEY` |
| [xAI Grok](https://docs.x.ai/) | ✅ | ✅ | ✅ | OpenAI-compatible | `XAI_API_KEY` |
| [Mistral](https://mistral.ai/) | ✅ | ✅ | ✅ | OpenAI-compatible | `MISTRAL_API_KEY` |
| [DeepSeek](https://platform.deepseek.com/) | ✅ | ✅ | ✅ | OpenAI-compatible | `DEEPSEEK_API_KEY` |
| [GitHub Copilot](https://github.com/features/copilot) | ✅ | ✅ | ✅ | Copilot Chat Completions | `GITHUB_TOKEN` |
| [OpenRouter](https://openrouter.ai/) | ✅ | ✅ | ✅ | OpenAI-compatible | `OPENROUTER_API_KEY` |
| [Moonshot AI / Kimi](https://platform.kimi.ai/) | ✅ | ✅ | ✅ | OpenAI-compatible | `MOONSHOT_API_KEY` |
| [Perplexity](https://docs.perplexity.ai/) | ✅ | ✅ | ✅ | OpenAI-compatible / Sonar | `PERPLEXITY_API_KEY` |
| [Nous Portal](https://portal.nousresearch.com/) | ✅ | ✅ | ✅ | OpenAI-compatible | `NOUS_API_KEY` |
| [Alibaba Qwen](https://dashscope.aliyun.com/) | ✅ | ✅ | ✅ | DashScope compatible-mode | `DASHSCOPE_API_KEY` |
| [Groq](https://groq.com/) | ✅ | ✅ | ✅ | OpenAI-compatible | `GROQ_API_KEY` |
| [Together AI](https://www.together.ai/) | ✅ | ✅ | ✅ | OpenAI-compatible | `TOGETHER_API_KEY` |
| [Fireworks AI](https://fireworks.ai/) | ✅ | ✅ | ✅ | OpenAI-compatible | `FIREWORKS_API_KEY` |
| [Cerebras](https://inference.cerebras.ai/) | ✅ | ✅ | ✅ | OpenAI-compatible | `CEREBRAS_API_KEY` |
| [SambaNova](https://cloud.sambanova.ai/) | ✅ | ✅ | ✅ | OpenAI-compatible | `SAMBANOVA_API_KEY` |
| [Zhipu AI / Z.ai (GLM)](https://z.ai/) | ✅ | ✅ | ✅ | OpenAI-compatible | `ZHIPU_API_KEY` |
| [MiniMax](https://www.minimax.io/) | ✅ | ✅ | ✅ | OpenAI-compatible | `MINIMAX_API_KEY` |
| [Hugging Face](https://huggingface.co/docs/inference-providers) | ✅ | ✅ | ✅ | OpenAI-compatible router | `HF_TOKEN` |
| [NVIDIA NIM](https://build.nvidia.com/) | ✅ | ✅ | ✅ | OpenAI-compatible | `NVIDIA_API_KEY` |
| [DeepInfra](https://deepinfra.com/) | ✅ | ✅ | ✅ | OpenAI-compatible | `DEEPINFRA_API_KEY` |
| [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) | ✅ | ✅ | ✅ | CF native `/ai/run/` | `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_API_TOKEN` |
| [LM Studio](https://lmstudio.ai/) | ✅ | ✅ | ✅ | LM Studio `/api/v0/` | _(none)_ |
| [Ollama](https://ollama.com/) | ✅ | ✅ | ✅ | Ollama `/api/` | _(none)_ |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | ✅ | ✅ | ✅ | llama-server `/v1/` | _(none)_ |
| [Hermes Agent API](https://hermes-agent.nousresearch.com/docs/user-guide/features/api-server) | ✅ | ✅ | — | Responses `/v1/responses` | `API_SERVER_KEY` |
| More coming… | — | — | — | — | — |

---

## Installation

Add baochuan to your `Cargo.toml`:

```toml
[dependencies]
baochuan = "0.2"
tokio = { version = "1", features = ["full"] }
```

---

## Quickstart

### DeepSeek

```rust
use baochuan::{providers::DeepSeekProvider, ChatMessage, ChatRequestBuilder, Provider, ThinkingConfig};

#[tokio::main]
async fn main() {
    let provider = DeepSeekProvider::new(
        std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY not set"),
    );

    let request = ChatRequestBuilder::new("deepseek-v4-flash")
        .message(ChatMessage::user("Tell me about the treasure ships of Zheng He."))
        .thinking(ThinkingConfig::disabled())
        .max_tokens(512)
        .build()
        .unwrap();

    let response = provider.chat(&request).await.unwrap();
    println!("{}", response.content().unwrap_or("(no response)"));
}
```

### Nous Portal

```rust
use baochuan::{providers::NousProvider, ChatMessage, ChatRequestBuilder, Provider};

#[tokio::main]
async fn main() {
    let provider = NousProvider::new(
        std::env::var("NOUS_API_KEY").expect("NOUS_API_KEY not set"),
    );

    let request = ChatRequestBuilder::new("Hermes-4-405B")
        .message(ChatMessage::user("Carry the Zheng He voyage onward."))
        .build()
        .unwrap();

    let response = provider.chat(&request).await.unwrap();
    println!("{}", response.content().unwrap_or("(no response)"));
}
```

### OpenRouter

```rust
use baochuan::{providers::OpenRouterProvider, ChatMessage, ChatRequestBuilder, Provider};

#[tokio::main]
async fn main() {
    let provider = OpenRouterProvider::new(
        std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY not set"),
    )
    .site_name("My App");

    // OpenRouter gives you access to hundreds of models:
    let request = ChatRequestBuilder::new("anthropic/claude-sonnet-4")
        .message(ChatMessage::user("What is the speed of light?"))
        .build()
        .unwrap();

    let response = provider.chat(&request).await.unwrap();
    println!("{}", response.content().unwrap_or("(no response)"));
}
```

### Streaming

```rust
use baochuan::{providers::DeepSeekProvider, ChatMessage, ChatRequestBuilder, Provider};
use futures_util::StreamExt;

#[tokio::main]
async fn main() {
    let provider = DeepSeekProvider::new(
        std::env::var("DEEPSEEK_API_KEY").unwrap(),
    );

    let request = ChatRequestBuilder::new("deepseek-v4-flash")
        .message(ChatMessage::user("Write a haiku about Rust."))
        .build()
        .unwrap();

    let mut stream = provider.stream_chat(&request).await.unwrap();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        if let Some(text) = chunk.delta_content() {
            print!("{text}");
        }
    }
    println!();
}
```

### Provider-agnostic code

```rust
use baochuan::{ChatMessage, ChatRequestBuilder, Provider};

async fn ask(provider: &dyn Provider, model: &str, question: &str) -> String {
    let request = ChatRequestBuilder::new(model)
        .message(ChatMessage::user(question))
        .build()
        .unwrap();

    provider
        .chat(&request)
        .await
        .unwrap()
        .content()
        .unwrap_or("")
        .to_string()
}
```

### AgentProvider for Hermes Agent

```rust
use baochuan::{providers::HermesAgentProvider, AgentProvider, AgentRunRequestBuilder};

#[tokio::main]
async fn main() {
    let provider = HermesAgentProvider::new(
        std::env::var("API_SERVER_KEY").expect("API_SERVER_KEY not set"),
    );

    let request = AgentRunRequestBuilder::new("hermes-agent", "What files are in my project?")
        .instructions("You are a helpful coding assistant.")
        .store(true)
        .build()
        .unwrap();

    let response = provider.run(&request).await.unwrap();
    println!("{}", response.output_text());
}
```

---

## Conversation with System Prompt

```rust
use baochuan::{providers::DeepSeekProvider, ChatMessage, ChatRequestBuilder, Provider};

#[tokio::main]
async fn main() {
    let provider = DeepSeekProvider::new(std::env::var("DEEPSEEK_API_KEY").unwrap());

    let request = ChatRequestBuilder::new("deepseek-v4-flash")
        .message(ChatMessage::system("You are a concise assistant. Reply in one sentence."))
        .message(ChatMessage::user("What is baochuan?"))
        .temperature(0.7)
        .max_tokens(128)
        .build()
        .unwrap();

    let response = provider.chat(&request).await.unwrap();
    println!("{}", response.content().unwrap_or(""));
}
```

---

## Adding a Provider

Implement the `Provider` trait for any HTTP-based AI API:

```rust
use async_trait::async_trait;
use baochuan::{
    BaochuanError, ChatRequest, ChatResponse, Provider,
    provider::ChunkStream,
};

pub struct MyProvider { /* ... */ }

#[async_trait]
impl Provider for MyProvider {
    fn name(&self) -> &str { "my-provider" }

    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, BaochuanError> {
        todo!()
    }

    async fn stream_chat(&self, request: &ChatRequest) -> Result<ChunkStream, BaochuanError> {
        todo!()
    }
}
```

---

## Security

- **Never hard‑code API keys.** Always read them from environment variables or a secrets manager.
- Use `.gitignore` to exclude `.env` files from version control.
- Rotate keys immediately if you suspect they have been exposed.

```bash
export DEEPSEEK_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
```

---

## License

MIT — see [LICENSE](LICENSE).
