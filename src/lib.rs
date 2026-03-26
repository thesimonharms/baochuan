//! # baochuan — 宝船
//!
//! **baochuan** (Bǎochuán, 宝船 — "Treasure Ship") is a multi-provider AI API
//! client for Rust. Just as Admiral Zheng He's colossal treasure ships sailed
//! from China to connect civilizations across the Indian Ocean, baochuan carries
//! your Rust code to every major AI provider through a single, unified interface.
//!
//! The companion Java library,
//! [ZhengHe](https://github.com/simonhochrein/zhenghe), is named after the
//! explorer who introduced China to Java (the island) — a fitting metaphor for
//! connecting Java to the DeepSeek API. baochuan continues that voyage, this
//! time in Rust, with a fleet of providers.
//!
//! ## Quickstart
//!
//! ```rust,no_run
//! use baochuan::{providers::DeepSeekProvider, ChatMessage, ChatRequestBuilder, Provider};
//!
//! #[tokio::main]
//! async fn main() {
//!     let provider = DeepSeekProvider::new(
//!         std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY not set"),
//!     );
//!
//!     let request = ChatRequestBuilder::new("deepseek-chat")
//!         .message(ChatMessage::user("Tell me about the treasure ships of Zheng He."))
//!         .max_tokens(512)
//!         .build()
//!         .unwrap();
//!
//!     let response = provider.chat(&request).await.unwrap();
//!     println!("{}", response.content().unwrap_or(""));
//! }
//! ```

pub mod error;
pub mod provider;
pub mod providers;
pub mod types;

// Top-level re-exports for ergonomic use
pub use error::BaochuanError;
pub use provider::Provider;
pub use types::{
    AudioInput, AudioOutput, AudioOutputConfig, ChatMessage, ChatRequest, ChatRequestBuilder,
    ChatResponse, ContentPart, DocumentInput, FunctionCall, FunctionDefinition, ImageUrl,
    MessageContent, ModelInfo, Role, Tool, ToolCall, ToolChoice, TtsRequest, TtsRequestBuilder,
    Usage,
};
