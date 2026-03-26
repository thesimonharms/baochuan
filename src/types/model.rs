use serde::{Deserialize, Serialize};

/// Information about a model available from a provider.
///
/// Fields are populated on a best-effort basis — providers expose different
/// levels of detail. Only [`id`](ModelInfo::id) is guaranteed to be present;
/// everything else depends on what the provider returns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// The model identifier used in [`ChatRequestBuilder::new`](crate::types::ChatRequestBuilder).
    pub id: String,

    /// Who owns or hosts this model (e.g. `"openai"`, `"meta"`, `"mistralai"`).
    pub owned_by: Option<String>,

    /// Maximum context window in tokens, if reported by the provider.
    pub context_length: Option<u32>,

    /// Human-readable display name, if available.
    pub display_name: Option<String>,
}

impl ModelInfo {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            owned_by: None,
            context_length: None,
            display_name: None,
        }
    }
}

// ── OpenAI-compatible model list (shared by OpenAI, DeepSeek, Grok, Mistral, llama.cpp) ──

#[derive(Deserialize)]
pub(crate) struct OpenAIModelList {
    pub data: Vec<OpenAIModelEntry>,
}

#[derive(Deserialize)]
pub(crate) struct OpenAIModelEntry {
    pub id: String,
    pub owned_by: Option<String>,
}

impl From<OpenAIModelEntry> for ModelInfo {
    fn from(e: OpenAIModelEntry) -> Self {
        Self {
            id: e.id,
            owned_by: e.owned_by,
            context_length: None,
            display_name: None,
        }
    }
}
