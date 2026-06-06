use serde::{Deserialize, Serialize};

// ── Tool definition ───────────────────────────────────────────────────────────

/// A tool (function) that the model may call during a conversation.
///
/// Serialises in the OpenAI format used by most providers:
/// ```json
/// {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
/// ```
/// Anthropic and Gemini use different wire formats; their providers convert automatically.
///
/// # Example
/// ```rust
/// use baochuan::Tool;
/// use serde_json::json;
///
/// let tool = Tool::function(
///     "get_weather",
///     "Returns the current weather for a city.",
///     json!({
///         "type": "object",
///         "properties": {
///             "location": {"type": "string", "description": "City name, e.g. Paris"},
///             "unit":     {"type": "string", "enum": ["celsius", "fahrenheit"]}
///         },
///         "required": ["location"]
///     }),
/// );
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: ToolType,
    pub function: FunctionDefinition,
}

impl Tool {
    /// Create a function tool with name, description, and a JSON Schema for its parameters.
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            tool_type: ToolType::Function,
            function: FunctionDefinition {
                name: name.into(),
                description: Some(description.into()),
                parameters,
            },
        }
    }
}

/// The type discriminant of a [`Tool`]. Currently only `Function` is defined.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    #[default]
    Function,
}

/// The definition of a callable function inside a [`Tool`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// The function name that will appear in model responses.
    pub name: String,
    /// Human-readable description that helps the model decide when to call this function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema object describing the function's parameters.
    pub parameters: serde_json::Value,
}

// ── ToolChoice ────────────────────────────────────────────────────────────────

/// Controls whether and how the model invokes tools.
///
/// Serialises in the OpenAI format. Anthropic and Gemini providers convert automatically.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// A preset string: [`ToolChoice::auto`], [`ToolChoice::required`], or [`ToolChoice::none`].
    Preset(ToolChoicePreset),
    /// Force the model to call a specific function.
    Function(ToolChoiceFunction),
}

impl ToolChoice {
    /// Let the model decide whether to call a tool.
    pub fn auto() -> Self {
        ToolChoice::Preset(ToolChoicePreset::Auto)
    }
    /// The model must call at least one tool.
    pub fn required() -> Self {
        ToolChoice::Preset(ToolChoicePreset::Required)
    }
    /// The model must not call any tool.
    pub fn none() -> Self {
        ToolChoice::Preset(ToolChoicePreset::None)
    }
    /// Force the model to call a specific function by name.
    pub fn function(name: impl Into<String>) -> Self {
        ToolChoice::Function(ToolChoiceFunction {
            choice_type: "function".to_string(),
            function: ToolChoiceFunctionName { name: name.into() },
        })
    }
}

/// A preset string value for [`ToolChoice`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoicePreset {
    Auto,
    Required,
    None,
}

/// Explicit function selection inside [`ToolChoice`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    #[serde(rename = "type")]
    pub choice_type: String,
    pub function: ToolChoiceFunctionName,
}

/// Name reference inside [`ToolChoiceFunction`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunctionName {
    pub name: String,
}

// ── Tool call in responses ────────────────────────────────────────────────────

/// A tool invocation returned by the model in an assistant message.
///
/// Appears in `ChatMessage.tool_calls` when the model decides to call a function.
/// Use the [`id`](ToolCall::id) to match with subsequent
/// [`ChatMessage::tool_result`](crate::types::ChatMessage::tool_result) messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned call identifier (matches `tool_call_id` in the tool-result reply).
    pub id: String,
    /// Always `"function"` currently.
    #[serde(rename = "type")]
    pub call_type: String,
    /// The function name and JSON-encoded arguments.
    pub function: FunctionCall,
}

/// The function name and arguments within a [`ToolCall`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Name of the function being called.
    pub name: String,
    /// JSON-encoded argument object, e.g. `"{\"location\":\"Paris\"}"`.
    pub arguments: String,
}

/// Partial tool call in a streaming response delta.
///
/// Arguments are delivered in fragments across multiple chunks. Accumulate the
/// `arguments` strings from successive chunks with the same `index` to
/// reconstruct the full JSON.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolCallDelta {
    /// Identifies which tool call this fragment belongs to when there are multiple.
    #[serde(default)]
    pub index: u32,
    /// Set only on the first chunk for this call.
    pub id: Option<String>,
    /// Always `"function"` when present.
    #[serde(rename = "type")]
    pub call_type: Option<String>,
    /// Partial function name/arguments.
    pub function: Option<FunctionCallDelta>,
}

/// Partial function call within a [`ToolCallDelta`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    /// Function name (first chunk only).
    pub name: Option<String>,
    /// Fragment of the JSON arguments string.
    pub arguments: Option<String>,
}
