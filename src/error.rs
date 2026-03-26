use thiserror::Error;

/// All errors that baochuan can produce.
#[derive(Debug, Error)]
pub enum BaochuanError {
    /// A network or transport-level failure from reqwest.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// The remote API returned a non-success status code.
    #[error("API error (HTTP {status}): {message}")]
    Api { status: u16, message: String },

    /// The response body could not be parsed as expected JSON.
    #[error("Deserialization error: {0}")]
    Deserialization(#[from] serde_json::Error),

    /// The request was invalid before it was even sent.
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// The SSE stream contained a malformed or unexpected line.
    #[error("Stream error: {0}")]
    Stream(String),
}
