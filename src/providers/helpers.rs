use reqwest::RequestBuilder;
use tracing::error;

use crate::error::BaochuanError;
use crate::types::model::{ModelInfo, OpenAIModelList};

// ── Multimodal helpers ────────────────────────────────────────────────────────

/// Parses a `data:<media_type>;base64,<data>` URL.
///
/// Returns `(media_type, base64_data)` or `None` if the URL is not a data URL.
pub(crate) fn parse_data_url(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    let (media_type, _encoding) = meta.split_once(';')?;
    Some((media_type.to_string(), data.to_string()))
}

/// Guesses the MIME type of an image from its URL extension.
/// Defaults to `"image/jpeg"` for unknown extensions.
pub(crate) fn guess_image_mime_type(url: &str) -> &'static str {
    let lower = url.to_lowercase();
    if lower.contains(".png") {
        "image/png"
    } else if lower.contains(".gif") {
        "image/gif"
    } else if lower.contains(".webp") {
        "image/webp"
    } else {
        "image/jpeg"
    }
}

/// Fetch a model list from an OpenAI-compatible `GET /v1/models` endpoint.
///
/// `build_request` receives the `Client` and must return a configured
/// `RequestBuilder` (with auth headers, URL, etc.) ready to `.send()`.
pub async fn fetch_openai_models(request: RequestBuilder) -> Result<Vec<ModelInfo>, BaochuanError> {
    let response = request.send().await?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        error!(status = %status, body = %body, "model list API error");
        return Err(BaochuanError::Api {
            status: status.as_u16(),
            message: body,
        });
    }

    let list: OpenAIModelList = response.json().await?;
    Ok(list.data.into_iter().map(ModelInfo::from).collect())
}
