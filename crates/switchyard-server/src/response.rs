// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response encoding glue for libsy server endpoints.

use std::error::Error;
use std::sync::Arc;

use axum::Json;
use axum::response::{IntoResponse, Response as HttpResponse};
use switchyard_protocol::{LlmResponse, ProviderExtensions, Response as AlgorithmResponse};
use switchyard_translation::{
    WireFormat, encode_aggregated_response_with_extensions, encode_stream_with_extensions,
};

use crate::sse::frame_stream;

type BoxError = Box<dyn Error + Send + Sync>;

/// Encodes a libsy response into the endpoint's wire format, reporting
/// `served_model` as the response model so the body names the model that
/// answered rather than the route the caller addressed. `safeguards` answers
/// a Claude Code `safeguards` request on the reply.
pub(crate) async fn into_http_response(
    response: AlgorithmResponse,
    target_format: WireFormat,
    served_model: Option<String>,
    request_extensions: ProviderExtensions,
    safeguards: Option<crate::safeguards::Answer>,
    redactor: Arc<crate::redaction::Redactor>,
) -> Result<HttpResponse, BoxError> {
    match response.llm_response {
        LlmResponse::Agg(response) => {
            let mut body = encode_aggregated_response_with_extensions(
                &response,
                target_format,
                served_model.as_deref(),
                &request_extensions,
            )?;
            if let Some(answer) = safeguards {
                crate::safeguards::add_to_message(&mut body, answer).await;
            }
            Ok(Json(body).into_response())
        }
        LlmResponse::Stream(stream) => {
            let mut events = encode_stream_with_extensions(
                stream,
                target_format,
                served_model,
                &request_extensions,
            )?;
            if let Some(answer) = safeguards {
                events = crate::safeguards::add_to_stream(events, answer);
            }
            Ok(frame_stream(events, target_format, redactor).into_response())
        }
    }
}
