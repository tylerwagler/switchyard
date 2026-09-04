// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transparent relay for non-chat backends: `/v1/embeddings` and `/v1/rerank`
//! proxy to the configured `[embeddings.*]` / `[rerank.*]` backends. The default
//! route picks the first configured backend; an optional `/{name}` path segment
//! selects a named one. Bodies pass through as-is — these are OpenAI/Cohere-
//! shaped non-chat calls, not chat translation.

use std::time::Instant;

use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use opentelemetry::{KeyValue, global};
use serde_json::{Value, json};

use crate::ServerState;

const EMBEDDINGS_PATH: &str = "/embeddings";
const RERANK_PATH: &str = "/rerank";

pub(crate) async fn embeddings_default(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    relay_kind(&state, "embeddings", EMBEDDINGS_PATH, None, &headers, body).await
}

pub(crate) async fn embeddings_named(
    State(state): State<ServerState>,
    Path(name): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    relay_kind(&state, "embeddings", EMBEDDINGS_PATH, Some(&name), &headers, body).await
}

pub(crate) async fn rerank_default(
    State(state): State<ServerState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    relay_kind(&state, "rerank", RERANK_PATH, None, &headers, body).await
}

pub(crate) async fn rerank_named(
    State(state): State<ServerState>,
    Path(name): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    relay_kind(&state, "rerank", RERANK_PATH, Some(&name), &headers, body).await
}

/// Picks a named backend, or the first configured when no name is given.
fn pick_embeddings(
    state: &ServerState,
    name: Option<&str>,
) -> Option<(String, String, Option<String>)> {
    match name {
        Some(name) => state
            .embeddings()
            .get(name)
            .map(|config| (name.to_string(), config.base_url.clone(), config.api_key_env.clone())),
        None => state
            .embeddings()
            .iter()
            .next()
            .map(|(name, config)| (name.clone(), config.base_url.clone(), config.api_key_env.clone())),
    }
}

fn pick_rerank(state: &ServerState, name: Option<&str>) -> Option<(String, String)> {
    match name {
        Some(name) => state
            .rerank()
            .get(name)
            .map(|config| (name.to_string(), config.base_url.clone())),
        None => state
            .rerank()
            .iter()
            .next()
            .map(|(name, config)| (name.clone(), config.base_url.clone())),
    }
}

async fn relay_kind(
    state: &ServerState,
    kind: &str,
    path: &str,
    name: Option<&str>,
    headers: &HeaderMap,
    body: Bytes,
) -> Response {
    let (selected_name, base_url, api_key_env) = match kind {
        "embeddings" => match pick_embeddings(state, name) {
            Some(selected) => selected,
            None => return missing_backend(kind, name),
        },
        "rerank" => match pick_rerank(state, name) {
            Some((selected_name, base_url)) => (selected_name, base_url, None),
            None => return missing_backend(kind, name),
        },
        _ => unreachable!("relay kind is fixed at the call site"),
    };

    let started = Instant::now();
    match relay_request(state.http_client(), &base_url, path, api_key_env.as_deref(), headers, body)
        .await
    {
        Ok((status, content_type, bytes)) => {
            record(kind, &selected_name, "ok", started);
            let mut builder = Response::builder().status(status);
            if let Some(content_type) = content_type {
                builder = builder.header("content-type", content_type);
            }
            builder.body(Body::from(bytes)).unwrap_or_else(|_| internal_error())
        }
        Err(error) => {
            tracing::warn!(kind, %error, "aux relay failed");
            record(kind, &selected_name, "error", started);
            json_error(
                StatusCode::BAD_GATEWAY,
                format!("{kind} relay to {selected_name} failed: {error}"),
            )
        }
    }
}

async fn relay_request(
    client: &reqwest::Client,
    base_url: &str,
    path: &str,
    api_key_env: Option<&str>,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<(StatusCode, Option<String>, Bytes), String> {
    let url = format!("{}{}", base_url.trim_end_matches('/'), path);
    let mut request = client.post(&url).body(body);
    if let Some(content_type) = headers
        .get("content-type")
        .and_then(|value| value.to_str().ok())
    {
        request = request.header("content-type", content_type);
    }
    if let Some(key_env) = api_key_env {
        if let Ok(token) = std::env::var(key_env) {
            if !token.trim().is_empty() {
                request = request.header("authorization", format!("Bearer {token}"));
            }
        }
    }
    let response = request.send().await.map_err(|error| error.to_string())?;
    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let bytes = response.bytes().await.map_err(|error| error.to_string())?;
    Ok((status, content_type, bytes))
}

/// Capability entries appended to `/v1/models` so the listing is truthful about
/// non-chat backends as well as chat routes.
pub(crate) fn capability_entries(state: &ServerState) -> Vec<Value> {
    let mut entries = Vec::new();
    for (name, config) in state.embeddings() {
        entries.push(json!({
            "id": name,
            "object": "model",
            "type": "model",
            "kind": "embeddings",
            "served_by": config.base_url,
            "created": 0,
            "owned_by": "switchyard",
            "display_name": name,
            "capabilities": { "streaming": false, "tool_calling": false, "context_window": null },
        }));
    }
    for (name, config) in state.rerank() {
        entries.push(json!({
            "id": name,
            "object": "model",
            "type": "model",
            "kind": "rerank",
            "served_by": config.base_url,
            "model": config.model,
            "created": 0,
            "owned_by": "switchyard",
            "display_name": name,
            "capabilities": { "streaming": false, "tool_calling": false, "context_window": null },
        }));
    }
    for (name, config) in state.search() {
        entries.push(json!({
            "id": name,
            "object": "model",
            "type": "model",
            "kind": "search",
            "served_by": config.base_url,
            "created": 0,
            "owned_by": "switchyard",
            "display_name": name,
            "capabilities": { "streaming": false, "tool_calling": false, "context_window": null },
        }));
    }
    entries
}

fn missing_backend(kind: &str, name: Option<&str>) -> Response {
    let message = match name {
        Some(name) => format!("no {kind} backend named {name} is configured"),
        None => format!("no {kind} backend is configured"),
    };
    json_error(StatusCode::NOT_FOUND, message)
}

fn json_error(status: StatusCode, message: String) -> Response {
    (
        status,
        Json(json!({ "error": { "message": message, "type": "auxiliary_error" } })),
    )
        .into_response()
}

fn internal_error() -> Response {
    json_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "failed to build auxiliary response".to_string(),
    )
}

fn record(kind: &str, name: &str, outcome: &str, started: Instant) {
    let meter = global::meter("switchyard");
    meter
        .u64_counter("switchyard.aux_requests_total")
        .build()
        .add(
            1,
            &[
                KeyValue::new("kind", kind.to_string()),
                KeyValue::new("name", name.to_string()),
                KeyValue::new("outcome", outcome.to_string()),
            ],
        );
    meter
        .f64_histogram("switchyard.aux_duration_seconds")
        .build()
        .record(
            started.elapsed().as_secs_f64(),
            &[KeyValue::new("kind", kind.to_string())],
        );
}
