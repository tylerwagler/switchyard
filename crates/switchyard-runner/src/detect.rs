// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Startup discovery of the model a `detect_model` target's backend serves.
//!
//! Probes the OpenAI-style `GET /v1/models` first, because it names the model and
//! its metadata. Falls back to `GET /health`, which llama.cpp-family servers
//! (pulsar included) answer with the loaded model. A target whose backend cannot
//! be reached keeps its configured id; the caller sees a warning for it.

use std::collections::BTreeMap;
use std::time::Duration;

use reqwest::Client;
use serde_json::Value;
use switchyard_protocol::ModelId;
use tracing::{info, warn};

use crate::config::DetectedTarget;

const DETECT_TIMEOUT: Duration = Duration::from_secs(3);

/// Builds the models URL for a client base URL, which may already end in `/v1`.
fn models_url(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{base}/models")
    } else {
        format!("{base}/v1/models")
    }
}

/// Builds the health URL at the server root.
fn health_url(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    match base.strip_suffix("/v1") {
        Some(root) => format!("{root}/health"),
        None => format!("{base}/health"),
    }
}

async fn fetch_json(client: &Client, url: &str) -> Option<Value> {
    let response = client.get(url).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    response.json().await.ok()
}

/// Reads the first model id from an OpenAI-shaped models list.
fn model_id(body: &Value) -> Option<ModelId> {
    body.get("data")
        .and_then(Value::as_array)
        .and_then(|models| models.first())
        .and_then(|model| model.get("id"))
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(ModelId::from)
}

/// Reads the model name from a health body.
fn health_model(body: &Value) -> Option<ModelId> {
    body.get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.is_empty())
        .map(ModelId::from)
}

async fn probe(client: &Client, base_url: &str) -> Option<ModelId> {
    if let Some(body) = fetch_json(client, &models_url(base_url)).await
        && let Some(id) = model_id(&body)
    {
        return Some(id);
    }
    let body = fetch_json(client, &health_url(base_url)).await?;
    health_model(&body)
}

/// Probes each target's backend once and returns the discovered ids by target name.
///
/// A backend that cannot be reached logs a warning and is left out of the map, so
/// the target keeps the id configured in the deployment file. Probes run one
/// after another; a deployment has only a few detected targets, and startup
/// talks to each backend at most twice.
pub(crate) async fn resolve(targets: &[DetectedTarget]) -> BTreeMap<String, ModelId> {
    let Ok(client) = Client::builder().timeout(DETECT_TIMEOUT).build() else {
        warn!("could not build the model-detection HTTP client; keeping configured target ids");
        return BTreeMap::new();
    };
    let mut resolved = BTreeMap::new();
    for target in targets {
        match probe(&client, &target.base_url).await {
            Some(id) => {
                info!(
                    name = %target.name,
                    model = %id,
                    "detected the model served by the target's backend"
                );
                resolved.insert(target.name.clone(), id);
            }
            None => warn!(
                name = %target.name,
                base_url = %target.base_url,
                fallback = %target.fallback,
                "could not detect the model served by the target's backend; keeping the configured id"
            ),
        }
    }
    resolved
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    #[test]
    fn builds_models_and_health_urls_from_either_base_form() {
        let cases = [
            (
                "http://sparky.defense.lan:8000",
                "http://sparky.defense.lan:8000/v1/models",
            ),
            (
                "http://sparky.defense.lan:8000/",
                "http://sparky.defense.lan:8000/v1/models",
            ),
            (
                "http://embed.defense.lan:8000/v1",
                "http://embed.defense.lan:8000/v1/models",
            ),
            (
                "http://embed.defense.lan:8000/v1/",
                "http://embed.defense.lan:8000/v1/models",
            ),
        ];
        for (base, models) in cases {
            assert_eq!(models_url(base), models, "base: {base}");
        }
        let cases = [
            (
                "http://sparky.defense.lan:8000",
                "http://sparky.defense.lan:8000/health",
            ),
            (
                "http://sparky.defense.lan:8000/",
                "http://sparky.defense.lan:8000/health",
            ),
            (
                "http://embed.defense.lan:8000/v1",
                "http://embed.defense.lan:8000/health",
            ),
            (
                "http://embed.defense.lan:8000/v1/",
                "http://embed.defense.lan:8000/health",
            ),
        ];
        for (base, health) in cases {
            assert_eq!(health_url(base), health, "base: {base}");
        }
    }

    #[tokio::test]
    async fn resolves_the_served_model_from_the_models_list() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "object": "list",
                "data": [{"id": "qwen3.8-flash-next", "object": "model"}]
            })))
            .mount(&server)
            .await;

        let resolved = probe(&Client::new(), &server.uri()).await;
        assert_eq!(resolved, Some(ModelId::from("qwen3.8-flash-next")));
    }

    #[tokio::test]
    async fn falls_back_to_health_when_the_models_list_is_missing() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "status": "ok",
                "model": "qwen3.8-flash-next"
            })))
            .mount(&server)
            .await;

        let resolved = probe(&Client::new(), &server.uri()).await;
        assert_eq!(resolved, Some(ModelId::from("qwen3.8-flash-next")));
    }

    #[tokio::test]
    async fn an_unreachable_backend_resolves_to_nothing() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status": "ok"})))
            .mount(&server)
            .await;

        let resolved = probe(&Client::new(), &server.uri()).await;
        assert_eq!(resolved, None);
    }

    /// A models list without ids and a health body without a model name resolve to
    /// nothing, so `resolve` keeps the target's configured fallback.
    #[tokio::test]
    async fn resolve_leaves_failed_targets_out_of_the_map() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": []})))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/health"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"status": "ok"})))
            .mount(&server)
            .await;

        let targets = vec![DetectedTarget {
            name: "first".to_string(),
            base_url: server.uri(),
            fallback: ModelId::from("model/stale"),
        }];
        let resolved = resolve(&targets).await;
        assert!(resolved.is_empty());
    }
}
