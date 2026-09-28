// SPDX-License-Identifier: Apache-2.0

//! Authenticates API keys, enforces plan quotas, and meters usage in front of the
//! Switchyard router.
//!
//! Every request except `/health` needs an active key. A trusted forwarder key (Open WebUI)
//! acts for the user named in `X-OpenWebUI-User-Email` (or bills its own key when that
//! user has no portal account). Inference requests are checked
//! against the user's quotas and carry a [`UsageSink`] that records their tokens once
//! the response ends. Caller credentials are removed before the router sees the request.

pub mod keys;
pub mod quota;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::{ConnectInfo, Request, State};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use http::{HeaderMap, HeaderValue, StatusCode, header};
use redis::aio::ConnectionManager;
use serde_json::json;
use switchyard_server::{UsageReport, UsageSink};

use crate::keys::{Identity, KeyStore};
use crate::quota::UsageEvent;

const FORWARDED_EMAIL: &str = "x-openwebui-user-email";
const FORWARDED_CHAT: &str = "x-openwebui-chat-id";
/// Rough bytes per token, used only to estimate the prompt of a stream that ended early.
const BYTES_PER_TOKEN: u64 = 4;

pub struct Gate {
    pub keys: KeyStore,
    pub valkey: ConnectionManager,
}

/// Wraps `router` so every request passes through the gate.
pub fn layer(router: Router, gate: Arc<Gate>) -> Router {
    router.layer(middleware::from_fn_with_state(gate, check))
}

#[derive(Clone, Copy, PartialEq)]
enum Access {
    Open,
    User,
    Metered,
    Admin,
}

fn access(path: &str) -> Access {
    match path {
        "/health" => Access::Open,
        "/v1/messages" | "/v1/chat/completions" | "/v1/responses" => Access::Metered,
        "/metrics"
        | "/v1/stats"
        | "/v1/stats/reset"
        | "/v1/upstreams"
        | "/v1/decision"
        | "/v1/routing/session-stats" => Access::Admin,
        _ => Access::User,
    }
}

async fn check(State(gate): State<Arc<Gate>>, mut request: Request, next: Next) -> Response {
    let path = request.uri().path().to_string();
    let access = access(&path);
    if access == Access::Open || (access == Access::Admin && local_read_only(&request)) {
        return next.run(request).await;
    }
    let anthropic = path.starts_with("/v1/messages");
    let refuse = |status, kind, message: &str| error(status, kind, message, anthropic);

    let Some(key) = caller_key(request.headers()) else {
        return refuse(StatusCode::UNAUTHORIZED, Kind::Auth, "missing API key");
    };
    let identity = match gate.keys.by_key(&key).await {
        Ok(Some(identity)) => identity,
        Ok(None) => return refuse(StatusCode::UNAUTHORIZED, Kind::Auth, "invalid API key"),
        Err(e) => {
            tracing::error!(error = %e, "gate key lookup failed");
            return refuse(
                StatusCode::SERVICE_UNAVAILABLE,
                Kind::Unavailable,
                "authentication unavailable",
            );
        }
    };
    let (identity, via) = match forwarded_email(request.headers(), &identity) {
        None => (identity, "key"),
        Some(email) => match gate.keys.by_email(&email).await {
            Ok(Some(user)) => (user, "openwebui"),
            // A forwarder's user with no portal account (for example an old local login)
            // is billed to the forwarder's own key rather than refused.
            Ok(None) => {
                tracing::info!("forwarded user has no portal account; billing the forwarder key");
                (identity, "key")
            }
            Err(e) => {
                tracing::error!(error = %e, "gate user lookup failed");
                return refuse(
                    StatusCode::SERVICE_UNAVAILABLE,
                    Kind::Unavailable,
                    "authentication unavailable",
                );
            }
        },
    };
    if !identity.active {
        return refuse(StatusCode::FORBIDDEN, Kind::Permission, "account disabled");
    }
    if identity.role == "pending" {
        return refuse(
            StatusCode::FORBIDDEN,
            Kind::Permission,
            "account pending approval",
        );
    }
    if access == Access::Admin && identity.role != "admin" {
        return refuse(
            StatusCode::FORBIDDEN,
            Kind::Permission,
            "admin access required",
        );
    }

    if access == Access::Metered {
        let now = quota::now();
        let mut valkey = gate.valkey.clone();
        match quota::admit(&mut valkey, &identity.user_id, &identity.limits, now).await {
            Ok(Ok(())) => {}
            Ok(Err(refusal)) => {
                let mut response = refuse(
                    StatusCode::TOO_MANY_REQUESTS,
                    Kind::RateLimit,
                    refusal.describe(),
                );
                response.headers_mut().insert(
                    header::RETRY_AFTER,
                    HeaderValue::from(refusal.retry_after(now)),
                );
                return response;
            }
            Err(e) => {
                tracing::error!(error = %e, "gate quota check failed");
                return refuse(
                    StatusCode::SERVICE_UNAVAILABLE,
                    Kind::Unavailable,
                    "metering unavailable",
                );
            }
        }
        let sink = usage_sink(&gate, &identity, via, request.headers());
        request.extensions_mut().insert(sink);
    }

    if path == "/v1/models" {
        // Callers only see the chat models they can use right now: no search, embeddings
        // or rerank backends, and nothing whose upstream is down.
        *request.uri_mut() = http::Uri::from_static("/v1/models?available=true");
    }
    strip_caller_headers(request.headers_mut());
    next.run(request).await
}

/// Monitoring endpoints a dashboard in the same container (pulsar-gui) may read without a key.
const LOCAL_READ_ONLY: [&str; 4] = [
    "/metrics",
    "/v1/stats",
    "/v1/upstreams",
    "/v1/routing/session-stats",
];

/// A GET of a read-only monitoring endpoint from this host. Requests without a known peer
/// address are treated as remote.
fn local_read_only(request: &Request) -> bool {
    request.method() == http::Method::GET
        && LOCAL_READ_ONLY.contains(&request.uri().path())
        && request
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .is_some_and(|ConnectInfo(peer)| peer.ip().is_loopback())
}

fn caller_key(headers: &HeaderMap) -> Option<String> {
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let x_api_key = headers.get("x-api-key").and_then(|v| v.to_str().ok());
    bearer
        .or(x_api_key)
        .map(str::trim)
        .filter(|k| !k.is_empty())
        .map(str::to_string)
}

/// The forwarded user, only honoured for a trusted forwarder key.
fn forwarded_email(headers: &HeaderMap, identity: &Identity) -> Option<String> {
    if !identity.trusted_forwarder {
        return None;
    }
    headers
        .get(FORWARDED_EMAIL)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|e| !e.is_empty())
        .map(str::to_string)
}

fn strip_caller_headers(headers: &mut HeaderMap) {
    headers.remove(header::AUTHORIZATION);
    headers.remove("x-api-key");
    let forwarded: Vec<_> = headers
        .keys()
        .filter(|name| name.as_str().starts_with("x-openwebui-"))
        .cloned()
        .collect();
    for name in forwarded {
        headers.remove(name);
    }
}

fn usage_sink(
    gate: &Arc<Gate>,
    identity: &Identity,
    via: &'static str,
    headers: &HeaderMap,
) -> UsageSink {
    let gate = Arc::clone(gate);
    let user_id = identity.user_id.clone();
    // A forwarded request is billed to the user, not to the forwarder's key.
    let api_key_id = if via == "key" {
        identity.key_id.clone()
    } else {
        None
    };
    let chat_id = headers
        .get(FORWARDED_CHAT)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let body_bytes = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let weights = identity.weights.clone();
    UsageSink::new(move |report| {
        let mut event = usage_event(
            &report,
            &user_id,
            api_key_id.clone(),
            via,
            chat_id.clone(),
            body_bytes,
        );
        event.billable_tokens = event.billable(&weights);
        let mut valkey = gate.valkey.clone();
        tokio::spawn(async move {
            if let Err(e) = quota::record(&mut valkey, &event).await {
                // Lost here means unbilled; the log line keeps it recoverable by hand.
                tracing::error!(error = %e, ?event, "gate failed to record usage");
            }
        });
    })
}

/// Turns a usage report into a billable event. A stream that stopped early usually has
/// no provider usage, so its prompt is estimated from the request size and its output
/// from the deltas already sent.
fn usage_event(
    report: &UsageReport,
    user_id: &str,
    api_key_id: Option<String>,
    via: &'static str,
    chat_id: Option<String>,
    body_bytes: u64,
) -> UsageEvent {
    let usage = report.usage.clone().unwrap_or_default();
    let estimated =
        !report.complete && (usage.input_tokens.is_none() || usage.output_tokens.is_none());
    let input_tokens = usage.input_tokens.unwrap_or(if estimated {
        body_bytes / BYTES_PER_TOKEN
    } else {
        0
    });
    let output_tokens = if report.complete {
        usage.output_tokens.unwrap_or(0)
    } else {
        usage.output_tokens.unwrap_or(0).max(report.output_deltas)
    };
    UsageEvent {
        ts: quota::now(),
        user_id: user_id.to_string(),
        api_key_id,
        via,
        model: report.model.clone(),
        input_tokens,
        output_tokens,
        cached_tokens: usage.cached_input_tokens().unwrap_or(0),
        cache_creation_tokens: usage.cache_creation_input_tokens().unwrap_or(0),
        reasoning_tokens: usage.reasoning_tokens.unwrap_or(0),
        // Set by the caller, which knows the account's weights.
        billable_tokens: 0,
        latency_ms: report.latency.as_millis() as u64,
        complete: report.complete,
        estimated,
        chat_id,
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Auth,
    Permission,
    RateLimit,
    Unavailable,
}

/// An error in the caller's wire format, so Claude Code and OpenAI clients show the message.
fn error(status: StatusCode, kind: Kind, message: &str, anthropic: bool) -> Response {
    let body = if anthropic {
        let kind = match kind {
            Kind::Auth => "authentication_error",
            Kind::Permission => "permission_error",
            Kind::RateLimit => "rate_limit_error",
            Kind::Unavailable => "api_error",
        };
        json!({"type": "error", "error": {"type": kind, "message": message}})
    } else {
        let (kind, code) = match kind {
            Kind::Auth => ("invalid_request_error", "invalid_api_key"),
            Kind::Permission => ("invalid_request_error", "permission_denied"),
            Kind::RateLimit => ("rate_limit_error", "rate_limit_exceeded"),
            Kind::Unavailable => ("server_error", "service_unavailable"),
        };
        json!({"error": {"message": message, "type": kind, "code": code}})
    };
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        Body::from(body.to_string()),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use switchyard_protocol::Usage;

    use super::*;

    fn identity(trusted_forwarder: bool) -> Identity {
        Identity {
            key_id: Some("k".to_string()),
            user_id: "u".to_string(),
            role: "user".to_string(),
            active: true,
            trusted_forwarder,
            limits: Default::default(),
            weights: Default::default(),
        }
    }

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for (name, value) in pairs {
            headers.insert(*name, HeaderValue::from_str(value).unwrap());
        }
        headers
    }

    fn report(complete: bool, usage: Option<Usage>, output_deltas: u64) -> UsageReport {
        UsageReport {
            model: "m".to_string(),
            usage,
            output_deltas,
            latency: Duration::from_millis(5),
            complete,
        }
    }

    #[test]
    fn paths_map_to_access_levels() {
        assert!(access("/health") == Access::Open);
        assert!(access("/v1/messages") == Access::Metered);
        assert!(access("/v1/stats/reset") == Access::Admin);
        assert!(access("/v1/models") == Access::User);
        assert!(access("/anything/else") == Access::User);
    }

    fn request_from(method: &str, path: &str, peer: Option<&str>) -> Request {
        let mut request = Request::builder()
            .method(method)
            .uri(path)
            .body(Body::empty())
            .unwrap();
        if let Some(peer) = peer {
            request
                .extensions_mut()
                .insert(ConnectInfo(peer.parse::<SocketAddr>().unwrap()));
        }
        request
    }

    #[test]
    fn only_local_gets_of_monitoring_endpoints_skip_the_key() {
        assert!(local_read_only(&request_from(
            "GET",
            "/metrics",
            Some("127.0.0.1:5000")
        )));
        assert!(local_read_only(&request_from(
            "GET",
            "/v1/stats",
            Some("[::1]:5000")
        )));
        assert!(!local_read_only(&request_from(
            "GET",
            "/metrics",
            Some("10.20.10.254:5000")
        )));
        assert!(!local_read_only(&request_from(
            "POST",
            "/v1/stats/reset",
            Some("127.0.0.1:5000")
        )));
        assert!(!local_read_only(&request_from(
            "GET",
            "/v1/decision",
            Some("127.0.0.1:5000")
        )));
        assert!(!local_read_only(&request_from("GET", "/metrics", None)));
    }

    #[test]
    fn caller_key_reads_bearer_or_x_api_key() {
        let key = |pairs| caller_key(&headers(pairs));
        assert_eq!(
            key(&[("authorization", "Bearer sk_a")]).as_deref(),
            Some("sk_a")
        );
        assert_eq!(key(&[("x-api-key", "sk_b")]).as_deref(), Some("sk_b"));
        assert_eq!(key(&[("authorization", "Bearer ")]), None);
        assert_eq!(key(&[("authorization", "Basic abc")]), None);
    }

    #[test]
    fn forwarded_user_is_honoured_only_for_trusted_forwarders() {
        let h = headers(&[(FORWARDED_EMAIL, "a@example.com")]);
        assert_eq!(
            forwarded_email(&h, &identity(true)).as_deref(),
            Some("a@example.com")
        );
        assert_eq!(forwarded_email(&h, &identity(false)), None);
    }

    #[test]
    fn caller_credentials_and_forwarded_headers_are_removed() {
        let mut h = headers(&[
            ("authorization", "Bearer sk"),
            ("x-api-key", "sk"),
            (FORWARDED_EMAIL, "a@example.com"),
            ("x-openwebui-user-name", "A"),
            ("anthropic-version", "2023-06-01"),
        ]);
        strip_caller_headers(&mut h);
        assert_eq!(h.len(), 1);
        assert!(h.contains_key("anthropic-version"));
    }

    #[test]
    fn complete_response_uses_provider_usage() {
        let usage = Usage {
            input_tokens: Some(100),
            output_tokens: Some(20),
            ..Usage::default()
        };
        let event = usage_event(&report(true, Some(usage), 7), "u", None, "key", None, 4000);
        assert_eq!(
            (event.input_tokens, event.output_tokens, event.estimated),
            (100, 20, false)
        );
    }

    #[test]
    fn stream_stopped_early_is_estimated() {
        let event = usage_event(&report(false, None, 7), "u", None, "key", None, 4000);
        assert_eq!(
            (event.input_tokens, event.output_tokens, event.estimated),
            (1000, 7, true)
        );
        assert!(!event.complete);
    }

    #[test]
    fn errors_use_the_caller_wire_format() {
        let anthropic = error(StatusCode::UNAUTHORIZED, Kind::Auth, "no", true);
        let openai = error(
            StatusCode::TOO_MANY_REQUESTS,
            Kind::RateLimit,
            "slow",
            false,
        );
        assert_eq!(anthropic.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(openai.status(), StatusCode::TOO_MANY_REQUESTS);
    }
}
