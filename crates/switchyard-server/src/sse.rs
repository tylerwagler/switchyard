// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SSE framing helpers for OpenAI, Anthropic, and Responses endpoints.

use std::convert::Infallible;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes, to_bytes};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::response::Response;
use axum::response::sse::{Event, KeepAlive, KeepAliveStream, Sse};
use futures_util::{Stream, StreamExt};
use serde_json::{Value, json};
use switchyard_runner::stream_error_summary;
use switchyard_translation::{LlmStreamError, RawEventStream, WireFormat};

use crate::redaction::Redactor;

/// Boxed stream type accepted by Axum's SSE response wrapper.
pub(crate) type SseFrameStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>;

/// How long a stream may stay silent before a keep-alive frame is sent.
///
/// Clients abort a stream that sends no bytes for a while; Claude Code waits five
/// minutes. A long prefill can be silent that long, and a translated upstream's own
/// keep-alives (pings or SSE comments) do not survive translation.
const KEEP_ALIVE_INTERVAL: Duration = Duration::from_secs(15);

/// How long a streaming request may wait for its upstream before its response starts.
///
/// Cloudflare drops a request whose first byte takes 100 s. A queued upstream or the
/// client's own retries can take that long before any upstream headers arrive.
const START_DELAY: Duration = Duration::from_secs(10);

/// Returns `response` unchanged when it is ready within [`START_DELAY`].
///
/// Otherwise starts an SSE response with status 200, sends keep-alive frames until
/// `response` is ready, and then sends its SSE body. A late error response becomes
/// an SSE error event, since its status can no longer be sent.
pub(crate) async fn start_when_slow(
    response: impl Future<Output = Response> + Send + 'static,
    target_format: WireFormat,
) -> Response {
    start_after(response, target_format, START_DELAY, KEEP_ALIVE_INTERVAL).await
}

async fn start_after(
    response: impl Future<Output = Response> + Send + 'static,
    target_format: WireFormat,
    delay: Duration,
    interval: Duration,
) -> Response {
    let mut response = Box::pin(response);
    tokio::select! {
        response = &mut response => return response,
        () = tokio::time::sleep(delay) => {}
    }
    let ping = Bytes::from_static(if target_format == WireFormat::AnthropicMessages {
        b"event: ping\ndata: {\"type\":\"ping\"}\n\n"
    } else {
        b":\n\n"
    });
    let body = async_stream::stream! {
        yield Ok::<_, axum::Error>(ping.clone());
        let mut ticks = tokio::time::interval(interval);
        ticks.tick().await;
        let response = loop {
            tokio::select! {
                response = &mut response => break response,
                _ = ticks.tick() => yield Ok(ping.clone()),
            }
        };
        if response.status().is_success() {
            let mut body = response.into_body().into_data_stream();
            while let Some(chunk) = body.next().await {
                yield chunk;
            }
        } else {
            yield Ok(late_error_event(response, target_format).await);
        }
    };
    Response::builder()
        .header(CONTENT_TYPE, "text/event-stream")
        .header(CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body))
        .unwrap_or_default()
}

/// Renders an error response that arrived after the SSE response started.
async fn late_error_event(response: Response, target_format: WireFormat) -> Bytes {
    let status = response.status();
    tracing::warn!(
        status = status.as_u16(),
        "upstream failed after the stream started"
    );
    let response = crate::render_error_response(response, target_format);
    let body = to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap_or_default();
    let mut value = serde_json::from_slice::<Value>(&body)
        .ok()
        .filter(|value| value["error"].is_object())
        .unwrap_or_else(|| {
            json!({"error": {
                "type": crate::anthropic_error_type(status),
                "message": String::from_utf8_lossy(&body),
            }})
        });
    let frame = match target_format {
        WireFormat::AnthropicMessages => {
            value["type"] = json!("error");
            // Claude Code retries an error event only when its type is `overloaded_error`.
            if matches!(status.as_u16(), 429 | 503 | 529) {
                value["error"]["type"] = json!("overloaded_error");
            }
            format!("event: error\ndata: {value}\n\n")
        }
        WireFormat::OpenAiResponses => format!("event: error\ndata: {value}\n\n"),
        WireFormat::OpenAiChat => format!("data: {value}\n\n"),
    };
    Bytes::from(frame)
}

/// Converts translated JSON events into endpoint-specific SSE frames, and sends a
/// keep-alive frame whenever the stream is silent for [`KEEP_ALIVE_INTERVAL`].
pub(crate) fn frame_stream(
    stream: RawEventStream,
    target_format: WireFormat,
    redactor: Arc<Redactor>,
) -> Sse<KeepAliveStream<SseFrameStream>> {
    frame_events(stream, target_format, redactor)
        .keep_alive(keep_alive(target_format, KEEP_ALIVE_INTERVAL))
}

/// The keep-alive frame for a format: the Anthropic `ping` event, which Anthropic
/// clients skip, or an SSE comment, which OpenAI clients skip.
fn keep_alive(target_format: WireFormat, interval: Duration) -> KeepAlive {
    let keep_alive = KeepAlive::new().interval(interval);
    if target_format == WireFormat::AnthropicMessages {
        keep_alive.event(Event::default().event("ping").data(r#"{"type":"ping"}"#))
    } else {
        keep_alive
    }
}

/// Converts translated JSON events into endpoint-specific SSE frames.
fn frame_events(
    stream: RawEventStream,
    target_format: WireFormat,
    redactor: Arc<Redactor>,
) -> Sse<SseFrameStream> {
    let framed = async_stream::stream! {
        let mut stream = stream;
        let mut failed = false;
        while let Some(item) = futures_util::StreamExt::next(&mut stream).await {
            let event = match item {
                Ok(value) => match frame_event(target_format, value, &redactor) {
                    Ok(event) => event,
                    Err(error) => {
                        failed = true;
                        error_event(target_format, error.to_string(), &redactor)
                    }
                },
                // Preserve the upstream error's fields, apart from credential redaction,
                // rather than replacing it with a synthesized error.
                Err(LlmStreamError::Upstream(value)) => {
                    // Log only the event type; the error message can quote request content.
                    let event_type = value.get("type").and_then(Value::as_str).unwrap_or("error");
                    tracing::warn!(event_type, "stream ended with an in-band error event");
                    failed = true;
                    frame_event(target_format, value.clone(), &redactor).unwrap_or_else(|error| {
                        tracing::warn!(error = %error, "in-band error event could not be framed");
                        error_event(target_format, value.to_string(), &redactor)
                    })
                }
                Err(LlmStreamError::Client(error)) => {
                    // The error text can quote request content, so the log
                    // records only the stable error class.
                    let summary = stream_error_summary(&error, None);
                    tracing::warn!(
                        error.kind = summary.kind.as_str(),
                        error.upstream_status = summary.upstream_status,
                        "stream iteration failed"
                    );
                    failed = true;
                    error_event(target_format, error.to_string(), &redactor)
                }
            };
            yield Ok(event);
            if failed {
                break;
            }
        }

        // `[DONE]` is the OpenAI Chat success sentinel: clients stop reading there and
        // keep what they have as a finished answer, so it must not follow a failed turn.
        if !failed && target_format == WireFormat::OpenAiChat {
            yield Ok(Event::default().data("[DONE]"));
        }
    };

    Sse::new(Box::pin(framed) as SseFrameStream)
}

fn frame_event(
    target_format: WireFormat,
    value: Value,
    redactor: &Redactor,
) -> Result<Event, serde_json::Error> {
    let data = redactor.json(serde_json::to_string(&value)?);
    match target_format {
        WireFormat::OpenAiChat => Ok(Event::default().data(data)),
        WireFormat::AnthropicMessages | WireFormat::OpenAiResponses => {
            let event_type = value
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("message")
                .to_string();
            Ok(Event::default().event(event_type).data(data))
        }
    }
}

fn error_event(target_format: WireFormat, message: String, redactor: &Redactor) -> Event {
    match target_format {
        WireFormat::OpenAiChat => Event::default().data(
            redactor.json(
                json!({
                    "error": {
                        "message": message,
                        "type": "SwitchyardError",
                    }
                })
                .to_string(),
            ),
        ),
        WireFormat::AnthropicMessages | WireFormat::OpenAiResponses => {
            let error_type = if target_format == WireFormat::AnthropicMessages {
                "api_error"
            } else {
                "SwitchyardError"
            };
            Event::default().event("error").data(
                redactor.json(
                    json!({
                        "type": "error",
                        "error": {
                            "message": message,
                            "type": error_type,
                        }
                    })
                    .to_string(),
                ),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use futures_util::stream;
    use switchyard_protocol::LlmClientError;
    use tracing_subscriber::layer::SubscriberExt;

    use super::*;

    type TestResult<T = ()> = Result<T, Box<dyn Error + Send + Sync>>;

    // Renders a stream that is silent before its one event, with a short keep-alive.
    async fn silent_body(target_format: WireFormat, event: Value) -> TestResult<String> {
        let stream: RawEventStream = Box::pin(stream::once(async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            Ok(event)
        }));
        let response = frame_events(stream, target_format, Arc::new(Redactor::default()))
            .keep_alive(keep_alive(target_format, Duration::from_millis(20)))
            .into_response();
        Ok(String::from_utf8(
            to_bytes(response.into_body(), usize::MAX).await?.to_vec(),
        )?)
    }

    // A silent Anthropic stream sends `ping` events, and the real event still follows.
    #[tokio::test]
    async fn silent_anthropic_stream_sends_pings() -> TestResult {
        let body = silent_body(
            WireFormat::AnthropicMessages,
            json!({"type": "message_stop"}),
        )
        .await?;
        assert!(
            body.starts_with("event: ping\ndata: {\"type\":\"ping\"}\n\n"),
            "{body}"
        );
        assert!(
            body.trim_end()
                .ends_with("event: message_stop\ndata: {\"type\":\"message_stop\"}"),
            "{body}"
        );
        Ok(())
    }

    // A silent Chat stream sends SSE comments, which Chat clients skip.
    #[tokio::test]
    async fn silent_chat_stream_sends_comments() -> TestResult {
        let body = silent_body(WireFormat::OpenAiChat, json!({"choices": []})).await?;
        assert!(body.lines().any(|line| line.starts_with(':')), "{body}");
        assert!(!body.contains("ping"), "{body}");
        Ok(())
    }

    // Renders a framed body for one Chat stream.
    async fn chat_body(items: Vec<Result<Value, LlmStreamError>>) -> TestResult<String> {
        let stream: RawEventStream = Box::pin(stream::iter(items));
        let response = frame_stream(
            stream,
            WireFormat::OpenAiChat,
            Arc::new(Redactor::default()),
        )
        .into_response();
        Ok(String::from_utf8(
            to_bytes(response.into_body(), usize::MAX).await?.to_vec(),
        )?)
    }

    #[tokio::test]
    async fn stream_error_terminates_without_done_marker() -> TestResult {
        let failure = LlmClientError::General("boom".to_string());
        let body = chat_body(vec![
            Ok(json!({"id": "before"})),
            Err(LlmStreamError::Client(failure)),
            Ok(json!({"id": "after"})),
        ])
        .await?;

        // A stream error is terminal: later chunks and success markers must not be emitted.
        assert!(body.contains("before"));
        assert!(body.contains("boom"));
        assert!(!body.contains("after"));
        assert!(!body.contains("[DONE]"));
        Ok(())
    }

    #[tokio::test]
    async fn in_band_error_is_forwarded_verbatim_without_done_marker() -> TestResult {
        let upstream_error = json!({
            "error": {"code": "stream_failed", "message": "boom", "type": "upstream_stream_error"}
        });
        let body = chat_body(vec![
            Ok(json!({"id": "before"})),
            Err(LlmStreamError::Upstream(upstream_error)),
            Ok(json!({"id": "after"})),
        ])
        .await?;

        // The upstream owns this error, so its code and type reach the client unchanged
        // rather than being flattened into a synthesized SwitchyardError frame.
        assert!(body.contains("before"));
        assert!(body.contains("stream_failed"));
        assert!(body.contains("upstream_stream_error"));
        assert!(!body.contains("SwitchyardError"));
        assert!(!body.contains("after"));
        assert!(!body.contains("[DONE]"));
        Ok(())
    }

    // Collects rendered warn events so the test can assert against the final
    // log sink rather than a field mid-pipeline.
    #[derive(Clone, Default)]
    struct WarnCapture(Arc<std::sync::Mutex<Vec<String>>>);

    impl tracing_subscriber::Layer<tracing_subscriber::Registry> for WarnCapture {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, tracing_subscriber::Registry>,
        ) {
            if *event.metadata().level() != tracing::Level::WARN {
                return;
            }
            let mut fields = String::new();
            event.record(
                &mut |field: &tracing::field::Field, value: &dyn std::fmt::Debug| {
                    fields.push_str(&format!("{}={value:?} ", field.name()));
                },
            );
            self.0.lock().unwrap().push(fields);
        }
    }

    // The client error text can quote request content, so the stream-failure
    // warn log records only the stable error class.
    #[test]
    fn stream_client_error_warn_redacts_upstream_body() -> TestResult {
        const LEAKED: &str = "SECRET-quoted-request-content";
        let capture = WarnCapture::default();
        let subscriber = tracing_subscriber::registry().with(capture.clone());
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        let body = crate::testing::with_subscriber(subscriber, || {
            runtime.block_on(chat_body(vec![Err(LlmStreamError::Client(
                LlmClientError::UpstreamHttp {
                    status: axum::http::StatusCode::BAD_GATEWAY,
                    body: format!("upstream failed: {LEAKED}"),
                    headers: Box::new(http::HeaderMap::new()),
                },
            ))]))
        })?;

        // The client still sees the error text in-band.
        assert!(body.contains(LEAKED), "{body}");

        let events = capture.0.lock().unwrap().clone();
        assert!(
            events
                .iter()
                .any(|event| event.contains("stream iteration failed")),
            "{events:?}"
        );
        assert!(
            !events.iter().any(|event| event.contains(LEAKED)),
            "{events:?}"
        );
        Ok(())
    }

    // An SSE response that is ready after `wait`.
    async fn ready_after(wait: Duration, response: Response) -> Response {
        tokio::time::sleep(wait).await;
        response
    }

    fn sse_response(body: &'static str) -> Response {
        Response::builder()
            .header(CONTENT_TYPE, "text/event-stream")
            .body(Body::from(body))
            .unwrap()
    }

    async fn started(response: Response) -> TestResult<(StatusCode, String, String)> {
        let start = start_after(
            ready_after(Duration::from_millis(150), response),
            WireFormat::AnthropicMessages,
            Duration::from_millis(20),
            Duration::from_millis(30),
        )
        .await;
        let status = start.status();
        let content_type = start.headers()[CONTENT_TYPE].to_str()?.to_string();
        let body = String::from_utf8(to_bytes(start.into_body(), usize::MAX).await?.to_vec())?;
        Ok((status, content_type, body))
    }

    // A slow upstream gets pings at once, then its own events.
    #[tokio::test]
    async fn slow_stream_starts_with_pings() -> TestResult {
        let (status, content_type, body) = started(sse_response(
            "event: message_start\ndata: {\"type\":\"message_start\"}\n\n",
        ))
        .await?;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "text/event-stream");
        let pings = body.matches("event: ping\n").count();
        assert!(pings >= 2, "{body}");
        assert!(
            body.ends_with("event: message_start\ndata: {\"type\":\"message_start\"}\n\n"),
            "{body}"
        );
        Ok(())
    }

    fn error_data(body: &str) -> TestResult<Value> {
        let data = body
            .split("event: error\ndata: ")
            .nth(1)
            .ok_or("missing error event")?;
        Ok(serde_json::from_str(data.trim_end())?)
    }

    // A slow upstream that fails with 503 ends with an `overloaded_error` event.
    #[tokio::test]
    async fn slow_unavailable_upstream_sends_overloaded_error_event() -> TestResult {
        let (status, _, body) =
            started(crate::client_error(&LlmClientError::TemporarilyUnavailable)).await?;
        assert_eq!(status, StatusCode::OK);
        assert!(body.starts_with("event: ping\n"), "{body}");
        let error = error_data(&body)?;
        assert_eq!(error["type"], "error");
        assert_eq!(error["error"]["type"], "overloaded_error");
        Ok(())
    }

    // A late context overflow keeps the token that starts Claude Code's compaction.
    #[tokio::test]
    async fn slow_context_overflow_keeps_prompt_too_long_token() -> TestResult {
        let (_, _, body) = started(crate::client_error(
            &LlmClientError::ContextWindowExceeded {
                model: switchyard_protocol::ModelId::new("m"),
                message: "too long".to_string(),
            },
        ))
        .await?;
        let error = error_data(&body)?;
        assert_eq!(error["error"]["type"], "invalid_request_error");
        assert_eq!(
            error["error"]["message"],
            "capability_rejected: prompt_too_long: too long"
        );
        Ok(())
    }

    // An upstream error that arrives before the delay keeps its HTTP status.
    #[tokio::test]
    async fn fast_error_keeps_its_status() -> TestResult {
        let response = start_after(
            async { crate::client_error(&LlmClientError::TemporarilyUnavailable) },
            WireFormat::AnthropicMessages,
            Duration::from_millis(100),
            Duration::from_millis(30),
        )
        .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        Ok(())
    }

    #[tokio::test]
    async fn anthropic_stream_error_uses_api_error_type() -> TestResult {
        let failure = LlmClientError::General("boom".to_string());
        let stream: RawEventStream =
            Box::pin(stream::iter(vec![Err(LlmStreamError::Client(failure))]));
        let response = frame_stream(
            stream,
            WireFormat::AnthropicMessages,
            Arc::new(Redactor::default()),
        )
        .into_response();
        let body = String::from_utf8(to_bytes(response.into_body(), usize::MAX).await?.to_vec())?;
        let data = body
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .ok_or("missing data line")?;
        assert_eq!(
            serde_json::from_str::<Value>(data)?,
            json!({"type": "error", "error": {"type": "api_error", "message": "boom"}})
        );
        Ok(())
    }
}
