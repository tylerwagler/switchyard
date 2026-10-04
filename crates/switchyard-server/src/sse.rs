// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SSE framing helpers for OpenAI, Anthropic, and Responses endpoints.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::response::sse::{Event, KeepAlive, KeepAliveStream, Sse};
use futures_util::Stream;
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

    use axum::{body::to_bytes, response::IntoResponse};
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

        let body = tracing::subscriber::with_default(subscriber, || {
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
