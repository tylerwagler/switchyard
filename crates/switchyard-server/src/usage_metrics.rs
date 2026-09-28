// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Response usage and full-turn latency metrics for routed requests.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use opentelemetry::{KeyValue, global};
use switchyard_protocol::{LlmResponse, LlmResponseChunk, Response, Usage};

use crate::SharedRoutingLog;
use crate::routing_log::RoutingLogContext;
use crate::stats::{StatsAccumulator, TokenUsage};

/// Final usage for one served request, handed to a [`UsageSink`].
#[derive(Clone, Debug)]
pub struct UsageReport {
    /// Model that served the request.
    pub model: String,
    /// Usage the provider reported. `None` when a stream ended before usage arrived.
    pub usage: Option<Usage>,
    /// Streamed text, reasoning, and tool-call deltas seen. Zero for non-streaming responses.
    pub output_deltas: u64,
    /// Time from request arrival to the end of the response.
    pub latency: Duration,
    /// False when the stream stopped early: the client left or the upstream failed.
    pub complete: bool,
}

/// Receives the final usage of each served request.
///
/// Insert one into a request's extensions before it reaches the router to meter that
/// request. Each served request reports exactly once.
#[derive(Clone)]
pub struct UsageSink(Arc<dyn Fn(UsageReport) + Send + Sync>);

impl UsageSink {
    pub fn new(report: impl Fn(UsageReport) + Send + Sync + 'static) -> Self {
        Self(Arc::new(report))
    }
}

/// Tracks one stream for a [`UsageSink`] and reports on drop if the stream never finished.
struct StreamMeter {
    sink: Option<UsageSink>,
    model: String,
    started: Instant,
    usage: Option<Usage>,
    output_deltas: u64,
}

impl StreamMeter {
    fn finish(&mut self, complete: bool) {
        if let Some(sink) = self.sink.take() {
            sink.0(UsageReport {
                model: self.model.clone(),
                usage: self.usage.clone(),
                output_deltas: self.output_deltas,
                latency: self.started.elapsed(),
                complete,
            });
        }
    }
}

impl Drop for StreamMeter {
    // A dropped, unfinished stream means the client disconnected mid-response.
    fn drop(&mut self) {
        self.finish(false);
    }
}

/// Observes a routed response without changing its aggregate or streaming contents.
pub(crate) fn observe(
    response: Response,
    model: &str,
    started: Instant,
    stats: StatsAccumulator,
    cache_eligible: f64,
    routing_log: Option<(SharedRoutingLog, RoutingLogContext)>,
    sink: Option<UsageSink>,
) -> Response {
    let Response {
        llm_response,
        metadata,
        upstream_headers,
    } = response;
    let model = model.to_string();

    let llm_response = match llm_response {
        LlmResponse::Agg(agg) => {
            record_terminal(&stats, &agg.usage, &model, started, cache_eligible);
            if let Some((log, context)) = routing_log {
                log.append(context, &model, None, &agg.usage);
            }
            if let Some(sink) = sink {
                sink.0(UsageReport {
                    model: model.clone(),
                    usage: Some(agg.usage.clone()),
                    output_deltas: 0,
                    latency: started.elapsed(),
                    complete: true,
                });
            }
            LlmResponse::Agg(agg)
        }
        LlmResponse::Stream(mut stream) => {
            let mut meter = StreamMeter {
                sink,
                model: model.clone(),
                started,
                usage: None,
                output_deltas: 0,
            };
            let wrapped = async_stream::stream! {
                let mut latest_usage = None;
                let mut terminal_seen = false;
                let mut recorded = false;
                while let Some(item) = stream.next().await {
                    let failed = match &item {
                        Err(_) => true,
                        Ok(event) => event.normalized().iter().any(|chunk| {
                            matches!(
                                chunk,
                                LlmResponseChunk::StreamError { .. }
                                    | LlmResponseChunk::DecodeError { .. }
                            )
                        }),
                    };
                    if let Ok(event) = &item {
                        for chunk in event.normalized() {
                            match chunk {
                                LlmResponseChunk::Usage(usage) => {
                                    latest_usage = Some(usage.clone());
                                    meter.usage = Some(usage.clone());
                                }
                                LlmResponseChunk::MessageStop { .. } => {
                                    terminal_seen = true;
                                }
                                LlmResponseChunk::TextDelta { .. }
                                | LlmResponseChunk::ReasoningDelta { .. }
                                | LlmResponseChunk::ToolCallDelta { .. } => {
                                    meter.output_deltas += 1;
                                }
                                _ => {}
                            }
                        }
                    }
                    if failed {
                        record_stream_error(&stats, &model);
                    }
                    // Responses clients may stop polling immediately after the terminal event.
                    // Commit first when that event already carries the final usage.
                    if !failed && !recorded && terminal_seen
                        && let Some(usage) = latest_usage.as_ref()
                    {
                        record_terminal(&stats, usage, &model, started, cache_eligible);
                        if let Some((log, context)) = routing_log.as_ref() {
                            log.append(context.clone(), &model, None, usage);
                        }
                        recorded = true;
                        meter.finish(true);
                    }
                    yield item;
                    if failed {
                        return;
                    }
                }
                if !recorded {
                    let usage = latest_usage.unwrap_or_default();
                    record_terminal(&stats, &usage, &model, started, cache_eligible);
                    if let Some((log, context)) = routing_log {
                        log.append(context, &model, None, &usage);
                    }
                }
                meter.finish(true);
            };
            LlmResponse::Stream(Box::pin(wrapped))
        }
    };

    Response {
        llm_response,
        metadata,
        upstream_headers,
    }
}

// Records a terminal stream failure after the routed call was already counted.
fn record_stream_error(stats: &StatsAccumulator, model: &str) {
    stats.record_response_error(model);
}

pub(crate) fn token_usage(usage: &Usage) -> TokenUsage {
    let cached_tokens = usage.cached_input_tokens().unwrap_or(0);
    let cache_creation_tokens = usage.cache_creation_input_tokens().unwrap_or(0);
    TokenUsage {
        prompt_tokens: usage
            .input_tokens
            .unwrap_or(0)
            .saturating_add(cached_tokens)
            .saturating_add(cache_creation_tokens),
        completion_tokens: usage.output_tokens.unwrap_or(0),
        cached_tokens,
        cache_creation_tokens,
        cacheable_prompt_tokens: 0,
        reasoning_tokens: usage.reasoning_tokens.unwrap_or(0),
    }
}

/// Records final usage and latency in both OpenTelemetry metrics and JSON stats.
fn record_terminal(
    stats: &StatsAccumulator,
    usage: &Usage,
    model: &str,
    started: Instant,
    cache_eligible: f64,
) {
    let total_latency = started.elapsed();
    record_usage(usage, model);
    record_latency(model, total_latency);
    let mut token_usage = token_usage(usage);
    token_usage.cacheable_prompt_tokens =
        (token_usage.prompt_tokens as f64 * cache_eligible).round() as u64;
    stats.record_usage(model, token_usage, total_latency.as_secs_f64() * 1_000.0);
}

fn attributes(model: &str) -> [KeyValue; 1] {
    [KeyValue::new("model", model.to_string())]
}

fn record_usage(usage: &Usage, model: &str) {
    let attributes = attributes(model);
    let meter = global::meter("switchyard");
    let cached = usage.cached_input_tokens();
    let cache_creation = usage.cache_creation_input_tokens();

    if usage.input_tokens.is_some() || cached.is_some() || cache_creation.is_some() {
        let prompt =
            usage.input_tokens.unwrap_or(0) + cached.unwrap_or(0) + cache_creation.unwrap_or(0);
        meter
            .u64_counter("switchyard.prompt_tokens")
            .build()
            .add(prompt, &attributes);
    }
    for (name, value) in [
        ("switchyard.completion_tokens", usage.output_tokens),
        ("switchyard.cached_tokens", cached),
        ("switchyard.cache_creation_tokens", cache_creation),
        ("switchyard.reasoning_tokens", usage.reasoning_tokens),
    ] {
        if let Some(value) = value {
            meter.u64_counter(name).build().add(value, &attributes);
        }
    }
}

fn record_latency(model: &str, latency: Duration) {
    global::meter("switchyard")
        .f64_histogram("switchyard.total_latency_ms")
        .build()
        .record(latency.as_secs_f64() * 1000.0, &attributes(model));
}

#[cfg(test)]
mod tests {
    use futures_util::{StreamExt, stream};
    use switchyard_protocol::{LlmResponseChunk, LlmResponseStreamEvent, Metadata, Response};

    use super::*;

    /// An OpenAI Responses client may stop polling immediately after receiving the
    /// terminal `response.completed` event. Switchyard must record usage and routing
    /// data before returning that event because the stream wrapper will not resume
    /// after the client drops it.
    #[tokio::test]
    async fn terminal_event_is_recorded_before_the_client_drops_the_stream() {
        let dir = tempfile::tempdir().expect("temp dir");
        let log = SharedRoutingLog::new(dir.path().join("routing.jsonl")).expect("routing log");
        let context = RoutingLogContext::from_metadata(&Metadata {
            session_id: Some("streaming-session".to_string()),
            ..Metadata::default()
        });
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(3),
            ..Usage::default()
        };
        let source = stream::iter([Ok(LlmResponseStreamEvent::new(vec![
            LlmResponseChunk::Usage(usage),
            LlmResponseChunk::MessageStop { reason: None },
        ]))]);
        let response = Response {
            llm_response: LlmResponse::Stream(Box::pin(source)),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        };
        let stats = StatsAccumulator::default();
        let observed = observe(
            response,
            "model/worker",
            Instant::now(),
            stats.clone(),
            0.0,
            Some((log.clone(), context)),
            None,
        );

        let LlmResponse::Stream(mut observed) = observed.llm_response else {
            panic!("expected stream");
        };
        assert!(observed.next().await.is_some());
        drop(observed);

        let routing = log
            .snapshot_session("streaming-session")
            .expect("read routing log")
            .expect("terminal event was recorded");
        let routing = serde_json::to_value(routing).expect("serialize routing stats");
        assert_eq!(routing["models"]["model/worker"]["calls"], 1);
        assert_eq!(routing["models"]["model/worker"]["prompt_tokens"], 10);
        assert_eq!(routing["models"]["model/worker"]["completion_tokens"], 3);

        let process = stats.snapshot();
        assert_eq!(process.models["model/worker"].prompt_tokens, 10);
        assert_eq!(process.models["model/worker"].completion_tokens, 3);
    }

    fn collecting_sink() -> (UsageSink, Arc<parking_lot::Mutex<Vec<UsageReport>>>) {
        let reports = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let sink_reports = Arc::clone(&reports);
        let sink = UsageSink::new(move |report| sink_reports.lock().push(report));
        (sink, reports)
    }

    fn text(text: &str) -> LlmResponseChunk {
        LlmResponseChunk::TextDelta {
            index: 0,
            text: text.to_string(),
        }
    }

    fn stream_response(events: Vec<Vec<LlmResponseChunk>>) -> Response {
        let source = stream::iter(
            events
                .into_iter()
                .map(|chunks| Ok(LlmResponseStreamEvent::new(chunks)))
                .collect::<Vec<_>>(),
        );
        Response {
            llm_response: LlmResponse::Stream(Box::pin(source)),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        }
    }

    fn observe_with(response: Response, sink: UsageSink) -> Response {
        observe(
            response,
            "model/worker",
            Instant::now(),
            StatsAccumulator::default(),
            0.0,
            None,
            Some(sink),
        )
    }

    #[tokio::test]
    async fn finished_stream_reports_usage_once() {
        let (sink, reports) = collecting_sink();
        let usage = Usage {
            input_tokens: Some(10),
            output_tokens: Some(2),
            ..Usage::default()
        };
        let response = stream_response(vec![
            vec![text("a")],
            vec![text("b")],
            vec![
                LlmResponseChunk::Usage(usage.clone()),
                LlmResponseChunk::MessageStop { reason: None },
            ],
        ]);
        let LlmResponse::Stream(observed) = observe_with(response, sink).llm_response else {
            panic!("expected stream");
        };
        observed.collect::<Vec<_>>().await;

        let reports = reports.lock();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].complete);
        assert_eq!(reports[0].usage, Some(usage));
        assert_eq!(reports[0].output_deltas, 2);
    }

    /// Local servers send usage only at the end of a stream. A client that leaves
    /// early must still be reported, with the deltas it already received.
    #[tokio::test]
    async fn dropped_stream_reports_partial_usage() {
        let (sink, reports) = collecting_sink();
        let response = stream_response(vec![vec![text("a")], vec![text("b")], vec![text("c")]]);
        let LlmResponse::Stream(mut observed) = observe_with(response, sink).llm_response else {
            panic!("expected stream");
        };
        observed.next().await;
        observed.next().await;
        assert!(reports.lock().is_empty());
        drop(observed);

        let reports = reports.lock();
        assert_eq!(reports.len(), 1);
        assert!(!reports[0].complete);
        assert_eq!(reports[0].usage, None);
        assert_eq!(reports[0].output_deltas, 2);
    }

    #[tokio::test]
    async fn aggregate_response_reports_usage() {
        let (sink, reports) = collecting_sink();
        let usage = Usage {
            input_tokens: Some(7),
            output_tokens: Some(1),
            ..Usage::default()
        };
        let response = Response {
            llm_response: LlmResponse::Agg(switchyard_protocol::AggLlmResponse {
                usage: usage.clone(),
                ..Default::default()
            }),
            metadata: None,
            upstream_headers: http::HeaderMap::new(),
        };
        observe_with(response, sink);

        let reports = reports.lock();
        assert_eq!(reports.len(), 1);
        assert!(reports[0].complete);
        assert_eq!(reports[0].usage, Some(usage));
    }
}
