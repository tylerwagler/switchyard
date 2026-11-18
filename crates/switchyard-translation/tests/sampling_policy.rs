// SPDX-License-Identifier: Apache-2.0

//! `SamplingPolicy` decides whether a request encoder follows the hosted provider's
//! sampling rules or writes the client's own values for a server that takes them verbatim.

use serde_json::{Value, json};
use switchyard_translation::{
    LossyConversionPolicy, PreservationPolicy, SamplingPolicy, TranslationEngine, TranslationError,
    TranslationPolicy, WireFormat,
};

const FORMATS: [WireFormat; 3] = [
    WireFormat::OpenAiChat,
    WireFormat::OpenAiResponses,
    WireFormat::AnthropicMessages,
];

// JSON text, so an integer stays an integer and a decimal keeps its digits.
const TEMPERATURES: [&str; 9] = ["0", "0.1", "0.2", "0.3", "0.4", "0.7", "1", "1.5", "2"];

fn policy(sampling: SamplingPolicy, preservation: PreservationPolicy) -> TranslationPolicy {
    TranslationPolicy {
        sampling,
        preservation,
        ..TranslationPolicy::default()
    }
}

fn source_body(format: WireFormat, temperature: Option<&Value>) -> Value {
    let mut body = match format {
        WireFormat::OpenAiChat => json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }),
        WireFormat::OpenAiResponses => json!({
            "model": "m",
            "input": [{"role": "user", "content": "hi"}],
        }),
        WireFormat::AnthropicMessages => json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        }),
    };
    if let Some(value) = temperature {
        body["temperature"] = value.clone();
    }
    body
}

fn wire_max(format: WireFormat) -> f64 {
    match format {
        WireFormat::AnthropicMessages => 1.0,
        WireFormat::OpenAiChat | WireFormat::OpenAiResponses => 2.0,
    }
}

// Upstream's rule, restated: scale source range -> IR 0..2 -> target range; Anthropic omits.
fn provider_expectation(source: WireFormat, target: WireFormat, value: f64) -> Option<f64> {
    if target == WireFormat::AnthropicMessages {
        return None;
    }
    Some(value / wire_max(source) * wire_max(target))
}

fn translate(
    source: WireFormat,
    target: WireFormat,
    body: &Value,
    policy: &TranslationPolicy,
) -> switchyard_translation::TranslationOutput {
    TranslationEngine::default()
        .translate_request(source, target, body, policy)
        .unwrap_or_else(|error| panic!("{source} -> {target}: {error}"))
}

#[test]
fn verbatim_writes_the_clients_temperature_for_every_pair() {
    for preservation in [PreservationPolicy::InMemory, PreservationPolicy::Disabled] {
        let policy = policy(SamplingPolicy::Verbatim, preservation);
        for source in FORMATS {
            for target in FORMATS {
                for text in TEMPERATURES {
                    let sent: Value = serde_json::from_str(text).expect("number");
                    let output =
                        translate(source, target, &source_body(source, Some(&sent)), &policy);
                    let got = output.body.get("temperature");
                    assert_eq!(got, Some(&sent), "{source} -> {target} {text}");
                    // Same JSON token, not only the same number: `1` stays `1`.
                    assert_eq!(
                        got.map(Value::to_string).as_deref(),
                        Some(text),
                        "{source} -> {target} {text}"
                    );
                    assert!(
                        output
                            .diagnostics
                            .iter()
                            .all(|diagnostic| !diagnostic.message.contains("temperature")),
                        "{source} -> {target} {text}: {:?}",
                        output.diagnostics
                    );
                }
                let output = translate(source, target, &source_body(source, None), &policy);
                assert_eq!(output.body.get("temperature"), None, "{source} -> {target}");
            }
        }
    }
}

#[test]
fn provider_keeps_upstream_scaling_and_the_anthropic_omission() {
    // Preservation is off so a same-format pair is re-encoded instead of replayed.
    let policy = policy(SamplingPolicy::Provider, PreservationPolicy::Disabled);
    assert_eq!(
        TranslationPolicy::default().sampling,
        SamplingPolicy::Provider
    );
    for source in FORMATS {
        for target in FORMATS {
            for text in TEMPERATURES {
                let sent: Value = serde_json::from_str(text).expect("number");
                let value = sent.as_f64().expect("f64");
                let output = translate(source, target, &source_body(source, Some(&sent)), &policy);
                assert_eq!(
                    output.body.get("temperature").and_then(Value::as_f64),
                    provider_expectation(source, target, value),
                    "{source} -> {target} {text}"
                );
                let is_reported = output
                    .diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.message.contains("temperature"));
                let anthropic_value = value / wire_max(source);
                assert_eq!(
                    is_reported,
                    target == WireFormat::AnthropicMessages && anthropic_value != 1.0,
                    "{source} -> {target} {text}"
                );
            }
            let output = translate(source, target, &source_body(source, None), &policy);
            assert_eq!(output.body.get("temperature"), None, "{source} -> {target}");
        }
    }
    // The measured cases: 0.4 and greedy are lost, Anthropic values double.
    let chat = |value: f64| source_body(WireFormat::OpenAiChat, Some(&json!(value)));
    let anthropic = |value: f64| source_body(WireFormat::AnthropicMessages, Some(&json!(value)));
    for value in [0.4, 0.0] {
        let body = translate(
            WireFormat::OpenAiChat,
            WireFormat::AnthropicMessages,
            &chat(value),
            &policy,
        )
        .body;
        assert_eq!(body.get("temperature"), None);
    }
    for (sent, received) in [(1.0, 2.0), (0.3, 0.6)] {
        let body = translate(
            WireFormat::AnthropicMessages,
            WireFormat::OpenAiChat,
            &anthropic(sent),
            &policy,
        )
        .body;
        assert_eq!(body["temperature"], json!(received));
    }
}

#[test]
fn reject_policy_refuses_temperature_only_under_provider() {
    let body = source_body(WireFormat::OpenAiChat, Some(&json!(0.4)));
    let engine = TranslationEngine::default();
    let mut policy = policy(SamplingPolicy::Provider, PreservationPolicy::InMemory);
    policy.lossy_conversion_policy = LossyConversionPolicy::Reject;
    let refused = engine.translate_request(
        WireFormat::OpenAiChat,
        WireFormat::AnthropicMessages,
        &body,
        &policy,
    );
    assert!(
        matches!(&refused, Err(TranslationError::LossyConversion { .. })),
        "{refused:?}"
    );

    policy.sampling = SamplingPolicy::Verbatim;
    for source in FORMATS {
        for target in FORMATS {
            for text in TEMPERATURES {
                let sent: Value = serde_json::from_str(text).expect("number");
                let output = engine
                    .translate_request(source, target, &source_body(source, Some(&sent)), &policy)
                    .unwrap_or_else(|error| panic!("{source} -> {target} {text}: {error}"));
                assert_eq!(output.body.get("temperature"), Some(&sent));
                assert!(output.diagnostics.is_empty(), "{:?}", output.diagnostics);
            }
        }
    }
}

#[test]
fn verbatim_writes_an_ir_temperature_as_it_stands() {
    let engine = TranslationEngine::default();
    let verbatim = policy(SamplingPolicy::Verbatim, PreservationPolicy::Disabled);
    let provider = policy(SamplingPolicy::Provider, PreservationPolicy::Disabled);

    // Built on the IR: no client number exists.
    let mut request = switchyard_translation::LlmRequest::default();
    request.sampling.temperature = Some(0.7);
    // Decoded, then edited: the client's 0.3 no longer describes the request.
    let mut edited = engine
        .decode_request(
            WireFormat::AnthropicMessages,
            &source_body(WireFormat::AnthropicMessages, Some(&json!(0.3))),
            &verbatim,
        )
        .expect("decode")
        .request;
    assert_eq!(edited.sampling.temperature, Some(0.6));
    assert_eq!(
        edited
            .sampling
            .client_temperature()
            .map(ToString::to_string)
            .as_deref(),
        Some("0.3")
    );
    edited.sampling.temperature = Some(0.7);
    assert_eq!(edited.sampling.client_temperature(), None);

    for request in [&request, &edited] {
        for target in FORMATS {
            let body = engine
                .encode_request(target, request, &verbatim)
                .expect("encode")
                .body;
            assert_eq!(body["temperature"], json!(0.7), "{target}");
            let body = engine
                .encode_request(target, request, &provider)
                .expect("encode")
                .body;
            let expected = match target {
                WireFormat::AnthropicMessages => None,
                _ => Some(json!(0.7)),
            };
            assert_eq!(body.get("temperature"), expected.as_ref(), "{target}");
        }
    }
}

#[test]
fn verbatim_carries_top_k_and_top_p_across_formats() {
    let verbatim = policy(SamplingPolicy::Verbatim, PreservationPolicy::Disabled);
    let provider = policy(SamplingPolicy::Provider, PreservationPolicy::Disabled);
    for source in FORMATS {
        let mut body = source_body(source, None);
        body["top_k"] = json!(40);
        body["top_p"] = json!(0.9);
        for target in FORMATS {
            let output = translate(source, target, &body, &verbatim);
            assert_eq!(output.body["top_k"], json!(40), "{source} -> {target}");
            assert_eq!(output.body["top_p"], json!(0.9), "{source} -> {target}");

            let output = translate(source, target, &body, &provider);
            assert_eq!(output.body["top_p"], json!(0.9), "{source} -> {target}");
            // Upstream writes `top_k` only from an Anthropic request to an Anthropic target.
            let is_kept =
                source == WireFormat::AnthropicMessages && target == WireFormat::AnthropicMessages;
            assert_eq!(
                output.body.get("top_k"),
                is_kept.then(|| json!(40)).as_ref(),
                "{source} -> {target}"
            );
        }
    }
}

#[test]
fn client_temperature_is_absent_from_serialized_ir_without_a_client_value() {
    let request = switchyard_translation::LlmRequest::default();
    let value = serde_json::to_value(&request).expect("serialize");
    assert_eq!(
        value["sampling"],
        json!({"temperature": null, "top_p": null, "top_k": null})
    );
    let policy: TranslationPolicy = serde_json::from_value(json!({
        "lossy_conversion_policy": "AllowWithDiagnostics",
        "deterministic_ids": "Preserve",
        "preservation": "InMemory",
        "target_capabilities": {},
    }))
    .expect("a policy written before the sampling field still loads");
    assert_eq!(policy.sampling, SamplingPolicy::Provider);
}
