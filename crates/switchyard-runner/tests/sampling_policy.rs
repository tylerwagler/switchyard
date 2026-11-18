// SPDX-License-Identifier: Apache-2.0

//! `[llm_clients.<name>] sampling` reaches the request sent to that model server.

use serde_json::{Value, json};
use switchyard_protocol::{Metadata, Request, WireFormat};
use switchyard_runner::Runner;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const TEMPERATURES: [&str; 9] = ["0", "0.1", "0.2", "0.3", "0.4", "0.7", "1", "1.5", "2"];

// One Anthropic-format and one OpenAI-format server per mode, as a deployment declares them.
fn config(endpoint: &str) -> String {
    let mut config = String::from("schema_version = 1\n");
    for (name, format, base_url, setting) in [
        (
            "messages_verbatim",
            "anthropic_messages",
            "",
            "sampling = \"verbatim\"",
        ),
        (
            "chat_verbatim",
            "openai_chat",
            "/v1",
            "sampling = \"verbatim\"",
        ),
        ("messages_default", "anthropic_messages", "", ""),
        (
            "chat_provider",
            "openai_chat",
            "/v1",
            "sampling = \"provider\"",
        ),
    ] {
        config.push_str(&format!(
            r#"
[llm_clients.{name}]
format = "{format}"
base_url = "{endpoint}{base_url}"
max_retries = 0
failure_cooldown_ms = 0
{setting}

[targets.{name}]
id = "served/{name}"
llm_client = "{name}"

[routes.{name}]
id = "{name}"
type = "passthrough"
target = "{name}"
"#
        ));
    }
    config
}

async fn server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "1", "object": "chat.completion", "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {}
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        })))
        .mount(&server)
        .await;
    server
}

fn client_body(format: WireFormat, temperature: Option<&str>) -> Value {
    let mut body = match format {
        WireFormat::AnthropicMessages => json!({
            "model": "client-facing", "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}]
        }),
        WireFormat::OpenAiResponses => json!({
            "model": "client-facing", "input": [{"role": "user", "content": "hi"}]
        }),
        WireFormat::OpenAiChat => json!({
            "model": "client-facing", "messages": [{"role": "user", "content": "hi"}]
        }),
    };
    if let Some(text) = temperature {
        body["temperature"] = serde_json::from_str(text).expect("number");
    }
    body
}

// Runs one client request through a route and returns the body the model server received.
async fn upstream_body(
    runner: &Runner,
    server: &MockServer,
    route: &str,
    source: WireFormat,
    temperature: Option<&str>,
) -> Value {
    let before = server.received_requests().await.expect("recorded").len();
    let request = Request {
        llm_request: switchyard_translation::decode_request(
            source,
            &client_body(source, temperature),
        )
        .expect("client request"),
        raw_request: None,
        metadata: Some(Metadata::default()),
    };
    let outcome = runner
        .route(route)
        .expect("route")
        .execute(request, None)
        .await
        .unwrap_or_else(|error| panic!("{route} from {source}: {error}"));
    outcome
        .response
        .llm_response
        .into_agg()
        .await
        .expect("response");
    let received = server.received_requests().await.expect("recorded requests");
    assert_eq!(received.len(), before + 1, "{route} from {source}");
    serde_json::from_slice(&received[before].body).expect("provider JSON")
}

const SOURCES: [WireFormat; 3] = [
    WireFormat::OpenAiChat,
    WireFormat::OpenAiResponses,
    WireFormat::AnthropicMessages,
];

#[tokio::test]
async fn a_verbatim_client_receives_the_callers_temperature() {
    let server = server().await;
    let runner = Runner::from_toml(&config(&server.uri())).expect("config");
    for route in ["messages_verbatim", "chat_verbatim"] {
        for source in SOURCES {
            for text in TEMPERATURES {
                let body = upstream_body(&runner, &server, route, source, Some(text)).await;
                assert_eq!(
                    body.get("temperature").map(Value::to_string).as_deref(),
                    Some(text),
                    "{source} -> {route}: {body}"
                );
            }
            let body = upstream_body(&runner, &server, route, source, None).await;
            assert!(
                body.get("temperature").is_none(),
                "{source} -> {route}: {body}"
            );
        }
    }
}

#[tokio::test]
async fn an_unset_or_provider_client_keeps_the_provider_rules() {
    let server = server().await;
    let runner = Runner::from_toml(&config(&server.uri())).expect("config");
    for text in ["0", "0.4"] {
        let body = upstream_body(
            &runner,
            &server,
            "messages_default",
            WireFormat::OpenAiChat,
            Some(text),
        )
        .await;
        assert!(body.get("temperature").is_none(), "{body}");
    }
    for (sent, received) in [("1.0", 2.0), ("0.3", 0.6)] {
        let body = upstream_body(
            &runner,
            &server,
            "chat_provider",
            WireFormat::AnthropicMessages,
            Some(sent),
        )
        .await;
        assert_eq!(body["temperature"], json!(received), "{body}");
    }
}
