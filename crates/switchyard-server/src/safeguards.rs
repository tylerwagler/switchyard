// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Answers Claude Code's server-side auto mode request.
//!
//! In auto mode, Claude Code can send a `safeguards` field on `/v1/messages`.
//! It asks the server to check each tool use in the reply for danger. The reply
//! carries the verdicts in `safeguard_results`. Switchyard owns this field: it
//! removes it before routing, so no backend sees it, and answers it on the
//! reply. The backends behind Switchyard do not run this check, so the answer
//! is `unsupported`. Claude Code then runs its own classifier requests.

use futures_util::StreamExt;
use serde_json::{Value, json};
use switchyard_translation::RawEventStream;

const DANGEROUS_TOOL_USE: &str = "dangerous_tool_use";

/// Removes `safeguards` from an Anthropic request body. Returns true when it
/// asked for the `dangerous_tool_use` check, which the reply must answer.
///
/// Any shape is accepted. A request is never refused over this field, because
/// Claude Code denies every auto mode action after such a refusal.
pub(crate) fn take_request(body: &mut Value) -> bool {
    let Some(safeguards) = body
        .as_object_mut()
        .and_then(|body| body.remove("safeguards"))
    else {
        return false;
    };
    safeguards.as_array().is_some_and(|entries| {
        entries
            .iter()
            .any(|entry| entry.get("type").and_then(Value::as_str) == Some(DANGEROUS_TOOL_USE))
    })
}

fn results() -> Value {
    json!([{"type": DANGEROUS_TOOL_USE, "status": {"type": "unsupported"}}])
}

/// Adds the answer to a buffered Anthropic message.
pub(crate) fn add_to_message(message: &mut Value) {
    if let Some(message) = message.as_object_mut() {
        message.insert("safeguard_results".to_string(), results());
    }
}

/// Adds the answer to a streamed Anthropic reply. Claude Code reads it from
/// the `delta` of the `message_delta` event, which ends the reply.
pub(crate) fn add_to_stream(events: RawEventStream) -> RawEventStream {
    Box::pin(events.map(|event| {
        event.map(|mut event| {
            if event.get("type").and_then(Value::as_str) == Some("message_delta")
                && let Some(delta) = event.get_mut("delta").and_then(Value::as_object_mut)
            {
                delta.insert("safeguard_results".to_string(), results());
            }
            event
        })
    }))
}

#[cfg(test)]
mod tests {
    use futures_util::stream;
    use switchyard_translation::LlmStreamError;

    use super::*;

    #[test]
    fn take_request_removes_the_field_and_reports_the_check() {
        let cases = [
            (
                json!({"model": "m", "safeguards": [{"type": "dangerous_tool_use",
                    "classifier_context": {"v": 1, "permission_mode": "auto"}}]}),
                true,
            ),
            (
                json!({"safeguards": [{"type": "other"}, {"type": "dangerous_tool_use"}]}),
                true,
            ),
            (json!({"safeguards": [{"type": "other"}]}), false),
            (json!({"safeguards": [1, "x", null, {"type": 7}]}), false),
            (json!({"safeguards": {"type": "dangerous_tool_use"}}), false),
            (json!({"safeguards": null}), false),
        ];
        for (mut body, asked) in cases {
            assert_eq!(take_request(&mut body), asked, "{body}");
            assert!(body.get("safeguards").is_none());
        }
        let mut body = json!({"model": "m"});
        assert!(!take_request(&mut body));
        assert_eq!(body, json!({"model": "m"}));
    }

    #[test]
    fn message_gets_the_unsupported_answer() {
        let mut message = json!({"type": "message", "stop_reason": "tool_use"});
        add_to_message(&mut message);
        assert_eq!(
            message["safeguard_results"],
            json!([{"type": "dangerous_tool_use", "status": {"type": "unsupported"}}])
        );
    }

    #[tokio::test]
    async fn only_message_delta_gets_the_answer() {
        let events: RawEventStream = Box::pin(stream::iter(vec![
            Ok(json!({"type": "message_start", "message": {}})),
            Ok(json!({"type": "content_block_delta", "delta": {"type": "text_delta"}})),
            Ok(
                json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 3}}),
            ),
            Ok(json!({"type": "message_stop"})),
            Err(LlmStreamError::Upstream(json!({"type": "error"}))),
        ]));
        let events: Vec<_> = add_to_stream(events).collect().await;
        let with_answer: Vec<_> = events
            .iter()
            .filter_map(|event| event.as_ref().ok())
            .filter(|event| event.to_string().contains("safeguard_results"))
            .collect();
        assert_eq!(with_answer.len(), 1);
        assert_eq!(
            with_answer[0]["delta"],
            json!({"stop_reason": "tool_use", "safeguard_results":
                [{"type": "dangerous_tool_use", "status": {"type": "unsupported"}}]})
        );
        assert!(with_answer[0].get("safeguard_results").is_none());
        assert!(events[4].is_err());
    }
}
