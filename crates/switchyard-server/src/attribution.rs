// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Claude Code's attribution block.
//!
//! Claude Code puts one `system` block first in every Anthropic request:
//! `x-anthropic-billing-header: cc_version=<version>.<fingerprint>; cc_entrypoint=<name>;`,
//! sometimes followed by `cc_workload=<label>;` and `cc_is_subagent=true;`.
//! Anthropic's API removes it by position. Other upstreams receive it as prompt
//! text, and its per-conversation fingerprint makes every session's prompt start
//! differently. A route can strip it before forwarding; the gateway reads it first.

use serde_json::Value;

const PREFIX: &str = "x-anthropic-billing-header:";

/// What the attribution block says about the client. Every value is checked,
/// because the client writes it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClientAttribution {
    /// Claude Code version, such as `2.1.289`, without the conversation fingerprint.
    pub version: Option<String>,
    /// How Claude Code was started, such as `cli` or `sdk-ts`.
    pub entrypoint: Option<String>,
    /// The workload label, when the client sets one.
    pub workload: Option<String>,
}

fn first_block_text(body: &Value) -> Option<&str> {
    let text = body
        .get("system")?
        .as_array()?
        .first()?
        .get("text")?
        .as_str()?;
    text.starts_with(PREFIX).then_some(text)
}

/// Reads the attribution block, when the request's first `system` block is one.
pub(crate) fn read(body: &Value) -> Option<ClientAttribution> {
    let text = first_block_text(body)?;
    let mut client = ClientAttribution::default();
    for field in text[PREFIX.len()..].split(';') {
        let Some((key, value)) = field.trim().split_once('=') else {
            continue;
        };
        match key {
            "cc_version" => client.version = version(value),
            "cc_entrypoint" => client.entrypoint = label(value),
            "cc_workload" => client.workload = label(value),
            _ => {}
        }
    }
    Some(client)
}

/// Removes the attribution block. Only the first `system` block can be one,
/// matching Anthropic's own positional strip. Returns whether it was removed.
pub(crate) fn strip(body: &mut Value) -> bool {
    if first_block_text(body).is_none() {
        return false;
    }
    if let Some(blocks) = body.get_mut("system").and_then(Value::as_array_mut) {
        blocks.remove(0);
    }
    true
}

// `2.1.289.a1b2c` -> `2.1.289`: three numeric parts, the fingerprint dropped.
fn version(value: &str) -> Option<String> {
    let parts = value.split('.').take(3).collect::<Vec<_>>();
    let numeric = |part: &&str| {
        !part.is_empty() && part.len() <= 6 && part.bytes().all(|b| b.is_ascii_digit())
    };
    (parts.len() == 3 && parts.iter().all(numeric)).then(|| parts.join("."))
}

fn label(value: &str) -> Option<String> {
    let ok = !value.is_empty()
        && value.len() <= 32
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'));
    ok.then(|| value.to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const BLOCK: &str = "x-anthropic-billing-header: cc_version=2.1.289.f3a; \
                         cc_entrypoint=cli; cc_workload=nightly; cc_is_subagent=true;";

    fn request(first: &str) -> Value {
        json!({"model": "m", "system": [
            {"type": "text", "text": first},
            {"type": "text", "text": "You are Claude Code."}
        ]})
    }

    #[test]
    fn reads_version_entrypoint_and_workload() {
        assert_eq!(
            read(&request(BLOCK)),
            Some(ClientAttribution {
                version: Some("2.1.289".to_string()),
                entrypoint: Some("cli".to_string()),
                workload: Some("nightly".to_string()),
            })
        );
    }

    #[test]
    fn bad_values_are_dropped_and_other_blocks_are_not_read() {
        let odd = "x-anthropic-billing-header: cc_version=latest; cc_entrypoint=a b;";
        assert_eq!(read(&request(odd)), Some(ClientAttribution::default()));
        assert_eq!(read(&request("You are Claude Code.")), None);
        let second = json!({"system": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": BLOCK}
        ]});
        assert_eq!(read(&second), None);
        assert_eq!(read(&json!({"system": BLOCK})), None);
    }

    #[test]
    fn strip_removes_only_a_leading_attribution_block() {
        let mut body = request(BLOCK);
        assert!(strip(&mut body));
        assert_eq!(
            body["system"],
            json!([{"type": "text", "text": "You are Claude Code."}])
        );
        let mut plain = request("You are Claude Code.");
        assert!(!strip(&mut plain));
        assert_eq!(plain, request("You are Claude Code."));
    }
}
