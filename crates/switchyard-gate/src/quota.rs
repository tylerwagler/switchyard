// SPDX-License-Identifier: Apache-2.0

//! Per-user quota counters and usage events, kept in Valkey.
//!
//! Counters use fixed UTC windows: minute, hour, day, week (from Monday), and calendar month.
//! Each served request adds its tokens to every window and appends one event to the
//! `usage:events` stream, which the portal copies into Postgres.

use std::time::{SystemTime, UNIX_EPOCH};

use redis::aio::ConnectionManager;

use crate::keys::{Limits, Weights};

pub const USAGE_STREAM: &str = "usage:events";
/// Caps the stream if the portal stops consuming it. Far above normal backlog.
const STREAM_MAX_LEN: usize = 1_000_000;

/// Which limit refused a request.
#[derive(Debug, PartialEq)]
pub enum Refusal {
    RequestsPerMinute,
    TokensPerMinute,
    Hourly,
    Daily,
    Weekly,
    Monthly,
}

impl Refusal {
    pub fn describe(&self) -> &'static str {
        match self {
            Refusal::RequestsPerMinute => "request rate limit reached; try again next minute",
            Refusal::TokensPerMinute => "token rate limit reached; try again next minute",
            Refusal::Hourly => "hourly token limit reached for your plan",
            Refusal::Daily => "daily token limit reached for your plan",
            Refusal::Weekly => "weekly token limit reached for your plan",
            Refusal::Monthly => "monthly token limit reached for your plan",
        }
    }

    /// Seconds until the refusing window rolls over.
    pub fn retry_after(&self, now: u64) -> u64 {
        let w = Windows::at(now);
        match self {
            Refusal::RequestsPerMinute | Refusal::TokensPerMinute => (w.minute + 1) * 60 - now,
            Refusal::Hourly => (w.hour + 1) * 3600 - now,
            Refusal::Daily => (w.day + 1) * 86_400 - now,
            Refusal::Weekly => (w.week + 1) * 7 * 86_400 - WEEK_OFFSET - now,
            Refusal::Monthly => month_start_after(now) - now,
        }
    }
}

/// Unix day 0 (1970-01-01) was a Thursday; shifting by three days starts weeks on Monday.
const WEEK_OFFSET: u64 = 3 * 86_400;

/// Window numbers for one instant. Each key expires once its window can no longer matter.
#[derive(Debug, PartialEq)]
struct Windows {
    minute: u64,
    hour: u64,
    day: u64,
    week: u64,
    month: u64,
}

impl Windows {
    fn at(now: u64) -> Self {
        let day = now / 86_400;
        let (year, month) = year_month(day);
        Self {
            minute: now / 60,
            hour: now / 3600,
            day,
            week: (now + WEEK_OFFSET) / (7 * 86_400),
            month: year * 12 + (month - 1),
        }
    }

    /// Token counter keys with their time-to-live in seconds.
    fn token_keys(&self, user: &str) -> [(String, i64); 5] {
        [
            (format!("q:{user}:tm:{}", self.minute), 120),
            (format!("q:{user}:th:{}", self.hour), 2 * 3600),
            (format!("q:{user}:td:{}", self.day), 2 * 86_400),
            (format!("q:{user}:tw:{}", self.week), 8 * 86_400),
            (format!("q:{user}:tmo:{}", self.month), 32 * 86_400),
        ]
    }
}

/// Civil year and month (1-12) for a Unix day number.
fn year_month(day: u64) -> (u64, u64) {
    // Howard Hinnant's days-to-civil algorithm, restricted to dates after 1970.
    let z = day + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    (year, month)
}

/// Unix time of the first second of the next calendar month.
fn month_start_after(now: u64) -> u64 {
    let (year, month) = year_month(now / 86_400);
    let (year, month) = if month == 12 {
        (year + 1, 1)
    } else {
        (year, month + 1)
    };
    days_from_civil(year, month) * 86_400
}

fn days_from_civil(year: u64, month: u64) -> u64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y / 400;
    let yoe = y - era * 400;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

/// Counts one request and checks every limit. Tokens are added later, when usage is known.
pub async fn admit(
    valkey: &mut ConnectionManager,
    user: &str,
    limits: &Limits,
    now: u64,
) -> redis::RedisResult<Result<(), Refusal>> {
    let windows = Windows::at(now);
    let requests_key = format!("q:{user}:rm:{}", windows.minute);
    let token_keys = windows.token_keys(user);
    let mut pipe = redis::pipe();
    pipe.atomic()
        .incr(&requests_key, 1)
        .expire(&requests_key, 120)
        .ignore();
    for (key, _) in &token_keys {
        pipe.get(key);
    }
    // Replies, in order: requests this minute, then tokens for each window of `token_keys`.
    let used: Vec<Option<i64>> = pipe.query_async(valkey).await?;
    let used = |i: usize| used.get(i).copied().flatten().unwrap_or(0);
    if limits.rpm.is_some_and(|limit| used(0) > limit) {
        return Ok(Err(Refusal::RequestsPerMinute));
    }
    let token_limits = [
        (limits.tpm, Refusal::TokensPerMinute),
        (limits.hourly, Refusal::Hourly),
        (limits.daily, Refusal::Daily),
        (limits.weekly, Refusal::Weekly),
        (limits.monthly, Refusal::Monthly),
    ];
    for (i, (limit, refusal)) in token_limits.into_iter().enumerate() {
        if limit.is_some_and(|limit| used(i + 1) >= limit) {
            return Ok(Err(refusal));
        }
    }
    Ok(Ok(()))
}

/// One metered request, as written to the usage stream.
#[derive(Debug, Default, PartialEq)]
pub struct UsageEvent {
    pub ts: u64,
    pub user_id: String,
    pub api_key_id: Option<String>,
    pub via: &'static str,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
    pub cache_creation_tokens: u64,
    pub reasoning_tokens: u64,
    /// Weighted token count; this is what counts against quotas and is billed.
    pub billable_tokens: u64,
    pub latency_ms: u64,
    pub complete: bool,
    pub estimated: bool,
    pub chat_id: Option<String>,
}

impl UsageEvent {
    /// Weighted sum of every kind of token, rounded to a whole token.
    pub fn billable(&self, w: &Weights) -> u64 {
        let sum = self.input_tokens as f64 * w.input
            + self.cached_tokens as f64 * w.cache_read
            + self.cache_creation_tokens as f64 * w.cache_write
            + self.output_tokens as f64 * w.output
            + self.reasoning_tokens as f64 * w.reasoning;
        sum.round().max(0.0) as u64
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        let flag = |b: bool| if b { "1" } else { "0" }.to_string();
        let mut fields = vec![
            ("ts", self.ts.to_string()),
            ("user_id", self.user_id.clone()),
            ("via", self.via.to_string()),
            ("model", self.model.clone()),
            ("input_tokens", self.input_tokens.to_string()),
            ("output_tokens", self.output_tokens.to_string()),
            ("cached_tokens", self.cached_tokens.to_string()),
            (
                "cache_creation_tokens",
                self.cache_creation_tokens.to_string(),
            ),
            ("reasoning_tokens", self.reasoning_tokens.to_string()),
            ("billable_tokens", self.billable_tokens.to_string()),
            ("latency_ms", self.latency_ms.to_string()),
            ("complete", flag(self.complete)),
            ("estimated", flag(self.estimated)),
        ];
        if let Some(key) = &self.api_key_id {
            fields.push(("api_key_id", key.clone()));
        }
        if let Some(chat) = &self.chat_id {
            fields.push(("chat_id", chat.clone()));
        }
        fields
    }
}

/// Adds the event's billable tokens to every quota window and appends it to the usage stream.
pub async fn record(valkey: &mut ConnectionManager, event: &UsageEvent) -> redis::RedisResult<()> {
    let tokens = event.billable_tokens as i64;
    let mut pipe = redis::pipe();
    pipe.atomic();
    for (key, ttl) in Windows::at(event.ts).token_keys(&event.user_id) {
        pipe.incr(&key, tokens).ignore().expire(&key, ttl).ignore();
    }
    pipe.cmd("XADD")
        .arg(USAGE_STREAM)
        .arg("MAXLEN")
        .arg("~")
        .arg(STREAM_MAX_LEN)
        .arg("*")
        .arg(event.fields())
        .ignore();
    pipe.query_async(valkey).await
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2026-09-25 23:59:30 UTC, a Friday.
    const FRI: u64 = 1_790_380_770;

    #[test]
    fn windows_use_utc_calendar_boundaries() {
        assert_eq!(year_month(FRI / 86_400), (2026, 9));
        assert_eq!(year_month(0), (1970, 1));
        assert_eq!(year_month(days_from_civil(2024, 2) + 28), (2024, 2)); // leap day
        assert_eq!(days_from_civil(1970, 1), 0);
    }

    #[test]
    fn retry_after_points_at_the_next_window() {
        assert_eq!(Refusal::RequestsPerMinute.retry_after(FRI), 30);
        assert_eq!(Refusal::Daily.retry_after(FRI), 30);
        // Friday 23:59:30 to Monday 00:00:00 is two days and thirty seconds.
        assert_eq!(Refusal::Weekly.retry_after(FRI), 2 * 86_400 + 30);
        // Next month starts 2026-10-01 00:00:00.
        assert_eq!(Refusal::Monthly.retry_after(FRI), 5 * 86_400 + 30);
    }

    #[test]
    fn billable_tokens_weight_each_kind() {
        let event = UsageEvent {
            input_tokens: 100,
            cached_tokens: 16_000,
            cache_creation_tokens: 4,
            output_tokens: 20,
            reasoning_tokens: 5,
            ..UsageEvent::default()
        };
        // Default weights: a cached prefix read counts a tenth of an uncached token.
        assert_eq!(
            event.billable(&Weights::default()),
            100 + 1_600 + 4 + 20 + 5
        );
        let flat = Weights {
            cache_read: 1.0,
            ..Weights::default()
        };
        assert_eq!(event.billable(&flat), 16_129);
    }
}
