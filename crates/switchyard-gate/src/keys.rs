// SPDX-License-Identifier: Apache-2.0

//! Looks up API keys and users in the portal database, with a short in-process cache.
//!
//! The gate's database role may only call `gate.lookup_key` and `gate.lookup_user_by_email`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use tokio_postgres::{Client, NoTls, Row};

/// A revoked key or a changed tier takes effect within this time.
const FOUND_TTL: Duration = Duration::from_secs(30);
/// Unknown keys are cached briefly so guessing does not reach the database on every request.
const MISSING_TTL: Duration = Duration::from_secs(10);

/// Who a request acts for, and that user's tier limits. A `None` limit is unlimited.
#[derive(Clone, Debug, PartialEq)]
pub struct Identity {
    pub key_id: Option<String>,
    pub user_id: String,
    pub role: String,
    pub active: bool,
    pub trusted_forwarder: bool,
    pub limits: Limits,
    pub weights: Weights,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Limits {
    pub rpm: Option<i64>,
    pub tpm: Option<i64>,
    pub hourly: Option<i64>,
    pub daily: Option<i64>,
    pub weekly: Option<i64>,
    pub monthly: Option<i64>,
}

/// How much each kind of token counts toward quotas and billing. Cached prefix reads
/// are cheap to serve, so they usually count for a fraction of an uncached token.
#[derive(Clone, Debug, PartialEq)]
pub struct Weights {
    pub input: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub output: f64,
    pub reasoning: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            input: 1.0,
            cache_read: 0.1,
            cache_write: 1.0,
            output: 1.0,
            reasoning: 1.0,
        }
    }
}

// Ids are read as text so the gate needs no UUID type support.
const COLUMNS: &str = "key_id::text as key_id, user_id::text as user_id, role, status, \
    trusted_forwarder, rate_limit_rpm, rate_limit_tpm, hourly_limit, daily_limit, \
    weekly_limit, monthly_limit, w_input, w_cache_read, w_cache_write, w_output, w_reasoning";

pub fn hash_key(key: &str) -> String {
    hex::encode(Sha256::digest(key.as_bytes()))
}

pub struct KeyStore {
    database_url: String,
    client: tokio::sync::Mutex<Option<Arc<Client>>>,
    cache: Mutex<HashMap<String, (Instant, Option<Identity>)>>,
}

impl KeyStore {
    pub fn new(database_url: String) -> Self {
        Self {
            database_url,
            client: tokio::sync::Mutex::new(None),
            cache: Mutex::new(HashMap::new()),
        }
    }

    pub async fn by_key(&self, key: &str) -> Result<Option<Identity>, tokio_postgres::Error> {
        let hash = hash_key(key);
        let query = format!("select {COLUMNS} from gate.lookup_key($1)");
        self.cached(format!("k:{hash}"), &query, &hash).await
    }

    pub async fn by_email(&self, email: &str) -> Result<Option<Identity>, tokio_postgres::Error> {
        let email = email.to_lowercase();
        let query = format!("select {COLUMNS} from gate.lookup_user_by_email($1)");
        self.cached(format!("e:{email}"), &query, &email).await
    }

    async fn cached(
        &self,
        cache_key: String,
        query: &str,
        arg: &str,
    ) -> Result<Option<Identity>, tokio_postgres::Error> {
        if let Some((at, identity)) = self.cache.lock().get(&cache_key) {
            let ttl = if identity.is_some() {
                FOUND_TTL
            } else {
                MISSING_TTL
            };
            if at.elapsed() < ttl {
                return Ok(identity.clone());
            }
        }
        let client = self.client().await?;
        let identity = match client.query_opt(query, &[&arg]).await {
            Ok(row) => row.as_ref().map(identity_from_row),
            Err(error) => {
                // Drop a broken connection so the next request reconnects.
                if client.is_closed() {
                    self.client.lock().await.take();
                }
                return Err(error);
            }
        };
        self.cache
            .lock()
            .insert(cache_key, (Instant::now(), identity.clone()));
        Ok(identity)
    }

    async fn client(&self) -> Result<Arc<Client>, tokio_postgres::Error> {
        let mut slot = self.client.lock().await;
        if let Some(client) = slot.as_ref().filter(|client| !client.is_closed()) {
            return Ok(Arc::clone(client));
        }
        let (client, connection) = tokio_postgres::connect(&self.database_url, NoTls).await?;
        tokio::spawn(async move {
            if let Err(error) = connection.await {
                tracing::warn!(%error, "gate database connection closed");
            }
        });
        let client = Arc::new(client);
        *slot = Some(Arc::clone(&client));
        Ok(client)
    }
}

fn identity_from_row(row: &Row) -> Identity {
    let uuid = |name: &str| row.get::<_, Option<String>>(name);
    Identity {
        key_id: uuid("key_id"),
        user_id: uuid("user_id").unwrap_or_default(),
        role: row.get("role"),
        active: row.get::<_, String>("status") == "active",
        trusted_forwarder: row.get("trusted_forwarder"),
        limits: Limits {
            rpm: row.get::<_, Option<i32>>("rate_limit_rpm").map(i64::from),
            tpm: row.get("rate_limit_tpm"),
            hourly: row.get("hourly_limit"),
            daily: row.get("daily_limit"),
            weekly: row.get("weekly_limit"),
            monthly: row.get("monthly_limit"),
        },
        weights: Weights {
            input: row.get("w_input"),
            cache_read: row.get("w_cache_read"),
            cache_write: row.get("w_cache_write"),
            output: row.get("w_output"),
            reasoning: row.get("w_reasoning"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The portal stores the same lowercase hex digest, so the formats must match exactly.
    #[test]
    fn key_hash_is_lowercase_sha256_hex() {
        assert_eq!(
            hash_key("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
