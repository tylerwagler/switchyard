// SPDX-License-Identifier: Apache-2.0

//! Per-route access rules: which portal users or roles may call a route.
//!
//! A route with no rule is open to every active key, as before. A restricted route answers
//! only callers whose portal user id or role its rule names; everyone else gets a 403 before
//! the request reaches the route's upstream. The rules live in their own TOML file, given to
//! the gate with `--access`:
//!
//! ```toml
//! # Tyler's own DeepSeek key: he pays per token.
//! [routes.deepseek-chat]
//! users = ["<portal user id>"]                       # portal user ids
//! roles = []                                           # or portal roles, such as "admin"
//! ```
//!
//! Quote a route id that contains dots (`[routes."qwen3.8-flash-next"]`); unquoted, TOML
//! reads the dots as nested tables. The gate checks the rules against the configured routes
//! at startup (and in `--dry-run`) and refuses to start if a restriction could be bypassed.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Deserialize;

use crate::keys::Identity;

/// The gate's per-route access rules, keyed by route id (the model id callers send).
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccessRules {
    #[serde(default)]
    routes: BTreeMap<String, RouteRule>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RouteRule {
    /// Portal user ids that may call the route.
    #[serde(default)]
    users: BTreeSet<String>,
    /// Portal roles whose members may call the route.
    #[serde(default)]
    roles: BTreeSet<String>,
}

impl AccessRules {
    /// Parses rules from TOML text.
    pub fn from_toml(text: &str) -> Result<Self, String> {
        let rules: Self = toml::from_str(text).map_err(|e| format!("invalid access rules: {e}"))?;
        for (route, rule) in &rules.routes {
            if rule.users.is_empty() && rule.roles.is_empty() {
                return Err(format!(
                    "access rule for route {route} names no users or roles, so nobody could call it"
                ));
            }
        }
        Ok(rules)
    }

    /// Reads and parses a rules file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read access rules {}: {e}", path.display()))?;
        Self::from_toml(&text)
    }

    /// Whether any route is restricted.
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }

    /// Number of restricted routes.
    pub fn len(&self) -> usize {
        self.routes.len()
    }

    /// Whether `route` has an access rule.
    pub fn is_restricted(&self, route: &str) -> bool {
        self.routes.contains_key(route)
    }

    /// Whether `identity` may call `route`. Routes without a rule are open.
    pub fn allows(&self, identity: &Identity, route: &str) -> bool {
        match self.routes.get(route) {
            None => true,
            Some(rule) => {
                rule.users.contains(&identity.user_id) || rule.roles.contains(&identity.role)
            }
        }
    }

    /// Checks the rules against the configured routes, so a restriction cannot be bypassed:
    /// every restricted id must be a configured route, a restricted route must not be the
    /// default route (models no route names land there), and no unrestricted route may share
    /// an upstream with a restricted one (it could reach the same paid backend, for example
    /// as a fallback target).
    pub fn validate(
        &self,
        routes: &[(String, Vec<String>)],
        default_route: Option<&str>,
    ) -> Result<(), String> {
        for restricted in self.routes.keys() {
            let Some((_, upstreams)) = routes.iter().find(|(id, _)| id == restricted) else {
                return Err(format!(
                    "access rule names route {restricted}, which is not configured"
                ));
            };
            if default_route == Some(restricted.as_str()) {
                return Err(format!(
                    "route {restricted} is restricted but is the default route, so callers \
                     naming any unknown model would reach it"
                ));
            }
            for (other, other_upstreams) in routes {
                if self.is_restricted(other) {
                    continue;
                }
                if let Some(shared) = other_upstreams.iter().find(|u| upstreams.contains(u)) {
                    return Err(format!(
                        "unrestricted route {other} shares upstream {shared} with restricted \
                         route {restricted}; restrict both or give them separate upstreams"
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(user_id: &str, role: &str) -> Identity {
        Identity {
            key_id: None,
            user_id: user_id.to_string(),
            role: role.to_string(),
            active: true,
            trusted_forwarder: false,
            limits: crate::keys::Limits::default(),
            weights: crate::keys::Weights::default(),
        }
    }

    const RULES: &str = r#"
        [routes.deepseek]
        users = ["u-tyler"]
        roles = ["owner"]
    "#;

    fn routes() -> Vec<(String, Vec<String>)> {
        vec![
            ("main".into(), vec!["http://sparky".into()]),
            ("deepseek".into(), vec!["https://api.deepseek.com".into()]),
        ]
    }

    #[test]
    fn restricted_routes_admit_named_users_and_roles_only() {
        let rules = AccessRules::from_toml(RULES).unwrap();
        assert!(rules.allows(&identity("u-tyler", "user"), "deepseek"));
        assert!(rules.allows(&identity("u-other", "owner"), "deepseek"));
        assert!(!rules.allows(&identity("u-other", "user"), "deepseek"));
        assert!(!rules.allows(&identity("u-other", "admin"), "deepseek"));
        // Routes without a rule stay open.
        assert!(rules.allows(&identity("u-other", "user"), "main"));
        assert!(rules.is_restricted("deepseek") && !rules.is_restricted("main"));
    }

    #[test]
    fn empty_rules_restrict_nothing() {
        let rules = AccessRules::from_toml("").unwrap();
        assert!(rules.is_empty());
        assert!(rules.allows(&identity("u", "user"), "anything"));
    }

    #[test]
    fn a_rule_naming_nobody_is_refused() {
        let err = AccessRules::from_toml("[routes.deepseek]\nusers = []\n").unwrap_err();
        assert!(err.contains("names no users or roles"), "{err}");
    }

    #[test]
    fn unknown_fields_are_refused() {
        assert!(AccessRules::from_toml("[routes.x]\nuser = [\"a\"]\n").is_err());
        assert!(AccessRules::from_toml("[route.x]\nusers = [\"a\"]\n").is_err());
    }

    #[test]
    fn validation_accepts_a_separate_restricted_route() {
        let rules = AccessRules::from_toml(RULES).unwrap();
        rules.validate(&routes(), Some("main")).unwrap();
    }

    #[test]
    fn validation_refuses_unknown_routes() {
        let rules = AccessRules::from_toml("[routes.nope]\nroles = [\"admin\"]\n").unwrap();
        let err = rules.validate(&routes(), Some("main")).unwrap_err();
        assert!(err.contains("not configured"), "{err}");
    }

    #[test]
    fn validation_refuses_a_restricted_default_route() {
        let rules = AccessRules::from_toml(RULES).unwrap();
        let err = rules.validate(&routes(), Some("deepseek")).unwrap_err();
        assert!(err.contains("default route"), "{err}");
    }

    #[test]
    fn validation_refuses_an_open_route_sharing_the_paid_upstream() {
        let rules = AccessRules::from_toml(RULES).unwrap();
        let mut routes = routes();
        routes.push((
            "fallback".into(),
            vec!["http://sparky".into(), "https://api.deepseek.com".into()],
        ));
        let err = rules.validate(&routes, Some("main")).unwrap_err();
        assert!(err.contains("shares upstream"), "{err}");
    }
}
