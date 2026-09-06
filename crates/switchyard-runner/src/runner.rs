// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Named route table and server-facing route metadata.

use std::collections::BTreeMap;
use std::path::Path;

use libsy::RoutingOutcome;
use serde_json::Value;
use switchyard_protocol::{ModelId, WireFormat};

use crate::config;
use crate::{
    EmbeddingsConfig, ModelCapabilities, RerankConfig, ResolvedWebSearch, Route, RunnerError,
    SearchConfig,
};

/// Immutable named route table.
pub struct Runner {
    routes: Vec<(ModelId, Route)>,
    /// Route serving models that match no route id. Without it an unrecognized
    /// model is a hard 404, so every id a client might send has to be
    /// enumerated up front.
    default_route: Option<ModelId>,
    fallback_base_url: Option<String>,
    web_search: Option<ResolvedWebSearch>,
    embeddings: BTreeMap<String, EmbeddingsConfig>,
    rerank: BTreeMap<String, RerankConfig>,
    search: BTreeMap<String, SearchConfig>,
    /// Configured llm client name -> base URL, for liveness probing. Every
    /// other surface is retrospective: this is what answers "is that box
    /// reachable right now", which is the question an outage actually raises.
    upstreams: BTreeMap<String, String>,
}

/// Borrowed model metadata returned while listing routes.
pub struct ModelInfo<'a> {
    pub id: &'a ModelId,
    pub algorithm: &'a str,
    pub capabilities: ModelCapabilities,
}

/// Fully resolved routing decision.
pub struct DecisionDescription {
    pub selected: DecisionTarget,
    pub fallbacks: Vec<DecisionTarget>,
}

/// Non-secret configured target details returned by the decision endpoint.
#[derive(Clone)]
pub struct DecisionTarget {
    pub target: String,
    pub model: ModelId,
    pub format: WireFormat,
    pub base_url: String,
    pub extra_body: BTreeMap<String, Value>,
}

impl Runner {
    /// Loads and validates a version-1 deployment TOML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, RunnerError> {
        config::load_runner(path)
    }

    /// Loads and validates a version-1 deployment TOML document.
    ///
    /// Use [`Self::load`] when the deployment is stored in a file.
    pub fn from_toml(source: &str) -> Result<Self, RunnerError> {
        config::runner_from_toml(source)
    }

    /// Builds a runner from named routes in caller-provided order.
    /// Pre-condition: There must be at least one route.
    pub fn new(routes: Vec<(ModelId, Route)>) -> Self {
        Self {
            routes,
            default_route: None,
            fallback_base_url: None,
            web_search: None,
            embeddings: BTreeMap::new(),
            rerank: BTreeMap::new(),
            search: BTreeMap::new(),
            upstreams: BTreeMap::new(),
        }
    }

    pub(crate) fn with_default_route(mut self, default_route: Option<ModelId>) -> Self {
        self.default_route = default_route;
        self
    }

    pub(crate) fn with_fallback_url(mut self, fallback_base_url: Option<String>) -> Self {
        self.fallback_base_url = fallback_base_url;
        self
    }

    pub(crate) fn with_web_search(mut self, web_search: Option<ResolvedWebSearch>) -> Self {
        self.web_search = web_search;
        self
    }

    pub(crate) fn with_embeddings(
        mut self,
        embeddings: BTreeMap<String, EmbeddingsConfig>,
    ) -> Self {
        self.embeddings = embeddings;
        self
    }

    pub(crate) fn with_rerank(mut self, rerank: BTreeMap<String, RerankConfig>) -> Self {
        self.rerank = rerank;
        self
    }

    pub(crate) fn with_search(mut self, search: BTreeMap<String, SearchConfig>) -> Self {
        self.search = search;
        self
    }

    pub(crate) fn with_upstreams(mut self, upstreams: BTreeMap<String, String>) -> Self {
        self.upstreams = upstreams;
        self
    }

    /// Configured llm clients as (name, base URL), for liveness probing.
    pub fn upstreams(&self) -> &BTreeMap<String, String> {
        &self.upstreams
    }

    /// Returns the resolved hosted web-search settings, if enabled.
    pub fn web_search(&self) -> Option<&ResolvedWebSearch> {
        self.web_search.as_ref()
    }

    /// Named embeddings backends (`[embeddings.*]`).
    pub fn embeddings(&self) -> &BTreeMap<String, EmbeddingsConfig> {
        &self.embeddings
    }

    /// Named rerank backends (`[rerank.*]`).
    pub fn rerank(&self) -> &BTreeMap<String, RerankConfig> {
        &self.rerank
    }

    /// Named search endpoints (`[search.*]`).
    pub fn search(&self) -> &BTreeMap<String, SearchConfig> {
        &self.search
    }

    /// Returns the route registered for a model, falling back to the
    /// configured default route when the id matches none.
    ///
    /// Every caller resolves through here -- the server and the Relay plugin
    /// alike -- so the default applies uniformly.
    pub fn route(&self, model: &str) -> Option<&Route> {
        self.exact_route(model).or_else(|| {
            self.default_route
                .as_ref()
                .and_then(|id| self.exact_route(id.as_str()))
        })
    }

    /// Route registered under exactly this id, ignoring the default.
    /// `/v1/models` advertises the configured ids only: a default route must
    /// not make the gateway claim it serves every id in existence.
    pub fn exact_route(&self, model: &str) -> Option<&Route> {
        self.routes
            .iter()
            .find(|(id, _)| id.as_str() == model)
            .map(|(_, route)| route)
    }

    /// The configured default route id, if any.
    pub fn default_route(&self) -> Option<&ModelId> {
        self.default_route.as_ref()
    }

    /// Iterates over configured routes in caller-provided order.
    pub fn models(&self) -> impl Iterator<Item = ModelInfo<'_>> {
        self.routes.iter().map(|(id, route)| ModelInfo {
            id,
            algorithm: route.algorithm_name(),
            capabilities: route.capabilities(),
        })
    }

    /// Returns the validated API root used for unmatched HTTP requests.
    pub fn fallback_base_url(&self) -> Option<&str> {
        self.fallback_base_url.as_deref()
    }

    /// Resolves an outcome to configured target names and non-secret client settings.
    pub fn describe_decision(
        &self,
        model: &ModelId,
        outcome: &RoutingOutcome,
    ) -> Option<DecisionDescription> {
        let route = self.route(model.as_str())?;
        let resolve = |selected: &ModelId| route.decision_target(selected);
        let mut model_ids = outcome.selected_model_ids.iter();
        Some(DecisionDescription {
            selected: resolve(model_ids.next()?)?,
            fallbacks: model_ids.map(resolve).collect::<Option<Vec<_>>>()?,
        })
    }
}
