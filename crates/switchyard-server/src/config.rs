// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compatibility wrapper for loading shared runner configuration.

use std::path::Path;

use switchyard_runner::Runner;

use crate::{ServerError, ServerResult, ServerState};

/// Loads a TOML deployment file and constructs the complete server state.
pub fn load_server_state(path: impl AsRef<Path>) -> ServerResult<ServerState> {
    ServerState::from_runner(Runner::load(path).map_err(ServerError::from)?)
}

/// Loads a TOML deployment file, first discovering the served model id for each
/// `detect_model` target. A target whose backend cannot be reached keeps its
/// configured id.
pub async fn load_server_state_detected(path: impl AsRef<Path>) -> ServerResult<ServerState> {
    ServerState::from_runner(
        Runner::load_detected(path)
            .await
            .map_err(ServerError::from)?,
    )
}
