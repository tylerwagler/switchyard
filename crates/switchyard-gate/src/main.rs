// SPDX-License-Identifier: Apache-2.0

//! Binary entrypoint for `switchyard-gate`: the Switchyard server with the gate in front.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use switchyard_gate::keys::KeyStore;
use switchyard_gate::{Gate, layer};
use switchyard_server::config::load_server_state;
use switchyard_server::{
    DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT, DEFAULT_LISTEN_BACKLOG, ServerRunOptions, run_server_with,
};

#[derive(Debug, Parser)]
#[command(
    name = "switchyard-gate",
    about = "Serve Switchyard routes behind API-key auth, quotas, and metering",
    version
)]
struct Args {
    /// TOML file defining LLM clients, targets, and algorithm routes.
    #[arg(long, value_name = "PATH")]
    config: PathBuf,

    /// Host address to bind.
    #[arg(long, default_value_t = IpAddr::V4(Ipv4Addr::UNSPECIFIED))]
    host: IpAddr,

    /// Port to bind.
    #[arg(short, long, default_value_t = 4000)]
    port: u16,

    /// Maximum time active requests may drain during shutdown.
    #[arg(long, default_value_t = humantime::Duration::from(DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT))]
    shutdown_timeout: humantime::Duration,

    /// Check the config and print the served models, then exit without connecting to
    /// Postgres or Valkey.
    #[arg(long)]
    dry_run: bool,

    /// Portal Postgres URL for the `switchyard_gate` role.
    #[arg(long, env = "GATE_DATABASE_URL", hide_env_values = true)]
    database_url: String,

    /// Valkey URL for quota counters and the usage stream.
    #[arg(long, env = "GATE_VALKEY_URL", hide_env_values = true)]
    valkey_url: String,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> ExitCode {
    if let Err(error) = switchyard_server::initialize_observability() {
        eprintln!("failed to initialize observability: {error}");
        return ExitCode::FAILURE;
    }
    let result = run(Args::parse()).await;
    switchyard_server::flush_observability();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(args: Args) -> Result<(), String> {
    let state = load_server_state(&args.config).map_err(|e| e.to_string())?;
    let options = ServerRunOptions {
        addr: SocketAddr::new(args.host, args.port),
        backlog: DEFAULT_LISTEN_BACKLOG,
        dry_run: args.dry_run,
        shutdown_timeout: args.shutdown_timeout.into(),
        tls: None,
    };
    if args.dry_run {
        return run_server_with(state, options, |router| router)
            .await
            .map_err(|e| e.to_string());
    }
    let client =
        redis::Client::open(args.valkey_url).map_err(|e| format!("invalid valkey url: {e}"))?;
    let valkey = redis::aio::ConnectionManager::new(client)
        .await
        .map_err(|e| format!("cannot connect to valkey: {e}"))?;
    let gate = Arc::new(Gate {
        keys: KeyStore::new(args.database_url),
        valkey,
    });
    run_server_with(state, options, |router| layer(router, gate))
        .await
        .map_err(|e| e.to_string())
}
