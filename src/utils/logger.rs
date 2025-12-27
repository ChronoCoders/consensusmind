use anyhow::Result;
use std::fs::OpenOptions;
use std::path::Path;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

pub fn init() -> Result<()> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let stdout_layer = fmt::layer()
        .with_target(false)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .with_filter(filter.clone());

    tracing_subscriber::registry().with(stdout_layer).init();

    Ok(())
}

pub fn init_with_file(log_file: &Path, level: &str) -> Result<()> {
    let filter = EnvFilter::new(level);

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_file)?;

    let file_layer = fmt::layer()
        .with_writer(std::sync::Arc::new(file))
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .with_ansi(false)
        .with_filter(filter.clone());

    let stdout_layer = fmt::layer()
        .with_target(false)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .with_filter(filter);

    tracing_subscriber::registry()
        .with(file_layer)
        .with(stdout_layer)
        .init();

    Ok(())
}
