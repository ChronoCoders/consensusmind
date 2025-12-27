use anyhow::Result;
use consensusmind::utils::{config::Config, logger};
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::load()?;

    logger::init_with_file(&config.logging.file, &config.logging.level)?;

    info!("ConsensusMind starting...");
    info!("Configuration loaded successfully");
    info!("LLM endpoint: {}", config.llm.endpoint);
    info!("LLM model: {}", config.llm.model);

    Ok(())
}
