use anyhow::Result;
use consensusmind::llm::LlmClient;
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

    let _client = LlmClient::new(
        config.llm.endpoint.clone(),
        config.llm.api_key.clone(),
        config.llm.model.clone(),
    )?;

    info!("LLM client initialized successfully");
    info!("System ready");

    Ok(())
}
