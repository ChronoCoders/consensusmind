use anyhow::Result;
use consensusmind::utils::config::Config;
use consensusmind::utils::logger;
use consensusmind::llm::{LlmClient, LlmRequest};

#[tokio::test]
async fn test_config_loads() -> Result<()> {
    let config = Config::load()?;
    assert!(!config.llm.endpoint.is_empty());
    assert!(!config.llm.model.is_empty());
    Ok(())
}

#[tokio::test]
async fn test_llm_client_creation() -> Result<()> {
    let config = Config::load()?;
    let _client = LlmClient::new(
        config.llm.endpoint,
        config.llm.api_key,
        config.llm.model,
    )?;
    Ok(())
}

#[tokio::test]
async fn test_llm_request_creation() {
    let request = LlmRequest {
        prompt: "Test prompt".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        stop: None,
    };
    assert_eq!(request.prompt, "Test prompt");
    assert_eq!(request.max_tokens, 100);
}
