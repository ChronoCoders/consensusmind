use anyhow::Result;
use consensusmind::knowledge::arxiv::ArxivClient;
use consensusmind::llm::{LlmClient, LlmRequest};
use consensusmind::utils::config::Config;

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

#[tokio::test]
async fn test_arxiv_client_creation() -> Result<()> {
    let _client = ArxivClient::new()?;
    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_arxiv_search_consensus() -> Result<()> {
    let client = ArxivClient::new()?;
    let papers = client
        .search("blockchain consensus", 5, 0)
        .await
        .expect("Search failed");

    assert!(!papers.is_empty(), "Should find papers");
    assert!(papers.len() <= 5, "Should respect max_results");

    for paper in &papers {
        assert!(!paper.id.is_empty(), "Paper ID should not be empty");
        assert!(!paper.title.is_empty(), "Paper title should not be empty");
        assert!(!paper.pdf_url.is_empty(), "PDF URL should not be empty");
    }

    Ok(())
}
