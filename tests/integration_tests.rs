use anyhow::Result;
use consensusmind::knowledge::arxiv::ArxivClient;
use consensusmind::knowledge::database::MetadataStore;
use consensusmind::knowledge::paper_parser::PdfParser;
use consensusmind::llm::{LlmClient, LlmRequest};
use consensusmind::utils::config::Config;
use std::path::PathBuf;

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
    let _client = LlmClient::new(config.llm.endpoint, config.llm.api_key, config.llm.model)?;
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

#[tokio::test]
#[ignore]
async fn test_arxiv_download_pdf() -> Result<()> {
    let client = ArxivClient::new()?;
    let papers = client
        .search("blockchain consensus", 1, 0)
        .await
        .expect("Search failed");

    assert!(!papers.is_empty(), "Should find at least one paper");

    let paper = &papers[0];
    let output_dir = PathBuf::from("data/papers");

    let filepath = client
        .download_pdf(paper, &output_dir, None)
        .await
        .expect("Download failed");

    assert!(!filepath.is_empty(), "Filepath should not be empty");
    assert!(PathBuf::from(&filepath).exists(), "PDF file should exist");

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_metadata_store() -> Result<()> {
    let client = ArxivClient::new()?;
    let metadata_path = PathBuf::from("data/test_metadata.json");

    if metadata_path.exists() {
        std::fs::remove_file(&metadata_path)?;
    }

    let mut store = MetadataStore::new(metadata_path.clone())?;

    let papers = client
        .search_and_store("blockchain consensus", 3, 0, &mut store)
        .await
        .expect("Search and store failed");

    assert_eq!(papers.len(), 3, "Should find 3 papers");
    assert_eq!(store.count(), 3, "Store should have 3 papers");
    assert_eq!(store.count_downloaded(), 0, "No papers downloaded yet");

    let paper = &papers[0];
    let output_dir = PathBuf::from("data/papers");

    let arxiv_id = paper.id.split('/').next_back().unwrap_or(&paper.id);
    let filename = format!("{}.pdf", arxiv_id.replace(':', "_"));
    let test_pdf_path = output_dir.join(&filename);

    if test_pdf_path.exists() {
        std::fs::remove_file(&test_pdf_path)?;
    }

    let filepath = client
        .download_pdf(paper, &output_dir, Some(&mut store))
        .await
        .expect("Download with metadata failed");

    assert!(PathBuf::from(&filepath).exists(), "PDF file should exist");
    assert_eq!(
        store.count_downloaded(),
        1,
        "One paper should be marked downloaded"
    );

    assert!(
        store.is_downloaded(arxiv_id),
        "Paper should be marked as downloaded"
    );

    let retrieved = store.get_paper(arxiv_id);
    assert!(retrieved.is_some(), "Should retrieve paper metadata");
    assert!(
        retrieved.unwrap().pdf_path.is_some(),
        "PDF path should be set"
    );

    std::fs::remove_file(&metadata_path)?;
    std::fs::remove_file(&test_pdf_path)?;

    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_pdf_text_extraction() -> Result<()> {
    let client = ArxivClient::new()?;
    let papers = client
        .search("blockchain consensus", 1, 0)
        .await
        .expect("Search failed");

    assert!(!papers.is_empty(), "Should find at least one paper");

    let paper = &papers[0];
    let output_dir = PathBuf::from("data/papers");

    let filepath = client
        .download_pdf(paper, &output_dir, None)
        .await
        .expect("Download failed");

    let parser = PdfParser::new();
    let text = parser
        .extract_text(&PathBuf::from(&filepath))
        .expect("Text extraction failed");

    assert!(!text.is_empty(), "Extracted text should not be empty");
    assert!(text.len() > 100, "Should extract substantial text");

    let metadata = parser.extract_metadata(&text);
    assert!(metadata.word_count > 0, "Should have word count");
    assert!(metadata.char_count > 0, "Should have character count");

    println!(
        "Extracted {} words, {} characters",
        metadata.word_count, metadata.char_count
    );
    if let Some(title) = metadata.title {
        println!("Detected title: {}", title);
    }

    Ok(())
}
