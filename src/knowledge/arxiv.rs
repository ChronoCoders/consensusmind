use anyhow::Result;
use chrono::{DateTime, Utc};
use quick_xml::events::Event;
use quick_xml::Reader;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Duration;
use thiserror::Error;
use tokio::time::sleep;
use tracing::{debug, info, warn};

#[derive(Debug, Error)]
pub enum ArxivError {
    #[error("HTTP request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("XML parsing failed: {0}")]
    XmlParseFailed(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Download failed: {0}")]
    DownloadFailed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArxivPaper {
    pub id: String,
    pub title: String,
    pub authors: Vec<String>,
    pub abstract_text: String,
    pub published: DateTime<Utc>,
    pub updated: DateTime<Utc>,
    pub pdf_url: String,
    pub categories: Vec<String>,
}

pub struct ArxivClient {
    client: Client,
    base_url: String,
    rate_limit_delay: Duration,
}

impl ArxivClient {
    pub fn new() -> Result<Self> {
        let client = Client::builder().timeout(Duration::from_secs(30)).build()?;

        Ok(Self {
            client,
            base_url: "https://export.arxiv.org/api/query".to_string(),
            rate_limit_delay: Duration::from_secs(3),
        })
    }

    pub async fn search(
        &self,
        query: &str,
        max_results: usize,
        start: usize,
    ) -> Result<Vec<ArxivPaper>, ArxivError> {
        info!(
            "Searching arXiv: query='{}', max_results={}, start={}",
            query, max_results, start
        );

        let url = format!(
            "{}?search_query={}&start={}&max_results={}",
            self.base_url,
            urlencoding::encode(query),
            start,
            max_results
        );

        debug!("arXiv API URL: {}", url);

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(ArxivError::RequestFailed(
                response.error_for_status().unwrap_err(),
            ));
        }

        let xml_text = response.text().await?;
        debug!("Received XML response: {} bytes", xml_text.len());

        let papers = self.parse_response(&xml_text)?;
        info!("Parsed {} papers from arXiv response", papers.len());

        sleep(self.rate_limit_delay).await;

        Ok(papers)
    }

    pub async fn download_pdf(
        &self,
        paper: &ArxivPaper,
        output_dir: &Path,
    ) -> Result<String, ArxivError> {
        fs::create_dir_all(output_dir)?;

        let arxiv_id = paper.id.split('/').next_back().unwrap_or(&paper.id);
        let filename = format!("{}.pdf", arxiv_id.replace(':', "_"));
        let filepath = output_dir.join(&filename);

        if filepath.exists() {
            info!("PDF already exists: {}", filepath.display());
            return Ok(filepath.to_string_lossy().to_string());
        }

        info!(
            "Downloading PDF: {} -> {}",
            paper.pdf_url,
            filepath.display()
        );

        let response = self.client.get(&paper.pdf_url).send().await?;

        if !response.status().is_success() {
            return Err(ArxivError::DownloadFailed(format!(
                "HTTP {}: {}",
                response.status(),
                paper.pdf_url
            )));
        }

        let bytes = response.bytes().await?;
        fs::write(&filepath, bytes)?;

        info!(
            "Downloaded PDF: {} ({} bytes)",
            filepath.display(),
            fs::metadata(&filepath)?.len()
        );

        sleep(self.rate_limit_delay).await;

        Ok(filepath.to_string_lossy().to_string())
    }

    fn parse_response(&self, xml: &str) -> Result<Vec<ArxivPaper>, ArxivError> {
        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(true);

        let mut papers = Vec::new();
        let mut buf = Vec::new();

        let mut in_entry = false;
        let mut current_paper: Option<ArxivPaper> = None;
        let mut current_tag = String::new();
        let mut current_text = String::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                    current_tag = name.clone();

                    if name == "entry" {
                        in_entry = true;
                        current_paper = Some(ArxivPaper {
                            id: String::new(),
                            title: String::new(),
                            authors: Vec::new(),
                            abstract_text: String::new(),
                            published: Utc::now(),
                            updated: Utc::now(),
                            pdf_url: String::new(),
                            categories: Vec::new(),
                        });
                    }
                }
                Ok(Event::Text(e)) => {
                    if in_entry {
                        current_text = e.unescape().unwrap_or_default().trim().to_string();
                    }
                }
                Ok(Event::End(ref e)) => {
                    let name = String::from_utf8_lossy(e.name().as_ref()).to_string();

                    if in_entry {
                        if let Some(ref mut paper) = current_paper {
                            match current_tag.as_str() {
                                "id" => paper.id = current_text.clone(),
                                "title" => paper.title = current_text.clone(),
                                "summary" => paper.abstract_text = current_text.clone(),
                                "published" => {
                                    if let Ok(dt) = DateTime::parse_from_rfc3339(&current_text) {
                                        paper.published = dt.with_timezone(&Utc);
                                    }
                                }
                                "updated" => {
                                    if let Ok(dt) = DateTime::parse_from_rfc3339(&current_text) {
                                        paper.updated = dt.with_timezone(&Utc);
                                    }
                                }
                                "name" => paper.authors.push(current_text.clone()),
                                _ => {}
                            }
                        }

                        if name == "entry" {
                            in_entry = false;
                            if let Some(mut paper) = current_paper.take() {
                                paper.pdf_url =
                                    format!("{}.pdf", paper.id.replace("/abs/", "/pdf/"));
                                papers.push(paper);
                            }
                        }
                    }

                    current_text.clear();
                }
                Ok(Event::Eof) => break,
                Err(e) => {
                    warn!("XML parsing error: {}", e);
                    return Err(ArxivError::XmlParseFailed(e.to_string()));
                }
                _ => {}
            }
            buf.clear();
        }

        Ok(papers)
    }
}
