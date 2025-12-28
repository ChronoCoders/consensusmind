use crate::knowledge::arxiv::ArxivPaper;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use tracing::{debug, info};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperMetadata {
    pub arxiv_id: String,
    pub title: String,
    pub authors: Vec<String>,
    pub abstract_text: String,
    pub published: DateTime<Utc>,
    pub updated: DateTime<Utc>,
    pub categories: Vec<String>,
    pub pdf_path: Option<String>,
    pub downloaded_at: Option<DateTime<Utc>>,
    pub file_size: Option<u64>,
}

impl From<ArxivPaper> for PaperMetadata {
    fn from(paper: ArxivPaper) -> Self {
        let arxiv_id = paper
            .id
            .split('/')
            .next_back()
            .unwrap_or(&paper.id)
            .to_string();

        Self {
            arxiv_id,
            title: paper.title,
            authors: paper.authors,
            abstract_text: paper.abstract_text,
            published: paper.published,
            updated: paper.updated,
            categories: paper.categories,
            pdf_path: None,
            downloaded_at: None,
            file_size: None,
        }
    }
}

pub struct MetadataStore {
    metadata_path: PathBuf,
    papers: HashMap<String, PaperMetadata>,
}

impl MetadataStore {
    pub fn new(metadata_path: PathBuf) -> Result<Self> {
        let mut store = Self {
            metadata_path,
            papers: HashMap::new(),
        };
        store.load()?;
        Ok(store)
    }

    pub fn add_paper(&mut self, mut metadata: PaperMetadata) -> Result<()> {
        let arxiv_id = metadata.arxiv_id.clone();

        if let Some(existing) = self.papers.get(&arxiv_id) {
            if existing.pdf_path.is_some() {
                metadata.pdf_path = existing.pdf_path.clone();
                metadata.downloaded_at = existing.downloaded_at;
                metadata.file_size = existing.file_size;
            }
        }

        self.papers.insert(arxiv_id, metadata);
        self.save()?;
        Ok(())
    }

    pub fn mark_downloaded(
        &mut self,
        arxiv_id: &str,
        pdf_path: &str,
        file_size: u64,
    ) -> Result<()> {
        let normalized_id = arxiv_id.split('/').next_back().unwrap_or(arxiv_id);

        if let Some(metadata) = self.papers.get_mut(normalized_id) {
            metadata.pdf_path = Some(pdf_path.to_string());
            metadata.downloaded_at = Some(Utc::now());
            metadata.file_size = Some(file_size);
            self.save()?;
            info!("Marked paper as downloaded: {}", normalized_id);
        }
        Ok(())
    }

    pub fn get_paper(&self, arxiv_id: &str) -> Option<&PaperMetadata> {
        let normalized_id = arxiv_id.split('/').next_back().unwrap_or(arxiv_id);
        self.papers.get(normalized_id)
    }

    pub fn is_downloaded(&self, arxiv_id: &str) -> bool {
        let normalized_id = arxiv_id.split('/').next_back().unwrap_or(arxiv_id);
        self.papers
            .get(normalized_id)
            .and_then(|p| p.pdf_path.as_ref())
            .is_some()
    }

    pub fn list_papers(&self) -> Vec<&PaperMetadata> {
        self.papers.values().collect()
    }

    pub fn count(&self) -> usize {
        self.papers.len()
    }

    pub fn count_downloaded(&self) -> usize {
        self.papers
            .values()
            .filter(|p| p.pdf_path.is_some())
            .count()
    }

    fn load(&mut self) -> Result<()> {
        if !self.metadata_path.exists() {
            debug!("Metadata file does not exist, starting fresh");
            return Ok(());
        }

        let contents = fs::read_to_string(&self.metadata_path)?;
        self.papers = serde_json::from_str(&contents)?;
        info!("Loaded {} papers from metadata store", self.papers.len());
        Ok(())
    }

    fn save(&self) -> Result<()> {
        if let Some(parent) = self.metadata_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let contents = serde_json::to_string_pretty(&self.papers)?;
        fs::write(&self.metadata_path, contents)?;
        debug!("Saved metadata for {} papers", self.papers.len());
        Ok(())
    }
}
