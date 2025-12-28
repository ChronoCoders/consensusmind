use anyhow::Result;
use pdf_extract::extract_text;
use std::path::Path;
use thiserror::Error;
use tracing::{debug, info, warn};

#[derive(Debug, Error)]
pub enum ParserError {
    #[error("Failed to extract text from PDF: {0}")]
    ExtractionFailed(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("PDF file not found: {0}")]
    FileNotFound(String),
}

pub struct PdfParser;

impl PdfParser {
    pub fn new() -> Self {
        Self
    }

    pub fn extract_text(&self, pdf_path: &Path) -> Result<String, ParserError> {
        if !pdf_path.exists() {
            return Err(ParserError::FileNotFound(pdf_path.display().to_string()));
        }

        info!("Extracting text from PDF: {}", pdf_path.display());

        let text =
            extract_text(pdf_path).map_err(|e| ParserError::ExtractionFailed(e.to_string()))?;

        let text_len = text.len();
        debug!("Extracted {} characters from PDF", text_len);

        if text.trim().is_empty() {
            warn!(
                "PDF appears to be empty or text extraction failed: {}",
                pdf_path.display()
            );
        }

        Ok(text)
    }

    pub fn extract_metadata(&self, text: &str) -> PaperTextMetadata {
        let lines: Vec<&str> = text.lines().collect();
        let char_count = text.len();
        let word_count = text.split_whitespace().count();
        let line_count = lines.len();

        let title = self.extract_title(&lines);
        let abstract_start = self.find_abstract_start(&lines);

        PaperTextMetadata {
            char_count,
            word_count,
            line_count,
            title,
            abstract_start,
        }
    }

    fn extract_title(&self, lines: &[&str]) -> Option<String> {
        for line in lines.iter().take(20) {
            let trimmed = line.trim();
            if !trimmed.is_empty() && trimmed.len() > 10 && trimmed.len() < 200 {
                return Some(trimmed.to_string());
            }
        }
        None
    }

    fn find_abstract_start(&self, lines: &[&str]) -> Option<usize> {
        for (i, line) in lines.iter().enumerate() {
            let lower = line.to_lowercase();
            if lower.contains("abstract") {
                return Some(i);
            }
        }
        None
    }
}

impl Default for PdfParser {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct PaperTextMetadata {
    pub char_count: usize,
    pub word_count: usize,
    pub line_count: usize,
    pub title: Option<String>,
    pub abstract_start: Option<usize>,
}
