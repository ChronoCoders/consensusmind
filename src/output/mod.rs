use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunReport {
    pub query: String,
    pub arxiv_results: usize,
    pub downloaded: usize,
    pub indexed: usize,
    pub top_hits: Vec<ReportHit>,
    pub llm_summary: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportHit {
    pub id: String,
    pub title: Option<String>,
    pub source_path: Option<String>,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct SavedReport {
    pub path: PathBuf,
}

pub fn save_report(output_root: &Path, report: &AgentRunReport) -> Result<SavedReport> {
    let reports_dir = output_root.join("reports");
    fs::create_dir_all(&reports_dir)?;

    let ts = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let filename = format!("report-{}.json", ts);
    let path = reports_dir.join(filename);

    let contents = serde_json::to_string_pretty(report)?;
    crate::utils::fs::atomic_write(&path, contents.as_bytes())?;

    Ok(SavedReport { path })
}
