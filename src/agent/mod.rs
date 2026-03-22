pub mod hypothesis;

use crate::knowledge::arxiv::ArxivClient;
use crate::knowledge::database::MetadataStore;
use crate::knowledge::embedding::{build_or_update_index_from_metadata, HashingEmbedder};
use crate::knowledge::paper_parser::PdfParser;
use crate::llm::{LlmClient, LlmRequest};
use crate::output::{save_hypothesis_report, HypothesisReport};
use crate::output::{save_report, AgentRunReport, ReportHit, SavedReport};
use crate::utils::config::Config;
use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Duration;
use tokio::time::timeout;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentMemory {
    pub runs: Vec<AgentRunMemory>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunMemory {
    pub query: String,
    pub created_at: String,
    pub steps: Vec<AgentStepMemory>,
    pub report_path: Option<String>,
    #[serde(default)]
    pub top_hit_ids: Vec<String>,
    pub llm_summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStepMemory {
    pub name: String,
    pub started_at: String,
    pub finished_at: String,
    pub ok: bool,
    pub details: Option<String>,
}

fn load_memory(path: &Path) -> Result<AgentMemory> {
    if !path.exists() {
        return Ok(AgentMemory::default());
    }
    let contents = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&contents)?)
}

fn save_memory(path: &Path, memory: &AgentMemory) -> Result<()> {
    let contents = serde_json::to_string_pretty(memory)?;
    crate::utils::fs::atomic_write(path, contents.as_bytes())
}

#[derive(Debug, Clone)]
enum AgentAction {
    ArxivSearchAndStore,
    DownloadPdfs,
    BuildOrUpdateIndex,
    SemanticRetrieve,
    LlmSummarize,
    Done,
}

#[derive(Debug, Clone, Default)]
struct AgentState {
    papers: Vec<crate::knowledge::arxiv::ArxivPaper>,
    downloaded: usize,
    index_docs: usize,
    hits: Vec<crate::knowledge::embedding::SearchResult>,
    llm_summary: Option<String>,
    index_built: bool,
}

struct AgentPlanner;

impl AgentPlanner {
    fn next_action(state: &AgentState, download_limit: usize) -> AgentAction {
        if state.papers.is_empty() {
            return AgentAction::ArxivSearchAndStore;
        }
        if !state.index_built {
            if state.downloaded < download_limit {
                return AgentAction::DownloadPdfs;
            }
            return AgentAction::BuildOrUpdateIndex;
        }
        if state.hits.is_empty() {
            return AgentAction::SemanticRetrieve;
        }
        if state.llm_summary.is_none() {
            return AgentAction::LlmSummarize;
        }
        AgentAction::Done
    }
}

fn action_name(action: &AgentAction) -> &'static str {
    match action {
        AgentAction::ArxivSearchAndStore => "arxiv_search_and_store",
        AgentAction::DownloadPdfs => "download_pdfs",
        AgentAction::BuildOrUpdateIndex => "build_or_update_index",
        AgentAction::SemanticRetrieve => "semantic_retrieve",
        AgentAction::LlmSummarize => "llm_summarize",
        AgentAction::Done => "done",
    }
}

fn format_prior_context(memory: &AgentMemory, max_runs: usize) -> String {
    let mut s = String::new();

    for run in memory.runs.iter().rev().take(max_runs).rev() {
        if let Some(summary) = run.llm_summary.as_deref() {
            if summary.trim().is_empty() {
                continue;
            }
            s.push_str("Prior run query:\n");
            s.push_str(&run.query);
            s.push_str("\nPrior run summary:\n");
            s.push_str(summary);
            s.push_str("\n\n");
        }
    }

    s
}

pub struct Agent {
    config: Config,
    llm: LlmClient,
    arxiv: ArxivClient,
    metadata_store: MetadataStore,
    pdf_parser: PdfParser,
}

impl Agent {
    pub fn new(config: Config) -> Result<Self> {
        let llm = LlmClient::new(
            config.llm.endpoint.clone(),
            config.llm.api_key.clone(),
            config.llm.model.clone(),
        )?;

        let arxiv = ArxivClient::new_with_max_pdf_bytes(config.knowledge.max_pdf_bytes)?;
        let metadata_store = MetadataStore::new(config.knowledge.metadata_file.clone())?;
        let pdf_parser = PdfParser::new();

        Ok(Self {
            config,
            llm,
            arxiv,
            metadata_store,
            pdf_parser,
        })
    }

    pub async fn run(&mut self, query: &str) -> Result<SavedReport> {
        let timeout_seconds = self.config.agent.timeout_seconds;
        let query = query.to_string();

        let (report, mut run_memory) = timeout(Duration::from_secs(timeout_seconds), async {
            self.run_inner(&query).await
        })
        .await??;

        let step_started = Utc::now().to_rfc3339();
        let saved = save_report(&self.config.paths.output, &report)?;
        let step_finished = Utc::now().to_rfc3339();

        run_memory.steps.push(AgentStepMemory {
            name: "save_report".to_string(),
            started_at: step_started,
            finished_at: step_finished,
            ok: true,
            details: Some(saved.path.to_string_lossy().to_string()),
        });
        run_memory.report_path = Some(saved.path.to_string_lossy().to_string());

        let mut memory = load_memory(&self.config.agent.memory_file)?;
        memory.runs.push(run_memory);
        if memory.runs.len() > self.config.agent.history_limit {
            let keep_from = memory.runs.len() - self.config.agent.history_limit;
            memory.runs = memory.runs.split_off(keep_from);
        }
        save_memory(&self.config.agent.memory_file, &memory)?;

        Ok(saved)
    }

    pub async fn hypothesize(&mut self, query: &str) -> Result<SavedReport> {
        let timeout_seconds = self.config.agent.timeout_seconds;
        let query = query.to_string();

        let saved = timeout(Duration::from_secs(timeout_seconds), async {
            self.hypothesize_inner(&query).await
        })
        .await??;

        Ok(saved)
    }

    async fn hypothesize_inner(&mut self, query: &str) -> Result<SavedReport> {
        let max_results = (self.config.agent.max_iterations as usize).clamp(1, 10);
        let download_limit = self.config.agent.download_limit;
        let top_k = self.config.agent.top_k;

        let papers = self
            .arxiv
            .search_and_store(query, max_results, 0, &mut self.metadata_store)
            .await?;

        for paper in papers.iter().take(download_limit) {
            let result = self
                .arxiv
                .download_pdf(
                    paper,
                    &self.config.paths.papers,
                    Some(&mut self.metadata_store),
                )
                .await;

            if let Err(e) = result {
                warn!("PDF download failed: {}", e);
            }
        }

        let index = build_or_update_index_from_metadata(
            &self.config.knowledge.index_file,
            self.config.knowledge.embedding_dims,
            &self.metadata_store,
            &self.pdf_parser,
        )?;

        let embedder = HashingEmbedder::new(index.dims);
        let q = embedder.embed(query);
        let hits = index.search(&q, top_k);

        let hypotheses = hypothesis::generate_hypotheses(
            &self.llm,
            hypothesis::HypothesisGenConfig {
                max_tokens: self.config.llm.max_tokens,
                temperature: self.config.llm.temperature,
                top_n: top_k,
            },
            query,
            &hits,
            &self.metadata_store,
            &index,
        )
        .await?;

        hypothesis::append_hypotheses(
            &self.config.knowledge.hypotheses_file,
            self.config.knowledge.hypotheses_limit,
            &hypotheses,
        )?;

        let report = HypothesisReport {
            query: query.to_string(),
            top_hits: hits
                .into_iter()
                .map(|h| ReportHit {
                    id: h.id,
                    title: h.title,
                    source_path: h.source_path,
                    score: h.score,
                })
                .collect(),
            hypotheses,
            created_at: Utc::now().to_rfc3339(),
        };

        save_hypothesis_report(&self.config.paths.output, &report)
    }

    async fn run_inner(&mut self, query: &str) -> Result<(AgentRunReport, AgentRunMemory)> {
        info!("Agent run started: {}", query);

        let prior_memory = load_memory(&self.config.agent.memory_file).unwrap_or_default();
        let prior_context = format_prior_context(&prior_memory, 3);

        let mut run_memory = AgentRunMemory {
            query: query.to_string(),
            created_at: Utc::now().to_rfc3339(),
            steps: Vec::new(),
            report_path: None,
            top_hit_ids: Vec::new(),
            llm_summary: None,
        };

        let max_results = (self.config.agent.max_iterations as usize).clamp(1, 10);
        let download_limit = self.config.agent.download_limit;
        let top_k = self.config.agent.top_k;

        let mut state = AgentState::default();

        for _ in 0..self.config.agent.max_iterations {
            let action = AgentPlanner::next_action(&state, download_limit);
            if matches!(action, AgentAction::Done) {
                break;
            }

            let step_name = action_name(&action);
            let step_started = Utc::now().to_rfc3339();

            let result: Result<Option<String>> = match action {
                AgentAction::ArxivSearchAndStore => {
                    let found = self
                        .arxiv
                        .search_and_store(query, max_results, 0, &mut self.metadata_store)
                        .await?;
                    state.papers = found;
                    Ok(Some(format!("papers={}", state.papers.len())))
                }
                AgentAction::DownloadPdfs => {
                    let mut attempted = 0usize;
                    for paper in state.papers.iter().take(download_limit) {
                        attempted += 1;
                        let result = self
                            .arxiv
                            .download_pdf(
                                paper,
                                &self.config.paths.papers,
                                Some(&mut self.metadata_store),
                            )
                            .await;

                        match result {
                            Ok(_) => state.downloaded += 1,
                            Err(e) => warn!("PDF download failed: {}", e),
                        }
                    }
                    Ok(Some(format!(
                        "attempted={} downloaded={}",
                        attempted, state.downloaded
                    )))
                }
                AgentAction::BuildOrUpdateIndex => {
                    let index_path = self.config.knowledge.index_file.clone();
                    let dims = self.config.knowledge.embedding_dims;
                    let index = build_or_update_index_from_metadata(
                        &index_path,
                        dims,
                        &self.metadata_store,
                        &self.pdf_parser,
                    )?;
                    state.index_docs = index.docs.len();
                    state.index_built = true;
                    Ok(Some(format!("indexed={}", state.index_docs)))
                }
                AgentAction::SemanticRetrieve => {
                    let index = crate::knowledge::embedding::VectorIndex::load(
                        &self.config.knowledge.index_file,
                    )?;
                    let embedder = HashingEmbedder::new(index.dims);
                    let q = embedder.embed(query);
                    state.hits = index.search(&q, top_k);
                    Ok(Some(format!("hits={}", state.hits.len())))
                }
                AgentAction::LlmSummarize => {
                    state.llm_summary = self
                        .summarize_with_llm(query, &state.hits, &prior_context)
                        .await
                        .ok();
                    Ok(state
                        .llm_summary
                        .as_ref()
                        .map(|s| format!("chars={}", s.len())))
                }
                AgentAction::Done => Ok(None),
            };

            let step_finished = Utc::now().to_rfc3339();
            let (ok, details) = match result {
                Ok(d) => (true, d),
                Err(e) => (false, Some(e.to_string())),
            };

            run_memory.steps.push(AgentStepMemory {
                name: step_name.to_string(),
                started_at: step_started,
                finished_at: step_finished,
                ok,
                details,
            });

            if !ok {
                break;
            }
        }

        let report = AgentRunReport {
            query: query.to_string(),
            arxiv_results: state.papers.len(),
            downloaded: state.downloaded,
            indexed: state.index_docs,
            top_hits: state
                .hits
                .into_iter()
                .map(|h| ReportHit {
                    id: h.id,
                    title: h.title,
                    source_path: h.source_path,
                    score: h.score,
                })
                .collect(),
            llm_summary: state.llm_summary.clone(),
            created_at: Utc::now().to_rfc3339(),
        };

        run_memory.top_hit_ids = report.top_hits.iter().map(|h| h.id.clone()).collect();
        run_memory.llm_summary = report.llm_summary.clone();
        Ok((report, run_memory))
    }

    async fn summarize_with_llm(
        &self,
        query: &str,
        hits: &[crate::knowledge::embedding::SearchResult],
        prior_context: &str,
    ) -> Result<String> {
        let mut context = String::new();
        if !prior_context.trim().is_empty() {
            context.push_str("Prior context:\n");
            context.push_str(prior_context);
            context.push_str("\n\n");
        }
        context.push_str("Research query:\n");
        context.push_str(query);
        context.push_str("\n\nTop retrieved papers:\n");

        for (i, hit) in hits.iter().enumerate() {
            context.push_str(&format!(
                "{}. {} (id={}, score={:.4})\n",
                i + 1,
                hit.title.as_deref().unwrap_or(""),
                hit.id,
                hit.score
            ));

            if let Some(paper) = self.metadata_store.get_paper(&hit.id) {
                if !paper.abstract_text.trim().is_empty() {
                    context.push_str("Abstract:\n");
                    context.push_str(&paper.abstract_text);
                    context.push('\n');
                }
            }

            context.push('\n');
        }

        let prompt = format!(
            "{}\n\nWrite a concise research summary of the retrieved papers and propose 3 next research steps.",
            context
        );

        let response = self
            .llm
            .generate(LlmRequest {
                prompt,
                max_tokens: self.config.llm.max_tokens,
                temperature: self.config.llm.temperature,
                stop: None,
            })
            .await?;

        Ok(response.text)
    }
}
