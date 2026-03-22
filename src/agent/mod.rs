use crate::knowledge::arxiv::ArxivClient;
use crate::knowledge::database::MetadataStore;
use crate::knowledge::embedding::{build_or_update_index_from_metadata, HashingEmbedder};
use crate::knowledge::paper_parser::PdfParser;
use crate::llm::{LlmClient, LlmRequest};
use crate::output::{save_report, AgentRunReport, ReportHit, SavedReport};
use crate::utils::config::Config;
use anyhow::Result;
use chrono::Utc;
use std::time::Duration;
use tokio::time::timeout;
use tracing::{info, warn};

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

        let report = timeout(Duration::from_secs(timeout_seconds), async {
            self.run_inner(&query).await
        })
        .await??;

        save_report(&self.config.paths.output, &report)
    }

    async fn run_inner(&mut self, query: &str) -> Result<AgentRunReport> {
        info!("Agent run started: {}", query);

        let max_results = (self.config.agent.max_iterations as usize).clamp(1, 10);
        let download_limit = max_results.min(3);

        let papers = self
            .arxiv
            .search_and_store(query, max_results, 0, &mut self.metadata_store)
            .await?;

        let mut downloaded = 0usize;
        for paper in papers.iter().take(download_limit) {
            let result = self
                .arxiv
                .download_pdf(
                    paper,
                    &self.config.paths.papers,
                    Some(&mut self.metadata_store),
                )
                .await;

            match result {
                Ok(_) => downloaded += 1,
                Err(e) => warn!("PDF download failed: {}", e),
            }
        }

        let index_path = self.config.knowledge.index_file.clone();
        let dims = self.config.knowledge.embedding_dims;

        let index = build_or_update_index_from_metadata(
            &index_path,
            dims,
            &self.metadata_store,
            &self.pdf_parser,
        )?;

        let embedder = HashingEmbedder::new(index.dims);
        let q = embedder.embed(query);
        let top_hits = index.search(&q, 5);

        let llm_summary = self.summarize_with_llm(query, &top_hits).await.ok();

        Ok(AgentRunReport {
            query: query.to_string(),
            arxiv_results: papers.len(),
            downloaded,
            indexed: index.docs.len(),
            top_hits: top_hits
                .into_iter()
                .map(|h| ReportHit {
                    id: h.id,
                    title: h.title,
                    source_path: h.source_path,
                    score: h.score,
                })
                .collect(),
            llm_summary,
            created_at: Utc::now().to_rfc3339(),
        })
    }

    async fn summarize_with_llm(
        &self,
        query: &str,
        hits: &[crate::knowledge::embedding::SearchResult],
    ) -> Result<String> {
        let mut context = String::new();
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
