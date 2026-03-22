use crate::knowledge::database::MetadataStore;
use crate::knowledge::embedding::{HashingEmbedder, SearchResult, VectorIndex};
use crate::llm::{LlmClient, LlmRequest};
use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypothesis {
    pub id: String,
    pub title: String,
    pub description: String,
    pub mechanism: String,
    pub evaluation_plan: Vec<String>,
    pub related_paper_ids: Vec<String>,
    pub novelty_score: f32,
    pub feasibility_score: f32,
    pub created_at: String,
}

#[derive(Debug, Clone, Copy)]
pub struct HypothesisGenConfig {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_n: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LlmHypothesis {
    title: String,
    description: String,
    mechanism: String,
    evaluation_plan: Vec<String>,
    related_paper_ids: Vec<String>,
    feasibility_score: Option<f32>,
}

pub async fn generate_hypotheses(
    llm: &LlmClient,
    config: HypothesisGenConfig,
    query: &str,
    hits: &[SearchResult],
    metadata_store: &MetadataStore,
    index: &VectorIndex,
) -> Result<Vec<Hypothesis>> {
    let prompt = build_prompt(query, hits, metadata_store, config.top_n);

    let response = llm
        .generate(LlmRequest {
            prompt,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            stop: None,
        })
        .await?;

    let raw = response.text;
    let parsed: Vec<LlmHypothesis> =
        parse_llm_json_array(&raw).with_context(|| "Failed to parse LLM hypothesis JSON array")?;

    let embedder = HashingEmbedder::new(index.dims);
    let created_at = Utc::now().to_rfc3339();

    let mut out = Vec::new();
    for (i, h) in parsed.into_iter().enumerate() {
        let id = format!("hyp-{}-{}", Utc::now().format("%Y%m%dT%H%M%SZ"), i + 1);
        let novelty_score = novelty_score(&embedder, index, &h);
        let feasibility_score = h.feasibility_score.unwrap_or(0.6).clamp(0.0, 1.0);

        out.push(Hypothesis {
            id,
            title: h.title,
            description: h.description,
            mechanism: h.mechanism,
            evaluation_plan: h.evaluation_plan,
            related_paper_ids: h.related_paper_ids,
            novelty_score,
            feasibility_score,
            created_at: created_at.clone(),
        });
    }

    out.sort_by(|a, b| {
        (b.novelty_score + b.feasibility_score)
            .partial_cmp(&(a.novelty_score + a.feasibility_score))
            .unwrap_or(Ordering::Equal)
    });

    Ok(out)
}

pub fn append_hypotheses(path: &Path, limit: usize, new_items: &[Hypothesis]) -> Result<()> {
    let mut all: Vec<Hypothesis> = if path.exists() {
        let contents = fs::read_to_string(path)?;
        serde_json::from_str(&contents).unwrap_or_default()
    } else {
        Vec::new()
    };

    all.extend_from_slice(new_items);
    if all.len() > limit {
        let keep_from = all.len() - limit;
        all = all.split_off(keep_from);
    }

    let contents = serde_json::to_string_pretty(&all)?;
    crate::utils::fs::atomic_write(path, contents.as_bytes())?;
    Ok(())
}

fn build_prompt(
    query: &str,
    hits: &[SearchResult],
    metadata_store: &MetadataStore,
    top_n: usize,
) -> String {
    let mut context = String::new();
    context.push_str("You are a researcher in blockchain consensus protocols.\n");
    context.push_str("Return ONLY valid JSON, no prose.\n\n");
    context.push_str("Task: propose novel consensus mechanism hypotheses.\n");
    context.push_str("Constraints:\n");
    context.push_str("- Output a JSON array of objects.\n");
    context.push_str("- Each object must have: title, description, mechanism, evaluation_plan, related_paper_ids, feasibility_score.\n");
    context.push_str("- feasibility_score is a float in [0,1].\n\n");
    context.push_str("Research query:\n");
    context.push_str(query);
    context.push_str("\n\nTop retrieved papers:\n");

    for hit in hits.iter().take(top_n) {
        context.push_str(&format!(
            "- id={} score={:.4} title={}\n",
            hit.id,
            hit.score,
            hit.title.as_deref().unwrap_or("")
        ));
        if let Some(paper) = metadata_store.get_paper(&hit.id) {
            if !paper.abstract_text.trim().is_empty() {
                context.push_str("  abstract: ");
                context.push_str(&truncate(&paper.abstract_text, 800));
                context.push('\n');
            }
        }
    }

    context.push_str("\nJSON schema example:\n");
    context.push_str(
        r#"[{"title":"...","description":"...","mechanism":"...","evaluation_plan":["..."],"related_paper_ids":["..."],"feasibility_score":0.7}]"#,
    );
    context
}

fn truncate(s: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i >= max_chars {
            break;
        }
        out.push(c);
    }
    out
}

fn parse_llm_json_array<T: for<'de> Deserialize<'de>>(text: &str) -> Result<T> {
    if let Ok(v) = serde_json::from_str::<T>(text) {
        return Ok(v);
    }

    let start = text.find('[').context("No '[' found in LLM output")?;
    let end = text.rfind(']').context("No ']' found in LLM output")?;
    let slice = &text[start..=end];
    Ok(serde_json::from_str::<T>(slice)?)
}

fn novelty_score(embedder: &HashingEmbedder, index: &VectorIndex, h: &LlmHypothesis) -> f32 {
    let mut text = String::new();
    text.push_str(&h.title);
    text.push('\n');
    text.push_str(&h.description);
    text.push('\n');
    text.push_str(&h.mechanism);

    let q = embedder.embed(&text);
    let results = index.search(&q, 10);
    let max_sim = results.iter().map(|r| r.score).fold(-1.0f32, f32::max);
    (1.0 - max_sim.clamp(-1.0, 1.0)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_llm_json_array_extracts_bracketed_json() {
        let s = "some preface\n[{\"title\":\"t\",\"description\":\"d\",\"mechanism\":\"m\",\"evaluation_plan\":[],\"related_paper_ids\":[],\"feasibility_score\":0.5}]\ntrailing";
        let v: Vec<LlmHypothesis> = parse_llm_json_array(s).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].title, "t");
    }
}
