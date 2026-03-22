use crate::agent::experiment::ExperimentRun;
use crate::agent::hypothesis::Hypothesis;
use crate::knowledge::database::MetadataStore;
use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedWhitepaper {
    pub md_path: PathBuf,
}

pub fn write_whitepaper_md(
    output_root: &Path,
    hypothesis: &Hypothesis,
    experiment_run: &ExperimentRun,
    metadata_store: &MetadataStore,
) -> Result<SavedWhitepaper> {
    let papers_dir = output_root.join("papers");
    fs::create_dir_all(&papers_dir)?;

    let ts = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let filename = format!("whitepaper-{}-{}.md", sanitize_filename(&hypothesis.id), ts);
    let md_path = papers_dir.join(filename);

    let md = render_md(hypothesis, experiment_run, metadata_store)?;
    crate::utils::fs::atomic_write(&md_path, md.as_bytes())?;

    Ok(SavedWhitepaper { md_path })
}

pub fn load_experiment_results(
    experiments_root: &Path,
    hypothesis_id: &str,
) -> Result<ExperimentRun> {
    let path = experiments_root.join(hypothesis_id).join("results.json");
    let contents = fs::read_to_string(&path)
        .with_context(|| format!("Failed to read experiment results: {}", path.display()))?;
    Ok(serde_json::from_str(&contents)?)
}

fn render_md(
    hypothesis: &Hypothesis,
    experiment_run: &ExperimentRun,
    metadata_store: &MetadataStore,
) -> Result<String> {
    let mut s = String::new();
    s.push_str("# ");
    s.push_str(&hypothesis.title);
    s.push('\n');
    s.push('\n');

    s.push_str("## Abstract\n");
    s.push_str(&hypothesis.description);
    s.push('\n');
    s.push('\n');

    s.push_str("## Hypothesis\n");
    s.push_str("**Mechanism.** ");
    s.push_str(&hypothesis.mechanism);
    s.push('\n');
    s.push('\n');

    if !hypothesis.evaluation_plan.is_empty() {
        s.push_str("## Evaluation Plan\n");
        for (i, step) in hypothesis.evaluation_plan.iter().enumerate() {
            s.push_str(&format!("{}. {}\n", i + 1, step));
        }
        s.push('\n');
    }

    s.push_str("## Experiments\n");
    s.push_str("Experiments are executed as parameter sweeps over simulation configurations with multiple random seeds. This report summarizes aggregate metrics.\n\n");

    if !experiment_run.raft.is_empty() {
        let mut raft_ranked = experiment_run.raft.clone();
        raft_ranked.sort_by(|a, b| {
            b.aggregate
                .mean_commit_rate_per_tick
                .partial_cmp(&a.aggregate.mean_commit_rate_per_tick)
                .unwrap_or(Ordering::Equal)
        });

        s.push_str("### Raft-style Simulator (Top Cases)\n");
        for case in raft_ranked.iter().take(5) {
            s.push_str(&format!(
                "- nodes={} ticks={} p_req={:.2} mean_commit_rate_per_tick={:.6} mean_elections={:.2} mean_leader_changes={:.2}\n",
                case.params.nodes,
                case.params.ticks,
                case.params.client_request_prob,
                case.aggregate.mean_commit_rate_per_tick,
                case.aggregate.mean_elections,
                case.aggregate.mean_leader_changes
            ));
        }
        s.push('\n');
    }

    if !experiment_run.leader.is_empty() {
        s.push_str("### Leader-failure Baseline\n");
        for case in experiment_run.leader.iter() {
            s.push_str(&format!(
                "- p_fail={:.2} mean_commit_rate={:.4}\n",
                case.params.leader_failure_prob, case.aggregate.mean_commit_rate
            ));
        }
        s.push('\n');
    }

    s.push_str("## Related Work\n");
    if hypothesis.related_paper_ids.is_empty() {
        s.push_str("No related papers were explicitly linked to this hypothesis.\n\n");
    } else {
        for paper_id in hypothesis.related_paper_ids.iter() {
            let title = metadata_store
                .get_paper(paper_id)
                .map(|p| p.title.clone())
                .unwrap_or_else(|| paper_id.to_string());
            s.push_str(&format!(
                "- {} (arXiv: {}) https://arxiv.org/abs/{}\n",
                title, paper_id, paper_id
            ));
        }
        s.push('\n');
    }

    s.push_str("## Reproducibility\n");
    s.push_str("- Results file: ");
    s.push_str(&format!(
        "`data/experiments/{}/results.json`\n",
        hypothesis.id
    ));
    s.push_str("- Generated at: ");
    s.push_str(&Utc::now().to_rfc3339());
    s.push('\n');

    Ok(s)
}

fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::experiment::{run_experiments, ExperimentOverrides};

    fn temp_path(name: &str) -> PathBuf {
        let pid = std::process::id();
        let ts = Utc::now().timestamp_nanos_opt().unwrap_or(0);
        std::env::temp_dir().join(format!("consensusmind-whitepaper-{}-{}-{}", name, pid, ts))
    }

    #[test]
    fn write_whitepaper_creates_md_file() {
        let root = temp_path("write");
        fs::create_dir_all(&root).unwrap();

        let hypotheses_path = root.join("hypotheses.json");
        let metadata_path = root.join("metadata.json");
        let output_root = root.join("out");
        fs::create_dir_all(&output_root).unwrap();

        let h = Hypothesis {
            id: "h1".to_string(),
            title: "Whitepaper Title".to_string(),
            description: "desc".to_string(),
            mechanism: "mech".to_string(),
            evaluation_plan: vec!["plan".to_string()],
            related_paper_ids: vec!["1234.56789".to_string()],
            novelty_score: 0.2,
            feasibility_score: 0.8,
            created_at: Utc::now().to_rfc3339(),
        };
        let h_contents = serde_json::to_string_pretty(&vec![h.clone()]).unwrap();
        crate::utils::fs::atomic_write(&hypotheses_path, h_contents.as_bytes()).unwrap();

        crate::utils::fs::atomic_write(&metadata_path, "{}".as_bytes()).unwrap();
        let store = MetadataStore::new(metadata_path).unwrap();

        let (_hyp, run, _results_path) = run_experiments(
            &root,
            &hypotheses_path,
            "h1",
            ExperimentOverrides {
                seeds: Some(2),
                ticks: Some(200),
                nodes: Some(3),
            },
        )
        .unwrap();

        let saved = write_whitepaper_md(&output_root, &h, &run, &store).unwrap();
        assert!(saved.md_path.exists());
        let contents = fs::read_to_string(saved.md_path).unwrap();
        assert!(contents.contains("# Whitepaper Title"));
        assert!(contents.contains("## Experiments"));
    }
}
