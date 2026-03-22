use crate::agent::experiment::ExperimentRun;
use crate::agent::hypothesis::Hypothesis;
use crate::knowledge::database::MetadataStore;
use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedPaper {
    pub tex_path: PathBuf,
}

pub fn write_paper_tex(
    output_root: &Path,
    hypothesis: &Hypothesis,
    experiment_run: &ExperimentRun,
    metadata_store: &MetadataStore,
) -> Result<SavedPaper> {
    let papers_dir = output_root.join("papers");
    fs::create_dir_all(&papers_dir)?;

    let ts = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let filename = format!("paper-{}-{}.tex", sanitize_filename(&hypothesis.id), ts);
    let tex_path = papers_dir.join(filename);

    let tex = render_tex(hypothesis, experiment_run, metadata_store)?;
    crate::utils::fs::atomic_write(&tex_path, tex.as_bytes())?;

    Ok(SavedPaper { tex_path })
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

fn render_tex(
    hypothesis: &Hypothesis,
    experiment_run: &ExperimentRun,
    metadata_store: &MetadataStore,
) -> Result<String> {
    let mut s = String::new();
    s.push_str("\\documentclass[11pt]{article}\n");
    s.push_str("\\usepackage[margin=1in]{geometry}\n");
    s.push_str("\\usepackage{hyperref}\n");
    s.push_str("\\usepackage{amsmath}\n");
    s.push_str("\\usepackage{booktabs}\n");
    s.push_str("\\title{");
    s.push_str(&escape_tex(&hypothesis.title));
    s.push_str("}\n");
    s.push_str("\\author{ConsensusMind}\n");
    s.push_str("\\date{");
    s.push_str(&escape_tex(&Utc::now().format("%Y-%m-%d").to_string()));
    s.push_str("}\n");
    s.push_str("\\begin{document}\n\\maketitle\n");

    s.push_str("\\begin{abstract}\n");
    s.push_str(&escape_tex(&hypothesis.description));
    s.push_str("\n\\end{abstract}\n\n");

    s.push_str("\\section{Introduction}\n");
    s.push_str("We investigate a new hypothesis for blockchain consensus mechanisms and evaluate it using simulation-based experiments. ");
    s.push_str("This paper is auto-generated from a structured hypothesis description and experiment outputs.\n\n");

    s.push_str("\\section{Hypothesis}\n");
    s.push_str("\\textbf{Mechanism.} ");
    s.push_str(&escape_tex(&hypothesis.mechanism));
    s.push_str("\n\n");

    if !hypothesis.evaluation_plan.is_empty() {
        s.push_str("\\textbf{Evaluation plan.}\n\\begin{enumerate}\n");
        for step in hypothesis.evaluation_plan.iter() {
            s.push_str("\\item ");
            s.push_str(&escape_tex(step));
            s.push('\n');
        }
        s.push_str("\\end{enumerate}\n\n");
    }

    s.push_str("\\section{Experimental Setup}\n");
    s.push_str("We run two families of experiments: (1) a leader-based baseline with an injected leader-failure probability and (2) a simplified Raft-style simulator with message delays, randomized election timeouts, and client request arrivals.\n\n");

    s.push_str("\\section{Results}\n");
    s.push_str("\\subsection{Raft-style simulator}\n");
    if experiment_run.raft.is_empty() {
        s.push_str("No Raft simulation cases were recorded.\n\n");
    } else {
        s.push_str("Table~\\ref{tab:raft} summarizes mean metrics across seeds for each parameter setting.\n\n");
        s.push_str("\\begin{table}[h]\n\\centering\n");
        s.push_str("\\begin{tabular}{rrrrrr}\n\\toprule\n");
        s.push_str(
            "nodes & ticks & $p_{req}$ & elections & leader changes & commit rate\\\\\n\\midrule\n",
        );
        for case in experiment_run.raft.iter() {
            s.push_str(&format!(
                "{} & {} & {:.2} & {:.2} & {:.2} & {:.6}\\\\\n",
                case.params.nodes,
                case.params.ticks,
                case.params.client_request_prob,
                case.aggregate.mean_elections,
                case.aggregate.mean_leader_changes,
                case.aggregate.mean_commit_rate_per_tick
            ));
        }
        s.push_str("\\bottomrule\n\\end{tabular}\n");
        s.push_str(
            "\\caption{Raft-style simulator aggregates (means across seeds).}\\label{tab:raft}\n",
        );
        s.push_str("\\end{table}\n\n");
    }

    s.push_str("\\subsection{Leader-failure baseline}\n");
    if experiment_run.leader.is_empty() {
        s.push_str("No leader-baseline cases were recorded.\n\n");
    } else {
        s.push_str("Table~\\ref{tab:leader} summarizes mean commit rate across seeds.\n\n");
        s.push_str("\\begin{table}[h]\n\\centering\n");
        s.push_str("\\begin{tabular}{rr}\n\\toprule\n");
        s.push_str("$p_{fail}$ & mean commit rate\\\\\n\\midrule\n");
        for case in experiment_run.leader.iter() {
            s.push_str(&format!(
                "{:.2} & {:.4}\\\\\n",
                case.params.leader_failure_prob, case.aggregate.mean_commit_rate
            ));
        }
        s.push_str("\\bottomrule\n\\end{tabular}\n");
        s.push_str("\\caption{Leader-failure baseline aggregates (means across seeds).}\\label{tab:leader}\n");
        s.push_str("\\end{table}\n\n");
    }

    s.push_str("\\section{Related Work}\n");
    if hypothesis.related_paper_ids.is_empty() {
        s.push_str("No related papers were explicitly linked to this hypothesis.\n\n");
    } else {
        s.push_str("We cite the most relevant papers identified during literature review.\n\n");
    }

    s.push_str("\\section{References}\n");
    s.push_str("\\begin{thebibliography}{99}\n");
    for paper_id in hypothesis.related_paper_ids.iter() {
        let label = bib_label(paper_id);
        let title = metadata_store
            .get_paper(paper_id)
            .map(|p| p.title.clone())
            .unwrap_or_else(|| paper_id.to_string());
        s.push_str("\\bibitem{");
        s.push_str(&escape_tex(&label));
        s.push('}');
        s.push_str(&escape_tex(&title));
        s.push_str(". ");
        s.push_str("\\href{https://arxiv.org/abs/");
        s.push_str(&escape_tex(paper_id));
        s.push_str("}{arXiv:");
        s.push_str(&escape_tex(paper_id));
        s.push_str("}.\n");
    }
    s.push_str("\\end{thebibliography}\n");

    s.push_str("\\end{document}\n");
    Ok(s)
}

fn bib_label(paper_id: &str) -> String {
    format!("arxiv:{}", paper_id.replace([':', '/'], "_"))
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

fn escape_tex(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\textbackslash{}"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            '$' => out.push_str("\\$"),
            '&' => out.push_str("\\&"),
            '#' => out.push_str("\\#"),
            '%' => out.push_str("\\%"),
            '_' => out.push_str("\\_"),
            '^' => out.push_str("\\^{}"),
            '~' => out.push_str("\\~{}"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::experiment::{run_experiments, ExperimentOverrides};

    fn temp_path(name: &str) -> PathBuf {
        let pid = std::process::id();
        let ts = Utc::now().timestamp_nanos_opt().unwrap_or(0);
        std::env::temp_dir().join(format!("consensusmind-paper-{}-{}-{}", name, pid, ts))
    }

    #[test]
    fn render_tex_contains_title_and_tables() {
        let root = temp_path("render");
        fs::create_dir_all(&root).unwrap();

        let hypotheses_path = root.join("hypotheses.json");
        let metadata_path = root.join("metadata.json");

        let h = Hypothesis {
            id: "h1".to_string(),
            title: "A New Hypothesis".to_string(),
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

        let (_hyp, run, _path) = run_experiments(
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

        let tex = render_tex(&h, &run, &store).unwrap();
        assert!(tex.contains("\\title{A New Hypothesis}"));
        assert!(tex.contains("\\begin{table}"));
        assert!(tex.contains("\\bibitem{"));
    }
}
