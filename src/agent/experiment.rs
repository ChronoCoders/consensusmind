use crate::agent::hypothesis::Hypothesis;
use crate::consensus::{simulate_leader_based, simulate_raft, LeaderSimParams, RaftSimParams};
use anyhow::{bail, Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
pub struct ExperimentOverrides {
    pub seeds: Option<usize>,
    pub ticks: Option<u64>,
    pub nodes: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentRun {
    pub hypothesis_id: String,
    pub created_at: String,
    pub raft: Vec<RaftExperimentCase>,
    pub leader: Vec<LeaderExperimentCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftExperimentCase {
    pub params: RaftSimParams,
    pub seeds: Vec<u64>,
    pub per_seed: Vec<RaftSimResultRow>,
    pub aggregate: RaftAggregate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftSimResultRow {
    pub seed: u64,
    pub elections: u64,
    pub leader_changes: u64,
    pub committed_entries: u64,
    pub commit_rate_per_tick: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftAggregate {
    pub seeds: usize,
    pub mean_elections: f64,
    pub mean_leader_changes: f64,
    pub mean_committed_entries: f64,
    pub mean_commit_rate_per_tick: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderExperimentCase {
    pub params: LeaderSimParams,
    pub per_seed: Vec<LeaderSimResultRow>,
    pub aggregate: LeaderAggregate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderSimResultRow {
    pub seed: u64,
    pub committed: u32,
    pub commit_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderAggregate {
    pub seeds: usize,
    pub mean_committed: f64,
    pub mean_commit_rate: f64,
}

pub fn load_hypotheses(path: &Path) -> Result<Vec<Hypothesis>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let contents = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&contents).unwrap_or_default())
}

pub fn find_hypothesis(path: &Path, id: &str) -> Result<Hypothesis> {
    let hypotheses = load_hypotheses(path)?;
    hypotheses
        .into_iter()
        .find(|h| h.id == id)
        .with_context(|| format!("Hypothesis not found: {}", id))
}

pub fn run_experiments(
    experiments_root: &Path,
    hypotheses_file: &Path,
    hypothesis_id: &str,
    overrides: ExperimentOverrides,
) -> Result<(Hypothesis, ExperimentRun, PathBuf)> {
    let hypothesis = find_hypothesis(hypotheses_file, hypothesis_id)?;

    let seeds = generate_seeds(overrides.seeds.unwrap_or(5));
    if seeds.is_empty() {
        bail!("No seeds configured");
    }

    let nodes_grid = overrides
        .nodes
        .map(|n| vec![n])
        .unwrap_or_else(|| vec![3, 5, 7]);
    let ticks_grid = overrides
        .ticks
        .map(|t| vec![t])
        .unwrap_or_else(|| vec![2000, 5000]);
    let client_probs = [0.1, 0.2, 0.3];

    let mut raft_cases = Vec::new();
    for nodes in nodes_grid {
        for ticks in ticks_grid.iter().copied() {
            for client_request_prob in client_probs.iter().copied() {
                let params = RaftSimParams {
                    nodes,
                    ticks,
                    seed: 0,
                    election_timeout_min: 40,
                    election_timeout_max: 80,
                    heartbeat_interval: 10,
                    network_delay_min: 1,
                    network_delay_max: 5,
                    client_request_prob,
                };
                raft_cases.push(run_raft_case(params, &seeds));
            }
        }
    }

    let leader_failure_probs = [0.0, 0.1, 0.2, 0.3];
    let mut leader_cases = Vec::new();
    for leader_failure_prob in leader_failure_probs {
        let params = LeaderSimParams {
            rounds: 10_000,
            leader_failure_prob,
            seed: 0,
        };
        leader_cases.push(run_leader_case(params, &seeds));
    }

    let created_at = Utc::now().to_rfc3339();
    let run = ExperimentRun {
        hypothesis_id: hypothesis.id.clone(),
        created_at,
        raft: raft_cases,
        leader: leader_cases,
    };

    let experiment_dir = experiments_root.join(&hypothesis.id);
    fs::create_dir_all(&experiment_dir)?;
    let results_path = experiment_dir.join("results.json");
    let contents = serde_json::to_string_pretty(&run)?;
    crate::utils::fs::atomic_write(&results_path, contents.as_bytes())?;

    Ok((hypothesis, run, results_path))
}

fn run_raft_case(mut base: RaftSimParams, seeds: &[u64]) -> RaftExperimentCase {
    let mut per_seed = Vec::new();
    for &seed in seeds {
        base.seed = seed;
        let r = simulate_raft(base);
        let commit_rate_per_tick = (r.committed_entries as f64) / (r.ticks as f64);
        per_seed.push(RaftSimResultRow {
            seed,
            elections: r.elections,
            leader_changes: r.leader_changes,
            committed_entries: r.committed_entries,
            commit_rate_per_tick,
        });
    }

    let aggregate = RaftAggregate {
        seeds: per_seed.len(),
        mean_elections: mean_u64(per_seed.iter().map(|r| r.elections)),
        mean_leader_changes: mean_u64(per_seed.iter().map(|r| r.leader_changes)),
        mean_committed_entries: mean_u64(per_seed.iter().map(|r| r.committed_entries)),
        mean_commit_rate_per_tick: mean_f64(per_seed.iter().map(|r| r.commit_rate_per_tick)),
    };

    RaftExperimentCase {
        params: RaftSimParams { seed: 0, ..base },
        seeds: seeds.to_vec(),
        per_seed,
        aggregate,
    }
}

fn run_leader_case(mut base: LeaderSimParams, seeds: &[u64]) -> LeaderExperimentCase {
    let mut per_seed = Vec::new();
    for &seed in seeds {
        base.seed = seed;
        let r = simulate_leader_based(base);
        let commit_rate = (r.committed as f64) / (r.rounds as f64);
        per_seed.push(LeaderSimResultRow {
            seed,
            committed: r.committed,
            commit_rate,
        });
    }

    let aggregate = LeaderAggregate {
        seeds: per_seed.len(),
        mean_committed: mean_u32(per_seed.iter().map(|r| r.committed)),
        mean_commit_rate: mean_f64(per_seed.iter().map(|r| r.commit_rate)),
    };

    LeaderExperimentCase {
        params: LeaderSimParams { seed: 0, ..base },
        per_seed,
        aggregate,
    }
}

fn mean_u64<I: Iterator<Item = u64>>(iter: I) -> f64 {
    let mut sum = 0f64;
    let mut n = 0f64;
    for x in iter {
        sum += x as f64;
        n += 1.0;
    }
    if n == 0.0 {
        0.0
    } else {
        sum / n
    }
}

fn mean_u32<I: Iterator<Item = u32>>(iter: I) -> f64 {
    let mut sum = 0f64;
    let mut n = 0f64;
    for x in iter {
        sum += x as f64;
        n += 1.0;
    }
    if n == 0.0 {
        0.0
    } else {
        sum / n
    }
}

fn mean_f64<I: Iterator<Item = f64>>(iter: I) -> f64 {
    let mut sum = 0f64;
    let mut n = 0f64;
    for x in iter {
        sum += x;
        n += 1.0;
    }
    if n == 0.0 {
        0.0
    } else {
        sum / n
    }
}

fn generate_seeds(n: usize) -> Vec<u64> {
    let mut out = Vec::new();
    for i in 0..n {
        out.push((i as u64) + 1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        let pid = std::process::id();
        let ts = Utc::now().timestamp_nanos_opt().unwrap_or(0);
        std::env::temp_dir().join(format!("consensusmind-{}-{}-{}", name, pid, ts))
    }

    #[test]
    fn run_experiments_writes_results_file() {
        let root = temp_path("experiments");
        let hypotheses = root.join("hypotheses.json");
        fs::create_dir_all(&root).unwrap();

        let h = Hypothesis {
            id: "h1".to_string(),
            title: "t".to_string(),
            description: "d".to_string(),
            mechanism: "m".to_string(),
            evaluation_plan: vec!["p".to_string()],
            related_paper_ids: vec![],
            novelty_score: 0.1,
            feasibility_score: 0.9,
            created_at: Utc::now().to_rfc3339(),
        };
        let contents = serde_json::to_string_pretty(&vec![h]).unwrap();
        crate::utils::fs::atomic_write(&hypotheses, contents.as_bytes()).unwrap();

        let (hyp, run, results_path) = run_experiments(
            &root,
            &hypotheses,
            "h1",
            ExperimentOverrides {
                seeds: Some(2),
                ticks: Some(200),
                nodes: Some(3),
            },
        )
        .unwrap();

        assert_eq!(hyp.id, "h1");
        assert_eq!(run.hypothesis_id, "h1");
        assert!(results_path.exists());
    }
}
