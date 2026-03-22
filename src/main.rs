use anyhow::Result;
use consensusmind::agent::Agent;
use consensusmind::consensus::{
    simulate_leader_based, simulate_raft, LeaderSimParams, RaftSimParams,
};
use consensusmind::knowledge::database::MetadataStore;
use consensusmind::knowledge::embedding::{
    build_or_update_index_from_metadata, HashingEmbedder, VectorIndex,
};
use consensusmind::knowledge::paper_parser::PdfParser;
use consensusmind::llm::LlmClient;
use consensusmind::utils::{config::Config, logger};
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let config = Config::load()?;

    logger::init_with_file(&config.logging.file, &config.logging.level)?;

    info!("ConsensusMind starting...");
    info!("Configuration loaded successfully");
    info!("LLM endpoint: {}", config.llm.endpoint);
    info!("LLM model: {}", config.llm.model);

    match args.get(1).map(|s| s.as_str()) {
        Some("index") => {
            let store = MetadataStore::new(config.knowledge.metadata_file.clone())?;
            let parser = PdfParser::new();
            let index_path = config.knowledge.index_file.clone();
            let index = build_or_update_index_from_metadata(
                &index_path,
                config.knowledge.embedding_dims,
                &store,
                &parser,
            )?;
            println!(
                "Indexed {} documents into {}",
                index.docs.len(),
                index_path.display()
            );
            return Ok(());
        }
        Some("semantic-search") => {
            let query = args.get(2).map(|s| s.as_str()).unwrap_or("");
            if query.trim().is_empty() {
                println!("Usage: consensusmind semantic-search <query> [top_k]");
                return Ok(());
            }

            let top_k = args
                .get(3)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(5);

            let index = VectorIndex::load(&config.knowledge.index_file)?;
            let embedder = HashingEmbedder::new(index.dims);
            let q = embedder.embed(query);
            let results = index.search(&q, top_k);

            for (i, r) in results.iter().enumerate() {
                println!(
                    "{}. score={:.4} id={} title={}",
                    i + 1,
                    r.score,
                    r.id,
                    r.title.as_deref().unwrap_or("")
                );
                if let Some(p) = &r.source_path {
                    println!("   {}", p);
                }
            }

            return Ok(());
        }
        Some("run") => {
            let query = args.get(2).map(|s| s.as_str()).unwrap_or("");
            if query.trim().is_empty() {
                println!("Usage: consensusmind run <query>");
                return Ok(());
            }
            let mut agent = Agent::new(config)?;
            let report = agent.run(query).await?;
            println!("Report saved to {}", report.path.display());
            return Ok(());
        }
        Some("simulate") => {
            let rounds = args
                .get(2)
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(1000);
            let leader_failure_prob = args
                .get(3)
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.2);
            let seed = args.get(4).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1);

            let result = simulate_leader_based(LeaderSimParams {
                rounds,
                leader_failure_prob,
                seed,
            });

            let commit_rate = (result.committed as f64) / (result.rounds as f64);
            println!(
                "leader_sim rounds={} committed={} commit_rate={:.4}",
                result.rounds, result.committed, commit_rate
            );
            return Ok(());
        }
        Some("raft-simulate") => {
            let nodes = args
                .get(2)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(5);
            let ticks = args
                .get(3)
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(5000);
            let seed = args.get(4).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1);

            let result = simulate_raft(RaftSimParams {
                nodes,
                ticks,
                seed,
                election_timeout_min: 40,
                election_timeout_max: 80,
                heartbeat_interval: 10,
                network_delay_min: 1,
                network_delay_max: 5,
                client_request_prob: 0.2,
            });

            let commit_rate = (result.committed_entries as f64) / (result.ticks as f64);
            println!(
                "raft_sim nodes={} ticks={} elections={} leader_changes={} committed_entries={} commit_rate={:.6}",
                result.nodes,
                result.ticks,
                result.elections,
                result.leader_changes,
                result.committed_entries,
                commit_rate
            );
            return Ok(());
        }
        _ => {}
    }

    let _client = LlmClient::new(
        config.llm.endpoint.clone(),
        config.llm.api_key.clone(),
        config.llm.model.clone(),
    )?;

    info!("LLM client initialized successfully");
    info!("System ready");

    Ok(())
}
