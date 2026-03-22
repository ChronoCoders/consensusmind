# ConsensusMind

Autonomous AI researcher for blockchain consensus mechanisms.

## Overview

ConsensusMind is an autonomous research agent that conducts end-to-end research on blockchain consensus protocols. It performs literature review, generates hypotheses, runs simulations, and writes academic papers.

## Status

**Current Version:** 0.9.0 - Milestone 9 Complete

### Completed Milestones

#### Milestone 1: Foundation & Infrastructure
- Project initialization and structure
- Configuration system with TOML and environment variable support
- Logging infrastructure (file + console)
- LLM client with exponential backoff retry logic
- Integration tests
- Production-ready code quality (zero warnings)

#### Milestone 2: Knowledge Ingestion
- arXiv API integration with HTTPS
- PDF download and local storage
- JSON-based metadata tracking
- PDF text extraction and analysis
- Rate limiting and error handling
- Search and store workflow

#### Milestone 3: Knowledge Base
- Local embedding index build/update
- Semantic search over indexed papers

#### Milestone 4: Agent Core
- Agent runner with step tracking and persisted run memory
- Report generation

#### Milestone 5: Consensus Simulator
- Deterministic simulators with CLI entrypoints

#### Milestone 6: Hypothesis Generation
- LLM-driven hypothesis generation with novelty scoring
- Hypotheses persistence and reporting

#### Milestone 7: Automated Experimentation
- Experiment runner tied to hypothesis IDs
- Results saved under data/experiments and summarized into output reports

#### Milestone 8: Paper Generation
- LaTeX paper generation from hypotheses and experiment results

#### Milestone 9: Integration & Polish
- Unified CLI entrypoints (help/version and command usage)
- Repo structure improvements for outputs and data directories

## Features

### Current (v0.9.0)
- Configuration management from TOML files
- Environment variable overrides for sensitive data
- Structured logging to file and console
- HTTP client for vLLM/RunPod inference endpoints
- Automatic retry with exponential backoff
- Comprehensive error handling
- arXiv paper search and retrieval
- PDF download with duplicate detection
- Metadata persistence in JSON
- PDF text extraction for analysis
- Local embedding index + semantic search
- Agent run pipeline (search, download, index, retrieve, summarize, report)
- Hypothesis generation and persistence
- Consensus simulation and experimentation
- LaTeX paper generation from experiment outputs
- CLI help/version and stable command interface

### Planned
- Automated LaTeX paper generation

## Architecture

Built in Rust for production reliability and performance.

**Core Components:**
- Agent executor with planning and memory
- Knowledge base with vector search
- Consensus protocol simulator
- LLM client for reasoning tasks
- LaTeX/Markdown output generation
- arXiv integration for paper retrieval
- PDF parsing and text extraction

**Tech Stack:**
- Language: Rust 2021 edition
- Async Runtime: Tokio
- HTTP Client: Reqwest with rustls
- Logging: Tracing
- Config: TOML
- LLM: Self-hosted vLLM (DeepSeek/Qwen)
- PDF Processing: pdf-extract
- Data Storage: JSON metadata + local files

## Requirements

- Rust 1.70+
- GPU inference server (RunPod, self-hosted vLLM, or compatible endpoint)
- Storage for paper corpus

## Installation
```bash
git clone https://github.com/ChronoCoders/consensusmind.git
cd consensusmind
cargo build --release
```

## Configuration

Create `config.toml` in the project root with LLM endpoint, model settings, paths, agent parameters, and logging configuration.

For secrets (like API keys), use either:
- Environment variables (recommended), or
- `config.local.toml` (kept out of git via `.gitignore`)

Environment variable overrides available:
- LLM_ENDPOINT
- LLM_API_KEY
- LLM_MODEL
- CONFIG_PATH

Additional settings:
- `knowledge.max_pdf_bytes` controls the maximum allowed PDF download size.

## Usage
```bash
consensusmind run "<query>"
consensusmind hypothesize "<query>"
consensusmind experiment <hypothesis-id> [--seeds N] [--ticks T] [--nodes N]
consensusmind paper <hypothesis-id>
consensusmind index
consensusmind semantic-search "<query>" [top_k]
consensusmind simulate [rounds] [leader_failure_prob] [seed]
consensusmind raft-simulate [nodes] [ticks] [seed]
```

Currently initializes the system, validates configuration, and provides arXiv search and PDF download capabilities.

## Development
```bash
cargo build          # Build debug
cargo test           # Run tests
cargo fmt            # Format code
cargo clippy         # Lint
cargo build --release # Build optimized
```

CI runs `cargo fmt --check`, `cargo clippy -- -D warnings`, and `cargo test` on push/PR, plus scheduled dependency/security checks.

## Roadmap

- [x] Milestone 1: Foundation & Infrastructure
- [x] Milestone 2: Knowledge Ingestion
- [x] Milestone 3: Knowledge Base
- [x] Milestone 4: Agent Core
- [x] Milestone 5: Consensus Simulator
- [x] Milestone 6: Hypothesis Generation
- [x] Milestone 7: Automated Experimentation
- [x] Milestone 8: Paper Generation
- [x] Milestone 9: Integration & Polish
- [ ] Milestone 10: Whitepaper & Research Paper

## License

Apache 2.0 - See LICENSE file

## Contact

Distributed Systems Labs, LLC
- GitHub: https://github.com/ChronoCoders/consensusmind
- Website: https://dslabs.network

## Contributing

This project maintains strict code quality standards:
- Zero compiler warnings
- Zero dead code
- Zero unused imports
- Production-ready quality required

Contributions welcome via pull requests.
