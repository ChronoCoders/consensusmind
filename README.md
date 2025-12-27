# ConsensusMind

Autonomous AI researcher for blockchain consensus mechanisms.

## Overview

ConsensusMind is an autonomous research agent that conducts end-to-end research on blockchain consensus protocols. It performs literature review, generates hypotheses, runs simulations, and writes academic papers.

## Status

**Current Version:** 0.1.0 - Milestone 1 Complete

### Completed Milestones

#### Milestone 1: Foundation & Infrastructure
- Project initialization and structure
- Configuration system with TOML and environment variable support
- Logging infrastructure (file + console)
- LLM client with exponential backoff retry logic
- Integration tests
- Production-ready code quality (zero warnings)

## Features

### Current (v0.1.0)
- Configuration management from TOML files
- Environment variable overrides for sensitive data
- Structured logging to file and console
- HTTP client for vLLM/RunPod inference endpoints
- Automatic retry with exponential backoff
- Comprehensive error handling

### Planned
- Automated literature analysis from arXiv and academic databases
- Semantic search over consensus research papers
- Hypothesis generation for novel consensus mechanisms
- Protocol simulation and benchmarking
- Automated LaTeX paper generation

## Architecture

Built in Rust for production reliability and performance.

**Core Components:**
- Agent executor with planning and memory
- Knowledge base with vector search
- Consensus protocol simulator
- LLM client for reasoning tasks
- LaTeX/Markdown output generation

**Tech Stack:**
- Language: Rust 2021 edition
- Async Runtime: Tokio
- HTTP Client: Reqwest with rustls
- Logging: Tracing
- Config: TOML
- LLM: Self-hosted vLLM (DeepSeek/Qwen)

## Requirements

- Rust 1.70+
- GPU inference server (RunPod, self-hosted vLLM, or compatible endpoint)
- Storage for paper corpus

## Installation

```cmd
git clone https://github.com/ChronoCoders/consensusmind.git
cd consensusmind
cargo build --release
```

## Configuration

Create `config.toml` in the project root:

```toml
[llm]
endpoint = "http://localhost:8000/v1"
api_key = ""
model = "deepseek-r1"
max_tokens = 4096
temperature = 0.7

[paths]
papers = "data/papers"
embeddings = "data/embeddings"
experiments = "data/experiments"
output = "output"

[agent]
max_iterations = 10
timeout_seconds = 300

[logging]
level = "info"
file = "consensusmind.log"
```

**Environment Variable Overrides:**
- `LLM_ENDPOINT` - Override LLM endpoint URL
- `LLM_API_KEY` - Override API key
- `LLM_MODEL` - Override model name
- `CONFIG_PATH` - Override config file path

## Usage

```cmd
consensusmind
```

Currently initializes the system and validates configuration.

## Development

**Build:**
```cmd
cargo build
```

**Test:**
```cmd
cargo test
```

**Quality Checks:**
```cmd
cargo fmt
cargo clippy
```

**Release Build:**
```cmd
cargo build --release
```

## Project Structure

```
consensusmind/
ÃÄÄ src/
³   ÃÄÄ main.rs              # Entry point
³   ÃÄÄ lib.rs               # Library root
³   ÃÄÄ agent/               # Agent executor (planned)
³   ÃÄÄ knowledge/           # Knowledge base (planned)
³   ÃÄÄ consensus/           # Simulator (planned)
³   ÃÄÄ llm/                 # LLM client
³   ÃÄÄ output/              # Output generation (planned)
³   ÀÄÄ utils/               # Config, logging
ÃÄÄ tests/
³   ÀÄÄ integration/         # Integration tests
ÃÄÄ data/                    # Paper storage
ÃÄÄ output/                  # Generated papers
ÀÄÄ docs/                    # Documentation
```

## Roadmap

- [x] **Milestone 1:** Foundation & Infrastructure
- [ ] **Milestone 2:** Knowledge Ingestion
- [ ] **Milestone 3:** Knowledge Base
- [ ] **Milestone 4:** Agent Core
- [ ] **Milestone 5:** Consensus Simulator
- [ ] **Milestone 6:** Hypothesis Generation
- [ ] **Milestone 7:** Automated Experimentation
- [ ] **Milestone 8:** Paper Generation
- [ ] **Milestone 9:** Integration & Polish
- [ ] **Milestone 10:** Whitepaper & Research Paper

## License

Apache 2.0 - See LICENSE file

## Contact

**Distributed Systems Labs, LLC**

- GitHub: https://github.com/ChronoCoders/consensusmind
- Website: https://dslabs.network

## Contributing

This project maintains strict code quality standards:
- Zero compiler warnings
- Zero dead code
- Zero unused imports
- Production-ready quality required

Contributions welcome via pull requests.
