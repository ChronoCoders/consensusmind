# ConsensusMind

Autonomous AI researcher for blockchain consensus mechanisms.

## Overview

ConsensusMind is an autonomous research agent that conducts end-to-end research on blockchain consensus protocols. It performs literature review, generates hypotheses, runs simulations, and writes academic papers.

## Features

- Automated literature analysis from arXiv and academic databases
- Semantic search over consensus research papers
- Hypothesis generation for novel consensus mechanisms
- Protocol simulation and benchmarking
- Automated LaTeX paper generation
- Self-hosted LLM inference (no API dependencies)

## Architecture

Built in Rust for production reliability and performance. Core components:

- Agent executor with planning and memory
- Knowledge base with vector search
- Consensus protocol simulator
- LLM client for reasoning tasks
- LaTeX/Markdown output generation

## Requirements

- Rust 1.70+
- GPU inference server (RunPod, self-hosted vLLM)
- Storage for paper corpus

## Installation
```cmd
git clone https://github.com/yourusername/consensusmind.git
cd consensusmind
cargo build --release
```

## Configuration

Create `config.toml`:
```toml
[llm]
endpoint = "https://your-runpod-endpoint.com"
api_key = "your-api-key"
model = "deepseek-r1"

[paths]
papers = "data/papers"
embeddings = "data/embeddings"
output = "output"
```

## Usage
```cmd
consensusmind --query "Analyze trade-offs in DAG-based consensus"
```

## Development
```cmd
cargo test
cargo clippy
cargo fmt
```

## License

Apache 2.0

## Contact

Distributed Systems Labs, LLC