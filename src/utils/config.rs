use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub llm: LlmConfig,
    pub paths: PathsConfig,
    pub agent: AgentConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathsConfig {
    pub papers: PathBuf,
    pub embeddings: PathBuf,
    pub experiments: PathBuf,
    pub output: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub max_iterations: u32,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub file: PathBuf,
}

impl Config {
    pub fn load() -> Result<Self> {
        let config_path =
            std::env::var("CONFIG_PATH").unwrap_or_else(|_| "config.toml".to_string());
        let contents = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config file: {}", config_path))?;

        let mut config: Config =
            toml::from_str(&contents).with_context(|| "Failed to parse config file")?;

        if let Ok(endpoint) = std::env::var("LLM_ENDPOINT") {
            config.llm.endpoint = endpoint;
        }

        if let Ok(api_key) = std::env::var("LLM_API_KEY") {
            config.llm.api_key = api_key;
        }

        if let Ok(model) = std::env::var("LLM_MODEL") {
            config.llm.model = model;
        }

        Ok(config)
    }
}
