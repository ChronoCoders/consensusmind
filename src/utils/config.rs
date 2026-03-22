use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub llm: LlmConfig,
    pub paths: PathsConfig,
    pub agent: AgentConfig,
    pub logging: LoggingConfig,
    #[serde(default)]
    pub knowledge: KnowledgeConfig,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub endpoint: String,
    pub api_key: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl std::fmt::Debug for LlmConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmConfig")
            .field("endpoint", &self.endpoint)
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .finish()
    }
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeConfig {
    pub max_pdf_bytes: u64,
}

impl Default for KnowledgeConfig {
    fn default() -> Self {
        Self {
            max_pdf_bytes: 100 * 1024 * 1024,
        }
    }
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

        if !config.llm.api_key.is_empty()
            && config.llm.endpoint.starts_with("http://")
            && !is_local_http_endpoint(&config.llm.endpoint)
        {
            bail!(
                "Refusing to send LLM API key to non-local http endpoint: {}",
                config.llm.endpoint
            );
        }

        Ok(config)
    }
}

fn is_local_http_endpoint(endpoint: &str) -> bool {
    let endpoint = endpoint.trim();
    endpoint.starts_with("http://localhost")
        || endpoint.starts_with("http://127.0.0.1")
        || endpoint.starts_with("http://[::1]")
}
