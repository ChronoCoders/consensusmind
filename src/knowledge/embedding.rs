use crate::knowledge::database::MetadataStore;
use crate::knowledge::paper_parser::PdfParser;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndex {
    pub dims: usize,
    pub docs: Vec<IndexedDoc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedDoc {
    pub id: String,
    pub title: Option<String>,
    pub source_path: Option<String>,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub title: Option<String>,
    pub source_path: Option<String>,
    pub score: f32,
}

impl VectorIndex {
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            docs: Vec::new(),
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&contents)?)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let contents = serde_json::to_string_pretty(self)?;
        crate::utils::fs::atomic_write(path, contents.as_bytes())?;
        Ok(())
    }

    pub fn upsert(&mut self, doc: IndexedDoc) {
        if let Some(existing) = self.docs.iter_mut().find(|d| d.id == doc.id) {
            *existing = doc;
            return;
        }
        self.docs.push(doc);
    }

    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = self
            .docs
            .iter()
            .map(|d| SearchResult {
                id: d.id.clone(),
                title: d.title.clone(),
                source_path: d.source_path.clone(),
                score: dot(query_embedding, &d.embedding),
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        results.truncate(top_k);
        results
    }
}

#[derive(Debug, Clone)]
pub struct HashingEmbedder {
    dims: usize,
}

impl HashingEmbedder {
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }

    pub fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dims];

        for token in tokenize(text) {
            let h = fnv1a_64(token.as_bytes());
            let idx = (h as usize) % self.dims;
            let sign = if (h & 1) == 0 { 1.0 } else { -1.0 };
            v[idx] += sign;
        }

        l2_normalize(&mut v);
        v
    }
}

pub fn build_or_update_index_from_metadata(
    index_path: &Path,
    dims: usize,
    metadata_store: &MetadataStore,
    pdf_parser: &PdfParser,
) -> Result<VectorIndex> {
    let embedder = HashingEmbedder::new(dims);

    let mut index = if index_path.exists() {
        VectorIndex::load(index_path)?
    } else {
        VectorIndex::new(dims)
    };

    if index.dims != dims {
        index = VectorIndex::new(dims);
    }

    for paper in metadata_store.list_papers() {
        let id = paper.arxiv_id.clone();

        let (text, source_path) = match paper.pdf_path.as_ref() {
            Some(p) if Path::new(p).exists() => {
                let extracted = pdf_parser.extract_text(Path::new(p)).unwrap_or_default();
                (extracted, Some(p.clone()))
            }
            _ => {
                let mut fallback = String::new();
                if !paper.title.is_empty() {
                    fallback.push_str(&paper.title);
                    fallback.push('\n');
                }
                if !paper.abstract_text.is_empty() {
                    fallback.push_str(&paper.abstract_text);
                }
                (fallback, paper.pdf_path.clone())
            }
        };

        if text.trim().is_empty() {
            continue;
        }

        let embedding = embedder.embed(&text);

        index.upsert(IndexedDoc {
            id,
            title: Some(paper.title.clone()).filter(|t| !t.trim().is_empty()),
            source_path,
            embedding,
        });
    }

    index.save(index_path)?;
    Ok(index)
}

pub fn default_index_path(embeddings_dir: &Path) -> PathBuf {
    embeddings_dir.join("index.json")
}

fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split_whitespace().filter_map(|raw| {
        let token: String = raw
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect();
        if token.is_empty() {
            None
        } else {
            Some(token)
        }
    })
}

fn l2_normalize(v: &mut [f32]) {
    let mut sum_sq = 0.0f32;
    for x in v.iter() {
        sum_sq += x * x;
    }
    let norm = sum_sq.sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashing_embedder_is_deterministic() {
        let embedder = HashingEmbedder::new(128);
        let a = embedder.embed("hello world");
        let b = embedder.embed("hello world");
        assert_eq!(a, b);
        assert!(dot(&a, &b) > 0.99);
    }

    #[test]
    fn vector_index_search_ranks_similar_text_higher() {
        let embedder = HashingEmbedder::new(128);
        let mut index = VectorIndex::new(128);

        index.upsert(IndexedDoc {
            id: "a".to_string(),
            title: None,
            source_path: None,
            embedding: embedder.embed("consensus protocol leader election safety"),
        });

        index.upsert(IndexedDoc {
            id: "b".to_string(),
            title: None,
            source_path: None,
            embedding: embedder.embed("neural networks image classification"),
        });

        let q = embedder.embed("leader election in consensus");
        let results = index.search(&q, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
        assert!(results[0].score >= results[1].score);
    }
}
