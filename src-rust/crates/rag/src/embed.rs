// Embedding engine — wraps fastembed for local vector generation.

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use tracing::info;

static EMBEDDER: OnceLock<Mutex<TextEmbedding>> = OnceLock::new();

/// Model cache path. fastembed defaults to a path relative to the working
/// directory, which downloads the model again in every new folder.
fn model_cache_dir() -> PathBuf {
    std::env::var_os("FASTEMBED_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| crate::store::rag_dir().join("models"))
}

pub struct Embedder;

impl Embedder {
    /// Load the embedding model (downloads on first use, ~97MB). On failure the
    /// embedder stays not ready and the next call tries again.
    pub fn init() -> Result<(), String> {
        if Self::is_ready() {
            return Ok(());
        }
        info!("Loading embedding model (all-MiniLM-L6-v2)...");
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                .with_show_download_progress(true)
                .with_cache_dir(model_cache_dir()),
        )
        .map_err(|e| format!("{e:#}"))?;
        let _ = EMBEDDER.set(Mutex::new(model));
        info!("Embedding model ready");
        Ok(())
    }

    /// Embed a single text string. Returns a 384-dim vector.
    pub fn embed_one(text: &str) -> Vec<f32> {
        let lock = EMBEDDER.get().expect("Embedder not initialized");
        let mut model = lock.lock().unwrap();
        let results = model
            .embed(vec![text.to_string()], None)
            .expect("Embedding failed");
        results.into_iter().next().unwrap_or_default()
    }

    /// Embed multiple texts in batch.
    pub fn embed_batch(texts: &[String]) -> Vec<Vec<f32>> {
        let lock = EMBEDDER.get().expect("Embedder not initialized");
        let mut model = lock.lock().unwrap();
        model.embed(texts, None).expect("Batch embedding failed")
    }

    /// Check if the embedder is initialized.
    pub fn is_ready() -> bool {
        EMBEDDER.get().is_some()
    }
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_cache_does_not_depend_on_working_directory() {
        assert!(model_cache_dir().is_absolute());
    }

    #[test]
    fn init_reports_an_unreachable_model_instead_of_panicking() {
        let cache = std::env::temp_dir().join(format!("cc-rag-no-model-{}", std::process::id()));
        let orig_cache = std::env::var_os("FASTEMBED_CACHE_DIR");
        let orig_endpoint = std::env::var_os("HF_ENDPOINT");
        std::env::set_var("FASTEMBED_CACHE_DIR", &cache);
        std::env::set_var("HF_ENDPOINT", "http://127.0.0.1:9");

        let result = Embedder::init();
        let ready = Embedder::is_ready();

        // Restore env vars
        for (key, orig) in [
            ("FASTEMBED_CACHE_DIR", orig_cache),
            ("HF_ENDPOINT", orig_endpoint),
        ] {
            match orig {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
        let _ = std::fs::remove_dir_all(&cache);

        assert!(result.is_err());
        assert!(!ready);
    }

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }
}
