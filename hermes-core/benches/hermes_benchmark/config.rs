//! Benchmark configuration

use std::path::PathBuf;

/// Benchmark configuration loaded from environment variables
pub struct BenchmarkConfig {
    pub data_dir: PathBuf,
    pub dense_dim: usize,
    pub num_queries: Option<usize>,
}

impl BenchmarkConfig {
    pub fn from_env() -> Self {
        let data_dir = std::env::var("BENCHMARK_DATA")
            .unwrap_or_else(|_| "benches/benchmark_data".to_string());

        let dense_dim = std::env::var("DENSE_DIM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);

        let num_queries = std::env::var("NUM_QUERIES")
            .ok()
            .and_then(|s| s.parse().ok());

        Self {
            data_dir: PathBuf::from(data_dir),
            dense_dim,
            num_queries,
        }
    }

    // Dense vector paths
    pub fn dense_embeddings_path(&self) -> PathBuf {
        self.data_dir.join("dense_embeddings.bin")
    }

    pub fn dense_queries_path(&self) -> PathBuf {
        self.data_dir.join("dense_queries.bin")
    }

    pub fn ground_truth_dense_full_path(&self) -> PathBuf {
        self.data_dir.join("ground_truth_dense_full.bin")
    }

    // Sparse vector paths
    pub fn sparse_embeddings_path(&self) -> PathBuf {
        self.data_dir.join("sparse_embeddings.bin")
    }

    pub fn sparse_queries_path(&self) -> PathBuf {
        self.data_dir.join("sparse_queries.bin")
    }

    // Text paths for BM25
    pub fn passages_path(&self) -> PathBuf {
        self.data_dir.join("passages.txt")
    }

    pub fn queries_text_path(&self) -> PathBuf {
        self.data_dir.join("queries.txt")
    }

    // Qrels for IR metrics
    pub fn qrels_path(&self) -> PathBuf {
        self.data_dir.join("qrels.bin")
    }

    pub fn has_data(&self) -> bool {
        self.dense_embeddings_path().exists()
    }
}
