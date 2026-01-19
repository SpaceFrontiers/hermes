//! Unified Hermes Benchmark Suite
//!
//! Consolidates all benchmarks into a single coherent output:
//! - Dense vector search (RaBitQ, IVF-RaBitQ, ScaNN)
//! - Matryoshka/MRL dimension comparison
//! - Sparse vector search (SPLADE)
//! - Text search (BM25)
//! - Hybrid search combinations
//!
//! Usage:
//!   # Generate benchmark data first
//!   TRITON_URL=... TRITON_API_KEY=... python benches/generate_benchmark_data.py --use-beir
//!
//!   # Run benchmarks
//!   BENCHMARK_DATA=benches/benchmark_data cargo bench --bench hermes_benchmark

pub mod config;
pub mod data;
pub mod dense;
pub mod metrics;
pub mod output;
pub mod sparse;
