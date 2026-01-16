//! Large-scale benchmarks for dense and sparse vector indexes
//!
//! Measures indexing, merging, querying latencies and recall using real embeddings.
//!
//! To generate benchmark data:
//!   python benches/generate_benchmark_data.py --num-docs 100000 --num-queries 1000
//!
//! Then run:
//!   BENCHMARK_DATA=benches/benchmark_data cargo bench --bench large_benchmark
//!
//! Or with custom path:
//!   BENCHMARK_DATA=/path/to/data cargo bench --bench large_benchmark

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::{Duration, Instant};

use comfy_table::{Cell, Color, Table};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

use hermes_core::structures::{
    BlockSparsePostingList, CoarseCentroids, IVFConfig, IVFRaBitQIndex, RaBitQConfig, RaBitQIndex,
    WeightQuantization,
};

// ============================================================================
// Data Loading
// ============================================================================

/// Dense embeddings: (vectors, dim)
fn load_dense_embeddings(path: &Path) -> Option<(Vec<Vec<f32>>, usize)> {
    let file = File::open(path).ok()?;
    let mut reader = BufReader::new(file);

    let mut header = [0u8; 8];
    reader.read_exact(&mut header).ok()?;

    let num_vectors = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let dim = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

    let mut data = vec![0u8; num_vectors * dim * 4];
    reader.read_exact(&mut data).ok()?;

    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| {
            let offset = i * dim * 4;
            (0..dim)
                .map(|j| {
                    let idx = offset + j * 4;
                    f32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]])
                })
                .collect()
        })
        .collect();

    println!(
        "Loaded {} dense vectors (dim={}) from {:?}",
        num_vectors, dim, path
    );
    Some((vectors, dim))
}

/// Sparse embeddings: Vec<(indices, values)>
fn load_sparse_embeddings(path: &Path) -> Option<Vec<(Vec<u32>, Vec<f32>)>> {
    let file = File::open(path).ok()?;
    let mut reader = BufReader::new(file);

    let mut header = [0u8; 4];
    reader.read_exact(&mut header).ok()?;
    let num_vectors = u32::from_le_bytes(header) as usize;

    let mut vectors = Vec::with_capacity(num_vectors);
    let mut total_nnz = 0usize;

    for _ in 0..num_vectors {
        let mut nnz_buf = [0u8; 4];
        reader.read_exact(&mut nnz_buf).ok()?;
        let nnz = u32::from_le_bytes(nnz_buf) as usize;
        total_nnz += nnz;

        let mut indices_buf = vec![0u8; nnz * 4];
        reader.read_exact(&mut indices_buf).ok()?;
        let indices: Vec<u32> = (0..nnz)
            .map(|i| {
                let idx = i * 4;
                u32::from_le_bytes([
                    indices_buf[idx],
                    indices_buf[idx + 1],
                    indices_buf[idx + 2],
                    indices_buf[idx + 3],
                ])
            })
            .collect();

        let mut values_buf = vec![0u8; nnz * 4];
        reader.read_exact(&mut values_buf).ok()?;
        let values: Vec<f32> = (0..nnz)
            .map(|i| {
                let idx = i * 4;
                f32::from_le_bytes([
                    values_buf[idx],
                    values_buf[idx + 1],
                    values_buf[idx + 2],
                    values_buf[idx + 3],
                ])
            })
            .collect();

        vectors.push((indices, values));
    }

    let avg_nnz = total_nnz as f64 / num_vectors as f64;
    println!(
        "Loaded {} sparse vectors (avg nnz={:.1}) from {:?}",
        num_vectors, avg_nnz, path
    );
    Some(vectors)
}

/// Ground truth: (num_queries, k, indices)
fn load_ground_truth(path: &Path) -> Option<Vec<Vec<u32>>> {
    let file = File::open(path).ok()?;
    let mut reader = BufReader::new(file);

    let mut header = [0u8; 8];
    reader.read_exact(&mut header).ok()?;

    let num_queries = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let k = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

    let mut data = vec![0u8; num_queries * k * 4];
    reader.read_exact(&mut data).ok()?;

    let ground_truth: Vec<Vec<u32>> = (0..num_queries)
        .map(|i| {
            let offset = i * k * 4;
            (0..k)
                .map(|j| {
                    let idx = offset + j * 4;
                    u32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]])
                })
                .collect()
        })
        .collect();

    println!("Loaded ground truth: {} queries, k={}", num_queries, k);
    Some(ground_truth)
}

// ============================================================================
// Metrics
// ============================================================================

fn compute_recall(predicted: &[usize], ground_truth: &[u32], k: usize) -> f32 {
    let gt_set: HashSet<u32> = ground_truth.iter().take(k).copied().collect();
    let correct = predicted
        .iter()
        .take(k)
        .filter(|&idx| gt_set.contains(&(*idx as u32)))
        .count();
    correct as f32 / k as f32
}

struct BenchmarkResult {
    name: String,
    num_vectors: usize,
    build_time: Duration,
    merge_time: Option<Duration>,
    avg_query_latency_us: f64,
    p99_query_latency_us: f64,
    recall_at_10: f32,
    recall_at_100: f32,
    index_size_bytes: usize,
}

fn print_results(results: &[BenchmarkResult]) {
    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("Index").fg(Color::Cyan),
        Cell::new("Vectors").fg(Color::Cyan),
        Cell::new("Build").fg(Color::Cyan),
        Cell::new("Merge").fg(Color::Cyan),
        Cell::new("Avg Latency").fg(Color::Cyan),
        Cell::new("P99 Latency").fg(Color::Cyan),
        Cell::new("R@10").fg(Color::Cyan),
        Cell::new("R@100").fg(Color::Cyan),
        Cell::new("Size").fg(Color::Cyan),
    ]);

    for r in results {
        table.add_row(vec![
            Cell::new(&r.name),
            Cell::new(format!("{}", r.num_vectors)),
            Cell::new(format!("{:.2?}", r.build_time)),
            Cell::new(
                r.merge_time
                    .map_or("-".to_string(), |d| format!("{:.2?}", d)),
            ),
            Cell::new(format!("{:.1} µs", r.avg_query_latency_us)),
            Cell::new(format!("{:.1} µs", r.p99_query_latency_us)),
            Cell::new(format!("{:.1}%", r.recall_at_10 * 100.0)).fg(if r.recall_at_10 > 0.9 {
                Color::Green
            } else {
                Color::Yellow
            }),
            Cell::new(format!("{:.1}%", r.recall_at_100 * 100.0)).fg(if r.recall_at_100 > 0.9 {
                Color::Green
            } else {
                Color::Yellow
            }),
            Cell::new(format!(
                "{:.1} MB",
                r.index_size_bytes as f64 / 1024.0 / 1024.0
            )),
        ]);
    }

    println!("\n{table}");
}

// ============================================================================
// Dense Vector Benchmarks
// ============================================================================

fn benchmark_dense_rabitq(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth: &[Vec<u32>],
    dim: usize,
) -> BenchmarkResult {
    let n = vectors.len();
    let k = 100;

    // Build index
    let build_start = Instant::now();
    let config = RaBitQConfig::new(dim);
    let index = RaBitQIndex::build(config, vectors, true);
    let build_time = build_start.elapsed();

    // Query and measure latency
    let mut latencies = Vec::with_capacity(queries.len());
    let mut total_recall_10 = 0.0f32;
    let mut total_recall_100 = 0.0f32;

    for (i, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = index.search(query, k, 3);
        let latency = start.elapsed();
        latencies.push(latency.as_micros() as f64);

        let predicted: Vec<usize> = results.iter().map(|(idx, _)| *idx).collect();
        total_recall_10 += compute_recall(&predicted, &ground_truth[i], 10);
        total_recall_100 += compute_recall(&predicted, &ground_truth[i], 100);
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p99_idx = (latencies.len() as f64 * 0.99) as usize;
    let p99_latency = latencies.get(p99_idx).copied().unwrap_or(avg_latency);

    BenchmarkResult {
        name: "RaBitQ".to_string(),
        num_vectors: n,
        build_time,
        merge_time: None,
        avg_query_latency_us: avg_latency,
        p99_query_latency_us: p99_latency,
        recall_at_10: total_recall_10 / queries.len() as f32,
        recall_at_100: total_recall_100 / queries.len() as f32,
        index_size_bytes: index.size_bytes(),
    }
}

fn benchmark_dense_ivf_rabitq(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth: &[Vec<u32>],
    dim: usize,
    nprobe: usize,
) -> BenchmarkResult {
    let n = vectors.len();
    let k = 100;
    let num_clusters = ((n as f64).sqrt() as usize).clamp(16, 1024);

    // Train centroids
    let train_start = Instant::now();
    let centroids = CoarseCentroids::train(vectors, num_clusters, 10, 42);
    let train_time = train_start.elapsed();

    // Build index
    let build_start = Instant::now();
    let config = IVFConfig::new(dim);
    let index = IVFRaBitQIndex::build(config, &centroids, vectors, None);
    let build_time = build_start.elapsed() + train_time;

    // Query and measure latency
    let mut latencies = Vec::with_capacity(queries.len());
    let mut total_recall_10 = 0.0f32;
    let mut total_recall_100 = 0.0f32;

    for (i, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = index.search(&centroids, query, k, nprobe);
        let latency = start.elapsed();
        latencies.push(latency.as_micros() as f64);

        let predicted: Vec<usize> = results.iter().map(|(idx, _)| *idx as usize).collect();
        total_recall_10 += compute_recall(&predicted, &ground_truth[i], 10);
        total_recall_100 += compute_recall(&predicted, &ground_truth[i], 100);
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p99_idx = (latencies.len() as f64 * 0.99) as usize;
    let p99_latency = latencies.get(p99_idx).copied().unwrap_or(avg_latency);

    BenchmarkResult {
        name: format!("IVF-RaBitQ (nprobe={})", nprobe),
        num_vectors: n,
        build_time,
        merge_time: None,
        avg_query_latency_us: avg_latency,
        p99_query_latency_us: p99_latency,
        recall_at_10: total_recall_10 / queries.len() as f32,
        recall_at_100: total_recall_100 / queries.len() as f32,
        index_size_bytes: index.size_bytes(),
    }
}

fn benchmark_dense_ivf_merge(vectors: &[Vec<f32>], dim: usize, num_segments: usize) -> Duration {
    let n = vectors.len();
    let vectors_per_segment = n / num_segments;
    let num_clusters = ((vectors_per_segment as f64).sqrt() as usize).clamp(16, 256);

    // Train shared centroids
    let centroids = CoarseCentroids::train(
        &vectors[..vectors_per_segment.min(10000)],
        num_clusters,
        10,
        42,
    );

    // Build segments
    let mut segments = Vec::new();
    for i in 0..num_segments {
        let start = i * vectors_per_segment;
        let end = ((i + 1) * vectors_per_segment).min(n);
        let segment_vectors: Vec<Vec<f32>> = vectors[start..end].to_vec();

        let config = IVFConfig::new(dim);
        let index = IVFRaBitQIndex::build(config, &centroids, &segment_vectors, None);
        segments.push(index);
    }

    // Merge
    let refs: Vec<&IVFRaBitQIndex> = segments.iter().collect();
    let offsets: Vec<u32> = (0..num_segments)
        .map(|i| (i * vectors_per_segment) as u32)
        .collect();

    let merge_start = Instant::now();
    let _merged = IVFRaBitQIndex::merge(&refs, &offsets).unwrap();
    merge_start.elapsed()
}

// ============================================================================
// Sparse Vector Benchmarks
// ============================================================================

/// Simple in-memory sparse index for benchmarking
struct SimpleSparseIndex {
    /// Inverted index: dimension -> Vec<(doc_id, weight)>
    postings: rustc_hash::FxHashMap<u32, Vec<(u32, f32)>>,
}

impl SimpleSparseIndex {
    fn build(vectors: &[(Vec<u32>, Vec<f32>)]) -> Self {
        let mut postings: rustc_hash::FxHashMap<u32, Vec<(u32, f32)>> =
            rustc_hash::FxHashMap::default();

        for (doc_id, (indices, values)) in vectors.iter().enumerate() {
            for (&idx, &val) in indices.iter().zip(values.iter()) {
                postings.entry(idx).or_default().push((doc_id as u32, val));
            }
        }

        // Sort each posting list by doc_id
        for list in postings.values_mut() {
            list.sort_by_key(|(doc_id, _)| *doc_id);
        }

        Self { postings }
    }

    fn search(&self, query_indices: &[u32], query_values: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut scores: rustc_hash::FxHashMap<u32, f32> = rustc_hash::FxHashMap::default();

        for (&idx, &weight) in query_indices.iter().zip(query_values.iter()) {
            if let Some(postings) = self.postings.get(&idx) {
                for &(doc_id, doc_weight) in postings {
                    *scores.entry(doc_id).or_insert(0.0) += weight * doc_weight;
                }
            }
        }

        let mut results: Vec<(u32, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    fn size_bytes(&self) -> usize {
        let mut size = 0;
        for list in self.postings.values() {
            size += list.len() * (4 + 4); // doc_id + weight
        }
        size
    }
}

/// Block-based sparse index using BlockSparsePostingList
struct BlockSparseIndex {
    postings: rustc_hash::FxHashMap<u32, BlockSparsePostingList>,
}

impl BlockSparseIndex {
    fn build(vectors: &[(Vec<u32>, Vec<f32>)], quantization: WeightQuantization) -> Self {
        // First, collect all postings per dimension
        let mut raw_postings: rustc_hash::FxHashMap<u32, Vec<(u32, f32)>> =
            rustc_hash::FxHashMap::default();

        for (doc_id, (indices, values)) in vectors.iter().enumerate() {
            for (&idx, &val) in indices.iter().zip(values.iter()) {
                raw_postings
                    .entry(idx)
                    .or_default()
                    .push((doc_id as u32, val));
            }
        }

        // Build block posting lists
        let mut postings = rustc_hash::FxHashMap::default();
        for (dim, mut list) in raw_postings {
            list.sort_by_key(|(doc_id, _)| *doc_id);
            if let Ok(block_list) = BlockSparsePostingList::from_postings(&list, quantization) {
                postings.insert(dim, block_list);
            }
        }

        Self { postings }
    }

    fn search(&self, query_indices: &[u32], query_values: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut scores: rustc_hash::FxHashMap<u32, f32> = rustc_hash::FxHashMap::default();

        for (&idx, &weight) in query_indices.iter().zip(query_values.iter()) {
            if let Some(posting_list) = self.postings.get(&idx) {
                let mut iter = posting_list.iterator();
                while !iter.is_exhausted() {
                    let doc_id = iter.doc();
                    let doc_weight = iter.weight();
                    *scores.entry(doc_id).or_insert(0.0) += weight * doc_weight;
                    iter.advance();
                }
            }
        }

        let mut results: Vec<(u32, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    fn size_bytes(&self) -> usize {
        self.postings.values().map(|p| p.size_bytes()).sum()
    }
}

fn benchmark_sparse_simple(
    vectors: &[(Vec<u32>, Vec<f32>)],
    queries: &[(Vec<u32>, Vec<f32>)],
    ground_truth: &[Vec<u32>],
) -> BenchmarkResult {
    let n = vectors.len();
    let k = 100;

    // Build index
    let build_start = Instant::now();
    let index = SimpleSparseIndex::build(vectors);
    let build_time = build_start.elapsed();

    // Query and measure latency
    let mut latencies = Vec::with_capacity(queries.len());
    let mut total_recall_10 = 0.0f32;
    let mut total_recall_100 = 0.0f32;

    for (i, (indices, values)) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = index.search(indices, values, k);
        let latency = start.elapsed();
        latencies.push(latency.as_micros() as f64);

        let predicted: Vec<usize> = results.iter().map(|(idx, _)| *idx as usize).collect();
        total_recall_10 += compute_recall(&predicted, &ground_truth[i], 10);
        total_recall_100 += compute_recall(&predicted, &ground_truth[i], 100);
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p99_idx = (latencies.len() as f64 * 0.99) as usize;
    let p99_latency = latencies.get(p99_idx).copied().unwrap_or(avg_latency);

    BenchmarkResult {
        name: "Sparse (Simple)".to_string(),
        num_vectors: n,
        build_time,
        merge_time: None,
        avg_query_latency_us: avg_latency,
        p99_query_latency_us: p99_latency,
        recall_at_10: total_recall_10 / queries.len() as f32,
        recall_at_100: total_recall_100 / queries.len() as f32,
        index_size_bytes: index.size_bytes(),
    }
}

fn benchmark_sparse_block(
    vectors: &[(Vec<u32>, Vec<f32>)],
    queries: &[(Vec<u32>, Vec<f32>)],
    ground_truth: &[Vec<u32>],
    quantization: WeightQuantization,
) -> BenchmarkResult {
    let n = vectors.len();
    let k = 100;

    let quant_name = match quantization {
        WeightQuantization::Float32 => "f32",
        WeightQuantization::Float16 => "f16",
        WeightQuantization::UInt8 => "u8",
        WeightQuantization::UInt4 => "u4",
    };

    // Build index
    let build_start = Instant::now();
    let index = BlockSparseIndex::build(vectors, quantization);
    let build_time = build_start.elapsed();

    // Query and measure latency
    let mut latencies = Vec::with_capacity(queries.len());
    let mut total_recall_10 = 0.0f32;
    let mut total_recall_100 = 0.0f32;

    for (i, (indices, values)) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = index.search(indices, values, k);
        let latency = start.elapsed();
        latencies.push(latency.as_micros() as f64);

        let predicted: Vec<usize> = results.iter().map(|(idx, _)| *idx as usize).collect();
        total_recall_10 += compute_recall(&predicted, &ground_truth[i], 10);
        total_recall_100 += compute_recall(&predicted, &ground_truth[i], 100);
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p99_idx = (latencies.len() as f64 * 0.99) as usize;
    let p99_latency = latencies.get(p99_idx).copied().unwrap_or(avg_latency);

    BenchmarkResult {
        name: format!("Sparse Block ({})", quant_name),
        num_vectors: n,
        build_time,
        merge_time: None,
        avg_query_latency_us: avg_latency,
        p99_query_latency_us: p99_latency,
        recall_at_10: total_recall_10 / queries.len() as f32,
        recall_at_100: total_recall_100 / queries.len() as f32,
        index_size_bytes: index.size_bytes(),
    }
}

// ============================================================================
// Criterion Benchmarks
// ============================================================================

fn bench_dense_search(c: &mut Criterion) {
    let data_dir = std::env::var("BENCHMARK_DATA").unwrap_or_else(|_| {
        println!("BENCHMARK_DATA not set, using default path");
        "benches/benchmark_data".to_string()
    });
    let data_path = Path::new(&data_dir);

    let dense_docs_path = data_path.join("dense_embeddings.bin");
    let dense_queries_path = data_path.join("dense_queries.bin");
    let dense_gt_path = data_path.join("ground_truth_dense.bin");

    let (doc_vectors, dim) = match load_dense_embeddings(&dense_docs_path) {
        Some(v) => v,
        None => {
            println!("Could not load dense embeddings from {:?}", dense_docs_path);
            println!("Run: python benches/generate_benchmark_data.py first");
            return;
        }
    };

    let (query_vectors, _) = match load_dense_embeddings(&dense_queries_path) {
        Some(v) => v,
        None => {
            println!("Could not load dense queries");
            return;
        }
    };

    let ground_truth = match load_ground_truth(&dense_gt_path) {
        Some(v) => v,
        None => {
            println!("Could not load dense ground truth");
            return;
        }
    };

    println!("\n========================================");
    println!("Dense Vector Benchmark");
    println!("========================================");
    println!("Documents: {}", doc_vectors.len());
    println!("Queries: {}", query_vectors.len());
    println!("Dimension: {}", dim);

    let mut results = Vec::new();

    // RaBitQ (flat)
    results.push(benchmark_dense_rabitq(
        &doc_vectors,
        &query_vectors,
        &ground_truth,
        dim,
    ));

    // IVF-RaBitQ with different nprobe values
    for nprobe in [8, 16, 32, 64] {
        results.push(benchmark_dense_ivf_rabitq(
            &doc_vectors,
            &query_vectors,
            &ground_truth,
            dim,
            nprobe,
        ));
    }

    // Merge benchmark
    let merge_time = benchmark_dense_ivf_merge(&doc_vectors, dim, 10);
    println!("\nIVF Merge (10 segments): {:?}", merge_time);

    print_results(&results);

    // Criterion micro-benchmarks
    let config = RaBitQConfig::new(dim);
    let rabitq_index = RaBitQIndex::build(config, &doc_vectors, true);

    c.bench_function("dense_rabitq_search", |b| {
        b.iter(|| {
            let q = &query_vectors[0];
            black_box(rabitq_index.search(q, 10, 3))
        })
    });

    let num_clusters = 256;
    let centroids = CoarseCentroids::train(&doc_vectors, num_clusters, 10, 42);
    let ivf_config = IVFConfig::new(dim);
    let ivf_index = IVFRaBitQIndex::build(ivf_config, &centroids, &doc_vectors, None);

    c.bench_function("dense_ivf_search_nprobe16", |b| {
        b.iter(|| {
            let q = &query_vectors[0];
            black_box(ivf_index.search(&centroids, q, 10, 16))
        })
    });
}

fn bench_sparse_search(c: &mut Criterion) {
    let data_dir =
        std::env::var("BENCHMARK_DATA").unwrap_or_else(|_| "benches/benchmark_data".to_string());
    let data_path = Path::new(&data_dir);

    let sparse_docs_path = data_path.join("sparse_embeddings.bin");
    let sparse_queries_path = data_path.join("sparse_queries.bin");
    let sparse_gt_path = data_path.join("ground_truth_sparse.bin");

    let doc_vectors = match load_sparse_embeddings(&sparse_docs_path) {
        Some(v) => v,
        None => {
            println!(
                "Could not load sparse embeddings from {:?}",
                sparse_docs_path
            );
            println!("Run: python benches/generate_benchmark_data.py first");
            return;
        }
    };

    let query_vectors = match load_sparse_embeddings(&sparse_queries_path) {
        Some(v) => v,
        None => {
            println!("Could not load sparse queries");
            return;
        }
    };

    let ground_truth = match load_ground_truth(&sparse_gt_path) {
        Some(v) => v,
        None => {
            println!("Could not load sparse ground truth");
            return;
        }
    };

    println!("\n========================================");
    println!("Sparse Vector Benchmark (SPLADE)");
    println!("========================================");
    println!("Documents: {}", doc_vectors.len());
    println!("Queries: {}", query_vectors.len());

    let mut results = Vec::new();

    // Simple sparse index
    results.push(benchmark_sparse_simple(
        &doc_vectors,
        &query_vectors,
        &ground_truth,
    ));

    // Block sparse with different quantizations
    for quant in [
        WeightQuantization::Float32,
        WeightQuantization::Float16,
        WeightQuantization::UInt8,
    ] {
        results.push(benchmark_sparse_block(
            &doc_vectors,
            &query_vectors,
            &ground_truth,
            quant,
        ));
    }

    print_results(&results);

    // Criterion micro-benchmarks
    let simple_index = SimpleSparseIndex::build(&doc_vectors);
    let block_index = BlockSparseIndex::build(&doc_vectors, WeightQuantization::UInt8);

    c.bench_function("sparse_simple_search", |b| {
        let (indices, values) = &query_vectors[0];
        b.iter(|| black_box(simple_index.search(indices, values, 10)))
    });

    c.bench_function("sparse_block_u8_search", |b| {
        let (indices, values) = &query_vectors[0];
        b.iter(|| black_box(block_index.search(indices, values, 10)))
    });
}

fn bench_indexing(c: &mut Criterion) {
    let data_dir =
        std::env::var("BENCHMARK_DATA").unwrap_or_else(|_| "benches/benchmark_data".to_string());
    let data_path = Path::new(&data_dir);

    // Dense indexing
    if let Some((doc_vectors, dim)) = load_dense_embeddings(&data_path.join("dense_embeddings.bin"))
    {
        let subset: Vec<Vec<f32>> = doc_vectors.into_iter().take(10000).collect();

        c.bench_function("dense_rabitq_build_10k", |b| {
            b.iter(|| {
                let config = RaBitQConfig::new(dim);
                black_box(RaBitQIndex::build(config, &subset, true))
            })
        });

        let num_clusters = 100;
        let centroids = CoarseCentroids::train(&subset, num_clusters, 10, 42);

        c.bench_function("dense_ivf_build_10k", |b| {
            b.iter(|| {
                let config = IVFConfig::new(dim);
                black_box(IVFRaBitQIndex::build(config, &centroids, &subset, None))
            })
        });
    }

    // Sparse indexing
    if let Some(doc_vectors) = load_sparse_embeddings(&data_path.join("sparse_embeddings.bin")) {
        let subset: Vec<(Vec<u32>, Vec<f32>)> = doc_vectors.into_iter().take(10000).collect();

        c.bench_function("sparse_simple_build_10k", |b| {
            b.iter(|| black_box(SimpleSparseIndex::build(&subset)))
        });

        c.bench_function("sparse_block_u8_build_10k", |b| {
            b.iter(|| black_box(BlockSparseIndex::build(&subset, WeightQuantization::UInt8)))
        });
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = bench_dense_search, bench_sparse_search, bench_indexing
);

criterion_main!(benches);
