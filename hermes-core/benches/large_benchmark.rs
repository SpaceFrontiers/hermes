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
    BlockSparsePostingList, CoarseCentroids, CoarseConfig, IVFPQConfig, IVFPQIndex,
    IVFRaBitQConfig, IVFRaBitQIndex, PQCodebook, PQConfig, RaBitQCodebook, RaBitQConfig,
    RaBitQIndex, WeightQuantization,
};

// ============================================================================
// Data Loading
// ============================================================================

/// Target dimension for Matryoshka truncation (set via DENSE_DIM env var, default 128)
fn get_target_dim() -> usize {
    std::env::var("DENSE_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128)
}

/// Truncate vectors to target dimension and re-normalize (Matryoshka)
fn truncate_vectors(vectors: Vec<Vec<f32>>, target_dim: usize) -> Vec<Vec<f32>> {
    vectors
        .into_iter()
        .map(|v| {
            if v.len() <= target_dim {
                return v;
            }
            let truncated: Vec<f32> = v.into_iter().take(target_dim).collect();
            let norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                truncated.into_iter().map(|x| x / norm).collect()
            } else {
                truncated
            }
        })
        .collect()
}

/// Dense embeddings: (vectors, dim)
fn load_dense_embeddings(path: &Path) -> Option<(Vec<Vec<f32>>, usize)> {
    let file = File::open(path).ok()?;
    let mut reader = BufReader::new(file);

    let mut header = [0u8; 8];
    reader.read_exact(&mut header).ok()?;

    let num_vectors = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let file_dim = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

    let mut data = vec![0u8; num_vectors * file_dim * 4];
    reader.read_exact(&mut data).ok()?;

    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| {
            let offset = i * file_dim * 4;
            (0..file_dim)
                .map(|j| {
                    let idx = offset + j * 4;
                    f32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]])
                })
                .collect()
        })
        .collect();

    // Apply Matryoshka truncation
    let target_dim = get_target_dim();
    let vectors = if file_dim > target_dim {
        println!(
            "Loaded {} dense vectors (file_dim={}, truncating to {}) from {:?}",
            num_vectors, file_dim, target_dim, path
        );
        truncate_vectors(vectors, target_dim)
    } else {
        println!(
            "Loaded {} dense vectors (dim={}) from {:?}",
            num_vectors, file_dim, path
        );
        vectors
    };

    let dim = target_dim.min(file_dim);
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

/// Qrels (relevance judgments): query_idx -> [relevant_passage_idxs]
fn load_qrels(path: &Path) -> Option<std::collections::HashMap<u32, Vec<u32>>> {
    let file = File::open(path).ok()?;
    let mut reader = BufReader::new(file);

    let mut header = [0u8; 4];
    reader.read_exact(&mut header).ok()?;
    let num_queries = u32::from_le_bytes(header) as usize;

    let mut qrels = std::collections::HashMap::new();
    let mut total_relevant = 0usize;

    for _ in 0..num_queries {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf).ok()?;
        let query_idx = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let num_relevant = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
        total_relevant += num_relevant;

        let mut data = vec![0u8; num_relevant * 4];
        reader.read_exact(&mut data).ok()?;
        let relevant: Vec<u32> = (0..num_relevant)
            .map(|i| {
                let idx = i * 4;
                u32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]])
            })
            .collect();
        qrels.insert(query_idx, relevant);
    }

    println!(
        "Loaded qrels: {} queries, {} total relevant passages",
        num_queries, total_relevant
    );
    Some(qrels)
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

/// Compute MRR (Mean Reciprocal Rank) against qrels
fn compute_mrr(predicted: &[usize], relevant: &[u32]) -> f32 {
    let relevant_set: HashSet<u32> = relevant.iter().copied().collect();
    for (rank, &idx) in predicted.iter().enumerate() {
        if relevant_set.contains(&(idx as u32)) {
            return 1.0 / (rank + 1) as f32;
        }
    }
    0.0
}

/// Compute Recall@k against qrels (proportion of relevant docs found)
fn compute_recall_at_k(predicted: &[usize], relevant: &[u32], k: usize) -> f32 {
    if relevant.is_empty() {
        return 0.0;
    }
    let relevant_set: HashSet<u32> = relevant.iter().copied().collect();
    let found = predicted
        .iter()
        .take(k)
        .filter(|&idx| relevant_set.contains(&(*idx as u32)))
        .count();
    found as f32 / relevant.len() as f32
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
    // IR metrics (only set when qrels are available)
    mrr_at_10: Option<f32>,
    recall_at_1000_ir: Option<f32>,
}

fn print_results(results: &[BenchmarkResult]) {
    // Check if any result has IR metrics
    let has_ir_metrics = results.iter().any(|r| r.mrr_at_10.is_some());

    let mut table = Table::new();
    let mut headers = vec![
        Cell::new("Index").fg(Color::Cyan),
        Cell::new("Vectors").fg(Color::Cyan),
        Cell::new("Build").fg(Color::Cyan),
        Cell::new("Merge").fg(Color::Cyan),
        Cell::new("Avg Latency").fg(Color::Cyan),
        Cell::new("P99 Latency").fg(Color::Cyan),
        Cell::new("R@10").fg(Color::Cyan),
        Cell::new("R@100").fg(Color::Cyan),
        Cell::new("Size").fg(Color::Cyan),
    ];
    if has_ir_metrics {
        headers.push(Cell::new("MRR@10").fg(Color::Cyan));
        headers.push(Cell::new("IR-R@1k").fg(Color::Cyan));
    }
    table.set_header(headers);

    for r in results {
        let mut row = vec![
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
        ];
        if has_ir_metrics {
            row.push(Cell::new(
                r.mrr_at_10.map_or("-".to_string(), |v| format!("{:.3}", v)),
            ));
            row.push(Cell::new(
                r.recall_at_1000_ir
                    .map_or("-".to_string(), |v| format!("{:.1}%", v * 100.0)),
            ));
        }
        table.add_row(row);
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
    qrels: Option<&std::collections::HashMap<u32, Vec<u32>>>,
) -> BenchmarkResult {
    let n = vectors.len();
    let k = 100;

    // Build index
    let build_start = Instant::now();
    let config = RaBitQConfig::new(dim);
    let index = RaBitQIndex::build(config, vectors);
    let build_time = build_start.elapsed();

    // Query and measure latency
    let mut latencies = Vec::with_capacity(queries.len());
    let mut total_recall_10 = 0.0f32;
    let mut total_recall_100 = 0.0f32;
    let mut total_mrr = 0.0f32;
    let mut total_ir_recall = 0.0f32;
    let mut ir_query_count = 0usize;

    for (i, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = index.search(query, k, 3);
        let latency = start.elapsed();
        latencies.push(latency.as_micros() as f64);

        let predicted: Vec<usize> = results
            .iter()
            .map(|(doc_id, _, _)| *doc_id as usize)
            .collect();
        total_recall_10 += compute_recall(&predicted, &ground_truth[i], 10);
        total_recall_100 += compute_recall(&predicted, &ground_truth[i], 100);

        // IR metrics if qrels available
        if let Some(qrels) = qrels
            && let Some(relevant) = qrels.get(&(i as u32))
        {
            total_mrr += compute_mrr(&predicted, relevant);
            total_ir_recall += compute_recall_at_k(&predicted, relevant, k);
            ir_query_count += 1;
        }
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
        mrr_at_10: if ir_query_count > 0 {
            Some(total_mrr / ir_query_count as f32)
        } else {
            None
        },
        recall_at_1000_ir: if ir_query_count > 0 {
            Some(total_ir_recall / ir_query_count as f32)
        } else {
            None
        },
    }
}

fn benchmark_dense_ivf_rabitq(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth: &[Vec<u32>],
    dim: usize,
    nprobe: usize,
    qrels: Option<&std::collections::HashMap<u32, Vec<u32>>>,
) -> BenchmarkResult {
    let n = vectors.len();
    let k = 100;
    let num_clusters = ((n as f64).sqrt() as usize).clamp(16, 1024);

    // Train centroids
    let train_start = Instant::now();
    let coarse_config = CoarseConfig::new(dim, num_clusters)
        .with_max_iters(10)
        .with_seed(42);
    let centroids = CoarseCentroids::train(&coarse_config, vectors);
    let train_time = train_start.elapsed();

    // Build index
    let build_start = Instant::now();
    let config = IVFRaBitQConfig::new(dim);
    let rabitq_codebook = RaBitQCodebook::new(RaBitQConfig::new(dim));
    let index = IVFRaBitQIndex::build(config, &centroids, &rabitq_codebook, vectors, None);
    let build_time = build_start.elapsed() + train_time;

    // Query and measure latency
    let mut latencies = Vec::with_capacity(queries.len());
    let mut total_recall_10 = 0.0f32;
    let mut total_recall_100 = 0.0f32;
    let mut total_mrr = 0.0f32;
    let mut total_ir_recall = 0.0f32;
    let mut ir_query_count = 0usize;

    for (i, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = index.search(&centroids, &rabitq_codebook, query, k, Some(nprobe));
        let latency = start.elapsed();
        latencies.push(latency.as_micros() as f64);

        let predicted: Vec<usize> = results.iter().map(|(idx, _, _)| *idx as usize).collect();
        total_recall_10 += compute_recall(&predicted, &ground_truth[i], 10);
        total_recall_100 += compute_recall(&predicted, &ground_truth[i], 100);

        // IR metrics if qrels available
        if let Some(qrels) = qrels
            && let Some(relevant) = qrels.get(&(i as u32))
        {
            total_mrr += compute_mrr(&predicted, relevant);
            total_ir_recall += compute_recall_at_k(&predicted, relevant, k);
            ir_query_count += 1;
        }
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
        index_size_bytes: index.to_bytes().unwrap_or_default().len(),
        mrr_at_10: if ir_query_count > 0 {
            Some(total_mrr / ir_query_count as f32)
        } else {
            None
        },
        recall_at_1000_ir: if ir_query_count > 0 {
            Some(total_ir_recall / ir_query_count as f32)
        } else {
            None
        },
    }
}

fn benchmark_dense_scann(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth: &[Vec<u32>],
    dim: usize,
    nprobe: usize,
    qrels: Option<&std::collections::HashMap<u32, Vec<u32>>>,
) -> BenchmarkResult {
    let n = vectors.len();
    let k = 100;
    let num_clusters = ((n as f64).sqrt() as usize).clamp(16, 1024);

    // Train centroids
    let train_start = Instant::now();
    let coarse_config = CoarseConfig::new(dim, num_clusters)
        .with_max_iters(10)
        .with_seed(42);
    let centroids = CoarseCentroids::train(&coarse_config, vectors);
    let train_time = train_start.elapsed();

    // Train PQ codebook
    let pq_config = PQConfig::new_balanced(dim);
    let codebook_start = Instant::now();
    let pq_codebook = PQCodebook::train(pq_config, vectors, 25);
    let codebook_time = codebook_start.elapsed();

    // Build index
    let build_start = Instant::now();
    let config = IVFPQConfig::new(dim);
    let index = IVFPQIndex::build(config, &centroids, &pq_codebook, vectors, None);
    let build_time = build_start.elapsed() + train_time + codebook_time;

    // Query and measure latency
    let mut latencies = Vec::with_capacity(queries.len());
    let mut total_recall_10 = 0.0f32;
    let mut total_recall_100 = 0.0f32;
    let mut total_mrr = 0.0f32;
    let mut total_ir_recall = 0.0f32;
    let mut ir_query_count = 0usize;

    for (i, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = index.search(&centroids, &pq_codebook, query, k, Some(nprobe));
        let latency = start.elapsed();
        latencies.push(latency.as_micros() as f64);

        let predicted: Vec<usize> = results.iter().map(|(idx, _, _)| *idx as usize).collect();
        total_recall_10 += compute_recall(&predicted, &ground_truth[i], 10);
        total_recall_100 += compute_recall(&predicted, &ground_truth[i], 100);

        // IR metrics if qrels available
        if let Some(qrels) = qrels
            && let Some(relevant) = qrels.get(&(i as u32))
        {
            total_mrr += compute_mrr(&predicted, relevant);
            total_ir_recall += compute_recall_at_k(&predicted, relevant, k);
            ir_query_count += 1;
        }
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p99_idx = (latencies.len() as f64 * 0.99) as usize;
    let p99_latency = latencies.get(p99_idx).copied().unwrap_or(avg_latency);

    BenchmarkResult {
        name: format!("ScaNN/IVF-PQ (nprobe={})", nprobe),
        num_vectors: n,
        build_time,
        merge_time: None,
        avg_query_latency_us: avg_latency,
        p99_query_latency_us: p99_latency,
        recall_at_10: total_recall_10 / queries.len() as f32,
        recall_at_100: total_recall_100 / queries.len() as f32,
        index_size_bytes: index.to_bytes().unwrap_or_default().len(),
        mrr_at_10: if ir_query_count > 0 {
            Some(total_mrr / ir_query_count as f32)
        } else {
            None
        },
        recall_at_1000_ir: if ir_query_count > 0 {
            Some(total_ir_recall / ir_query_count as f32)
        } else {
            None
        },
    }
}

fn benchmark_dense_ivf_merge(vectors: &[Vec<f32>], dim: usize, num_segments: usize) -> Duration {
    let n = vectors.len();
    let vectors_per_segment = n / num_segments;
    let num_clusters = ((vectors_per_segment as f64).sqrt() as usize).clamp(16, 256);

    // Train shared centroids
    let coarse_config = CoarseConfig::new(dim, num_clusters)
        .with_max_iters(10)
        .with_seed(42);
    let centroids =
        CoarseCentroids::train(&coarse_config, &vectors[..vectors_per_segment.min(10000)]);
    let rabitq_codebook = RaBitQCodebook::new(RaBitQConfig::new(dim));

    // Build segments
    let mut segments = Vec::new();
    for i in 0..num_segments {
        let start = i * vectors_per_segment;
        let end = ((i + 1) * vectors_per_segment).min(n);
        let segment_vectors: Vec<Vec<f32>> = vectors[start..end].to_vec();

        let config = IVFRaBitQConfig::new(dim);
        let index =
            IVFRaBitQIndex::build(config, &centroids, &rabitq_codebook, &segment_vectors, None);
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
            // Convert (doc_id, weight) to (doc_id, ordinal, weight)
            let with_ordinals: Vec<(u32, u16, f32)> =
                list.iter().map(|(d, w)| (*d, 0u16, *w)).collect();
            if let Ok(block_list) =
                BlockSparsePostingList::from_postings(&with_ordinals, quantization)
            {
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
    qrels: Option<&std::collections::HashMap<u32, Vec<u32>>>,
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
    let mut total_mrr = 0.0f32;
    let mut total_ir_recall = 0.0f32;
    let mut ir_query_count = 0usize;

    for (i, (indices, values)) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = index.search(indices, values, k);
        let latency = start.elapsed();
        latencies.push(latency.as_micros() as f64);

        let predicted: Vec<usize> = results.iter().map(|(idx, _)| *idx as usize).collect();
        total_recall_10 += compute_recall(&predicted, &ground_truth[i], 10);
        total_recall_100 += compute_recall(&predicted, &ground_truth[i], 100);

        // IR metrics if qrels available
        if let Some(qrels) = qrels
            && let Some(relevant) = qrels.get(&(i as u32))
        {
            total_mrr += compute_mrr(&predicted, relevant);
            total_ir_recall += compute_recall_at_k(&predicted, relevant, k);
            ir_query_count += 1;
        }
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p99_idx = (latencies.len() as f64 * 0.99) as usize;
    let p99_latency = latencies.get(p99_idx).copied().unwrap_or(avg_latency);

    let (mrr_at_10, recall_at_1000_ir) = if ir_query_count > 0 {
        (
            Some(total_mrr / ir_query_count as f32),
            Some(total_ir_recall / ir_query_count as f32),
        )
    } else {
        (None, None)
    };

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
        mrr_at_10,
        recall_at_1000_ir,
    }
}

fn benchmark_sparse_block(
    vectors: &[(Vec<u32>, Vec<f32>)],
    queries: &[(Vec<u32>, Vec<f32>)],
    ground_truth: &[Vec<u32>],
    quantization: WeightQuantization,
    qrels: Option<&std::collections::HashMap<u32, Vec<u32>>>,
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
    let mut total_mrr = 0.0f32;
    let mut total_ir_recall = 0.0f32;
    let mut ir_query_count = 0usize;

    for (i, (indices, values)) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = index.search(indices, values, k);
        let latency = start.elapsed();
        latencies.push(latency.as_micros() as f64);

        let predicted: Vec<usize> = results.iter().map(|(idx, _)| *idx as usize).collect();
        total_recall_10 += compute_recall(&predicted, &ground_truth[i], 10);
        total_recall_100 += compute_recall(&predicted, &ground_truth[i], 100);

        // IR metrics if qrels available
        if let Some(qrels) = qrels
            && let Some(relevant) = qrels.get(&(i as u32))
        {
            total_mrr += compute_mrr(&predicted, relevant);
            total_ir_recall += compute_recall_at_k(&predicted, relevant, k);
            ir_query_count += 1;
        }
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let p99_idx = (latencies.len() as f64 * 0.99) as usize;
    let p99_latency = latencies.get(p99_idx).copied().unwrap_or(avg_latency);

    let (mrr_at_10, recall_at_1000_ir) = if ir_query_count > 0 {
        (
            Some(total_mrr / ir_query_count as f32),
            Some(total_ir_recall / ir_query_count as f32),
        )
    } else {
        (None, None)
    };

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
        mrr_at_10,
        recall_at_1000_ir,
    }
}

// ============================================================================
// Sparse Weight Threshold Recall Benchmark
// ============================================================================

/// Result for sparse weight threshold comparison
struct SparseThresholdResult {
    threshold: f32,
    avg_nnz: f64,
    recall_at_10: f32,
    recall_at_100: f32,
    recall_vs_full: f32,
    avg_latency_us: f64,
    speedup: f64,
    index_size_bytes: usize,
}

fn print_sparse_threshold_results(results: &[SparseThresholdResult]) {
    println!(
        "\n{:<12} {:>10} {:>12} {:>12} {:>12} {:>15} {:>12} {:>12}",
        "Threshold", "Avg NNZ", "R@10", "R@100", "vs Full", "Latency (µs)", "Speedup", "Size"
    );
    println!("{}", "-".repeat(105));

    for r in results {
        let recall_diff = if r.threshold == 0.0 {
            "baseline".to_string()
        } else {
            format!("{:+.1}%", (r.recall_vs_full - 1.0) * 100.0)
        };

        println!(
            "{:<12.4} {:>10.1} {:>11.1}% {:>11.1}% {:>12} {:>14.1} {:>11.2}x {:>10.1} KB",
            r.threshold,
            r.avg_nnz,
            r.recall_at_10 * 100.0,
            r.recall_at_100 * 100.0,
            recall_diff,
            r.avg_latency_us,
            r.speedup,
            r.index_size_bytes as f64 / 1024.0
        );
    }
}

/// Apply weight threshold to sparse vectors (filter out weights below threshold)
fn apply_weight_threshold(
    vectors: &[(Vec<u32>, Vec<f32>)],
    threshold: f32,
) -> Vec<(Vec<u32>, Vec<f32>)> {
    vectors
        .iter()
        .map(|(indices, values)| {
            let filtered: Vec<(u32, f32)> = indices
                .iter()
                .zip(values.iter())
                .filter(|&(_, v)| v.abs() >= threshold)
                .map(|(&i, &v)| (i, v))
                .collect();
            let new_indices: Vec<u32> = filtered.iter().map(|(i, _)| *i).collect();
            let new_values: Vec<f32> = filtered.iter().map(|(_, v)| *v).collect();
            (new_indices, new_values)
        })
        .collect()
}

/// Benchmark sparse weight threshold impact on recall
fn benchmark_sparse_with_threshold(
    doc_vectors: &[(Vec<u32>, Vec<f32>)],
    query_vectors: &[(Vec<u32>, Vec<f32>)],
    ground_truth: &[Vec<u32>],
    threshold: f32,
) -> SparseThresholdResult {
    let k = 100;

    // Apply threshold to document vectors (simulating indexing with threshold)
    let filtered_docs = apply_weight_threshold(doc_vectors, threshold);

    // Calculate average NNZ after filtering
    let total_nnz: usize = filtered_docs.iter().map(|(i, _)| i.len()).sum();
    let avg_nnz = total_nnz as f64 / filtered_docs.len() as f64;

    // Build index with filtered vectors
    let build_start = Instant::now();
    let index = BlockSparseIndex::build(&filtered_docs, WeightQuantization::UInt8);
    let _build_time = build_start.elapsed();

    // Query using original (unfiltered) query vectors
    let mut latencies = Vec::with_capacity(query_vectors.len());
    let mut total_recall_10 = 0.0f32;
    let mut total_recall_100 = 0.0f32;

    for (i, (indices, values)) in query_vectors.iter().enumerate() {
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

    SparseThresholdResult {
        threshold,
        avg_nnz,
        recall_at_10: total_recall_10 / query_vectors.len() as f32,
        recall_at_100: total_recall_100 / query_vectors.len() as f32,
        recall_vs_full: 1.0, // Will be normalized later
        avg_latency_us: avg_latency,
        speedup: 1.0, // Will be normalized later
        index_size_bytes: index.size_bytes(),
    }
}

// ============================================================================
// MRL (Matryoshka) Recall Benchmark
// ============================================================================

/// Result for MRL dimension comparison
struct MrlResult {
    mrl_dim: usize,
    recall_at_10: f32,
    recall_vs_full: f32,
    avg_latency_us: f64,
    speedup: f64,
}

fn print_mlr_results(results: &[MrlResult], full_dim: usize) {
    println!(
        "\n{:<10} {:>12} {:>12} {:>15} {:>12}",
        "mrl_dim", "Recall@10", "vs Full", "Latency (µs)", "Speedup"
    );
    println!("{}", "-".repeat(65));

    for r in results {
        let recall_diff = if r.mrl_dim == full_dim {
            "baseline".to_string()
        } else {
            format!("{:+.1}%", (r.recall_vs_full - 1.0) * 100.0)
        };

        println!(
            "{:<10} {:>11.1}% {:>12} {:>14.1} {:>11.2}x",
            r.mrl_dim,
            r.recall_at_10 * 100.0,
            recall_diff,
            r.avg_latency_us,
            r.speedup
        );
    }
}

/// Benchmark MRL (Matryoshka) dimension trimming with real embeddings
fn bench_mlr_recall(c: &mut Criterion) {
    let data_dir =
        std::env::var("BENCHMARK_DATA").unwrap_or_else(|_| "benches/benchmark_data".to_string());
    let data_path = Path::new(&data_dir);

    let dense_docs_path = data_path.join("dense_embeddings.bin");
    let dense_queries_path = data_path.join("dense_queries.bin");
    let dense_gt_full_path = data_path.join("ground_truth_dense_full.bin");

    // Load full-dimensional embeddings (no truncation for this benchmark)
    let file = match File::open(&dense_docs_path) {
        Ok(f) => f,
        Err(_) => {
            println!("Could not load dense embeddings from {:?}", dense_docs_path);
            println!("Run: python benches/generate_benchmark_data.py --use-triton first");
            return;
        }
    };
    let mut reader = BufReader::new(file);
    let mut header = [0u8; 8];
    if reader.read_exact(&mut header).is_err() {
        return;
    }
    let num_docs = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let full_dim = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

    let mut data = vec![0u8; num_docs * full_dim * 4];
    if reader.read_exact(&mut data).is_err() {
        return;
    }

    let doc_vectors: Vec<Vec<f32>> = (0..num_docs)
        .map(|i| {
            let offset = i * full_dim * 4;
            (0..full_dim)
                .map(|j| {
                    let idx = offset + j * 4;
                    f32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]])
                })
                .collect()
        })
        .collect();

    // Load queries similarly
    let file = match File::open(&dense_queries_path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let mut reader = BufReader::new(file);
    let mut header = [0u8; 8];
    if reader.read_exact(&mut header).is_err() {
        return;
    }
    let num_queries = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let query_dim = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

    let mut data = vec![0u8; num_queries * query_dim * 4];
    if reader.read_exact(&mut data).is_err() {
        return;
    }

    let query_vectors: Vec<Vec<f32>> = (0..num_queries)
        .map(|i| {
            let offset = i * query_dim * 4;
            (0..query_dim)
                .map(|j| {
                    let idx = offset + j * 4;
                    f32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]])
                })
                .collect()
        })
        .collect();

    // Load ground truth for full dimension
    let ground_truth = match load_ground_truth(&dense_gt_full_path) {
        Some(v) => v,
        None => {
            println!("Could not load full-dim ground truth");
            return;
        }
    };

    println!("\n========================================");
    println!("Matryoshka/MRL Recall Benchmark (MS MARCO)");
    println!("========================================");
    println!("Documents: {} (full_dim={})", doc_vectors.len(), full_dim);
    println!("Queries: {}", query_vectors.len());

    // MRL dimensions to test (Jina v3 supports: 32, 64, 128, 256, 512, 768, 1024)
    let mrl_dims: Vec<usize> = vec![64, 128, 256, 512, 768, 1024]
        .into_iter()
        .filter(|&d| d <= full_dim)
        .collect();

    let mut results = Vec::new();
    let mut full_dim_latency = 0.0f64;
    let k = 10;

    for &mrl_dim in &mrl_dims {
        // Truncate and re-normalize vectors
        let trimmed_docs: Vec<Vec<f32>> = doc_vectors
            .iter()
            .map(|v| {
                let truncated: Vec<f32> = v.iter().take(mrl_dim).copied().collect();
                let norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    truncated.into_iter().map(|x| x / norm).collect()
                } else {
                    truncated
                }
            })
            .collect();

        let trimmed_queries: Vec<Vec<f32>> = query_vectors
            .iter()
            .map(|v| {
                let truncated: Vec<f32> = v.iter().take(mrl_dim).copied().collect();
                let norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    truncated.into_iter().map(|x| x / norm).collect()
                } else {
                    truncated
                }
            })
            .collect();

        // Build index with trimmed vectors
        let config = RaBitQConfig::new(mrl_dim);
        let index = RaBitQIndex::build(config, &trimmed_docs);

        // Measure recall against full-dimension ground truth
        let mut total_recall = 0.0f32;
        let search_start = Instant::now();

        for (i, query) in trimmed_queries.iter().enumerate() {
            let search_results = index.search(query, k, 3);
            let predicted: Vec<usize> = search_results
                .iter()
                .map(|(doc_id, _, _)| *doc_id as usize)
                .collect();
            total_recall += compute_recall(&predicted, &ground_truth[i], k);
        }

        let search_time = search_start.elapsed();
        let avg_latency = search_time.as_micros() as f64 / trimmed_queries.len() as f64;
        let avg_recall = total_recall / trimmed_queries.len() as f32;

        if mrl_dim == full_dim || mrl_dim == *mrl_dims.last().unwrap() {
            full_dim_latency = avg_latency;
        }

        let speedup = if full_dim_latency > 0.0 {
            full_dim_latency / avg_latency
        } else {
            1.0
        };

        results.push(MrlResult {
            mrl_dim,
            recall_at_10: avg_recall,
            recall_vs_full: avg_recall, // Will be normalized later
            avg_latency_us: avg_latency,
            speedup,
        });
    }

    // Normalize recall_vs_full relative to highest dimension
    if let Some(full_result) = results.last() {
        let full_recall = full_result.recall_at_10;
        for r in &mut results {
            r.recall_vs_full = if full_recall > 0.0 {
                r.recall_at_10 / full_recall
            } else {
                1.0
            };
        }
    }

    // Recalculate speedup relative to highest dimension
    if let Some(full_result) = results.last() {
        let full_latency = full_result.avg_latency_us;
        for r in &mut results {
            r.speedup = full_latency / r.avg_latency_us;
        }
    }

    print_mlr_results(&results, full_dim);

    println!("\nNote: Recall is measured against full-dimension ground truth.");
    println!("Lower mrl_dim = faster search but potentially lower recall.");

    // Criterion micro-benchmark for a few key dimensions
    for &mrl_dim in &[128, 256, 512] {
        if mrl_dim > full_dim {
            continue;
        }

        let trimmed_docs: Vec<Vec<f32>> = doc_vectors
            .iter()
            .map(|v| {
                let truncated: Vec<f32> = v.iter().take(mrl_dim).copied().collect();
                let norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-10 {
                    truncated.into_iter().map(|x| x / norm).collect()
                } else {
                    truncated
                }
            })
            .collect();

        let trimmed_query: Vec<f32> = {
            let truncated: Vec<f32> = query_vectors[0].iter().take(mrl_dim).copied().collect();
            let norm: f32 = truncated.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                truncated.into_iter().map(|x| x / norm).collect()
            } else {
                truncated
            }
        };

        let config = RaBitQConfig::new(mrl_dim);
        let index = RaBitQIndex::build(config, &trimmed_docs);

        c.bench_function(&format!("mlr_rabitq_dim{}", mrl_dim), |b| {
            b.iter(|| black_box(index.search(&trimmed_query, k, 3)))
        });
    }
}

/// Benchmark sparse weight threshold impact on recall
fn bench_sparse_weight_threshold(c: &mut Criterion) {
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

    // Calculate original average NNZ
    let total_nnz: usize = doc_vectors.iter().map(|(i, _)| i.len()).sum();
    let original_avg_nnz = total_nnz as f64 / doc_vectors.len() as f64;

    println!("\n========================================");
    println!("Sparse Weight Threshold Benchmark (MS MARCO)");
    println!("========================================");
    println!("Documents: {}", doc_vectors.len());
    println!("Queries: {}", query_vectors.len());
    println!("Original avg NNZ: {:.1}", original_avg_nnz);

    // Weight thresholds to test (typical SPLADE weights range from 0 to ~3)
    let thresholds: Vec<f32> = vec![0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0];

    let mut results = Vec::new();

    for &threshold in &thresholds {
        let result =
            benchmark_sparse_with_threshold(&doc_vectors, &query_vectors, &ground_truth, threshold);
        results.push(result);
    }

    // Normalize recall_vs_full and speedup relative to threshold=0.0
    if let Some(baseline) = results.first() {
        let baseline_recall = baseline.recall_at_10;
        let baseline_latency = baseline.avg_latency_us;

        for r in &mut results {
            r.recall_vs_full = if baseline_recall > 0.0 {
                r.recall_at_10 / baseline_recall
            } else {
                1.0
            };
            r.speedup = if r.avg_latency_us > 0.0 {
                baseline_latency / r.avg_latency_us
            } else {
                1.0
            };
        }
    }

    print_sparse_threshold_results(&results);

    println!("\nNote: Recall is measured against ground truth computed with no threshold.");
    println!("Higher threshold = smaller index, faster search, but potentially lower recall.");

    // Criterion micro-benchmarks for a few key thresholds
    for &threshold in &[0.0, 0.1, 0.5] {
        let filtered_docs = apply_weight_threshold(&doc_vectors, threshold);
        let index = BlockSparseIndex::build(&filtered_docs, WeightQuantization::UInt8);

        let threshold_str = format!("{:.2}", threshold).replace('.', "_");
        c.bench_function(&format!("sparse_threshold_{}", threshold_str), |b| {
            let (indices, values) = &query_vectors[0];
            b.iter(|| black_box(index.search(indices, values, 10)))
        });
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

    // Ground truth file depends on dimension: ground_truth_dense_128.bin or ground_truth_dense_full.bin
    let target_dim = get_target_dim();
    let dense_gt_path = data_path.join(format!("ground_truth_dense_{}.bin", target_dim));
    let dense_gt_full_path = data_path.join("ground_truth_dense_full.bin");

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

    // Try dimension-specific ground truth first, then fall back to full
    let ground_truth = match load_ground_truth(&dense_gt_path) {
        Some(v) => {
            println!("Using ground truth for {}-dim embeddings", target_dim);
            v
        }
        None => match load_ground_truth(&dense_gt_full_path) {
            Some(v) => {
                println!(
                    "Using full-dimensional ground truth (no {}-dim file found)",
                    target_dim
                );
                v
            }
            None => {
                println!(
                    "Could not load dense ground truth from {:?} or {:?}",
                    dense_gt_path, dense_gt_full_path
                );
                return;
            }
        },
    };

    // Try to load qrels for IR evaluation
    let qrels_path = data_path.join("qrels.bin");
    let qrels = load_qrels(&qrels_path);
    if qrels.is_some() {
        println!("IR evaluation enabled (qrels loaded)");
    }

    println!("\n========================================");
    println!("Dense Vector Benchmark (MS MARCO)");
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
        qrels.as_ref(),
    ));

    // IVF-RaBitQ with different nprobe values (requires at least 1000 vectors)
    if doc_vectors.len() >= 1000 {
        for nprobe in [8, 16, 32, 64] {
            results.push(benchmark_dense_ivf_rabitq(
                &doc_vectors,
                &query_vectors,
                &ground_truth,
                dim,
                nprobe,
                qrels.as_ref(),
            ));
        }

        // ScaNN (IVF-PQ) with different nprobe values
        for nprobe in [8, 16, 32, 64] {
            results.push(benchmark_dense_scann(
                &doc_vectors,
                &query_vectors,
                &ground_truth,
                dim,
                nprobe,
                qrels.as_ref(),
            ));
        }
    } else {
        println!(
            "Skipping IVF benchmarks (need >= 1000 vectors, have {})",
            doc_vectors.len()
        );
    }

    // Merge benchmark
    let merge_time = benchmark_dense_ivf_merge(&doc_vectors, dim, 10);
    println!("\nIVF Merge (10 segments): {:?}", merge_time);

    print_results(&results);

    // Criterion micro-benchmarks
    let config = RaBitQConfig::new(dim);
    let rabitq_index = RaBitQIndex::build(config, &doc_vectors);

    c.bench_function("dense_rabitq_search", |b| {
        b.iter(|| {
            let q = &query_vectors[0];
            black_box(rabitq_index.search(q, 10, 3))
        })
    });

    let num_clusters = 256;
    let coarse_config = CoarseConfig::new(dim, num_clusters)
        .with_max_iters(10)
        .with_seed(42);
    let centroids = CoarseCentroids::train(&coarse_config, &doc_vectors);
    let ivf_config = IVFRaBitQConfig::new(dim);
    let rabitq_codebook = RaBitQCodebook::new(RaBitQConfig::new(dim));
    let ivf_index =
        IVFRaBitQIndex::build(ivf_config, &centroids, &rabitq_codebook, &doc_vectors, None);

    c.bench_function("dense_ivf_search_nprobe16", |b| {
        b.iter(|| {
            let q = &query_vectors[0];
            black_box(ivf_index.search(&centroids, &rabitq_codebook, q, 10, Some(16)))
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

    // Try to load qrels for IR evaluation
    let qrels_path = data_path.join("qrels.bin");
    let qrels = load_qrels(&qrels_path);
    if qrels.is_some() {
        println!("IR evaluation enabled (qrels loaded)");
    }

    println!("\n========================================");
    println!("Sparse Vector Benchmark (MS MARCO + SPLADE)");
    println!("========================================");
    println!("Documents: {}", doc_vectors.len());
    println!("Queries: {}", query_vectors.len());

    let mut results = Vec::new();

    // Simple sparse index
    results.push(benchmark_sparse_simple(
        &doc_vectors,
        &query_vectors,
        &ground_truth,
        qrels.as_ref(),
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
            qrels.as_ref(),
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
                black_box(RaBitQIndex::build(config, &subset))
            })
        });

        let num_clusters = 100;
        let coarse_config = CoarseConfig::new(dim, num_clusters)
            .with_max_iters(10)
            .with_seed(42);
        let centroids = CoarseCentroids::train(&coarse_config, &subset);
        let rabitq_codebook = RaBitQCodebook::new(RaBitQConfig::new(dim));

        c.bench_function("dense_ivf_build_10k", |b| {
            b.iter(|| {
                let config = IVFRaBitQConfig::new(dim);
                black_box(IVFRaBitQIndex::build(
                    config,
                    &centroids,
                    &rabitq_codebook,
                    &subset,
                    None,
                ))
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
    targets = bench_dense_search, bench_sparse_search, bench_sparse_weight_threshold, bench_mlr_recall, bench_indexing
);

criterion_main!(benches);
