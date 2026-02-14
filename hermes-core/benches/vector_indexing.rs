//! Vector Indexing Benchmark
//!
//! Measures recall and latency for RaBitQ, IVF-RaBitQ, and ScaNN (IVF-PQ) vector search.
//! Can use synthetic random vectors or real embeddings from a binary file.
//!
//! To generate real embeddings using Qwen3-Embedding:
//!   python benches/generate_embeddings.py -n 100000 -d 128 -o benches/embeddings.bin
//!
//! Then run benchmark with:
//!   EMBEDDINGS_FILE=benches/embeddings.bin cargo bench --bench vector_indexing

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::prelude::*;

use hermes_core::structures::{
    CoarseCentroids, CoarseConfig, IVFPQConfig, IVFPQIndex, IVFRaBitQConfig, IVFRaBitQIndex,
    PQCodebook, PQConfig, RaBitQCodebook, RaBitQConfig, RaBitQIndex,
};

const DEFAULT_DIM: usize = 128; // Matryoshka truncated dimension

/// Load embeddings from binary file
/// Format: num_vectors (u32), dim (u32), vectors (f32 * num_vectors * dim)
fn load_embeddings(path: &Path) -> Option<(Vec<Vec<f32>>, usize)> {
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
        "Loaded {} vectors of dim {} from {:?}",
        num_vectors, dim, path
    );
    Some((vectors, dim))
}

/// Get vectors - either from file or generate random
fn get_vectors(n: usize, dim: usize, seed: u64) -> (Vec<Vec<f32>>, usize) {
    // Check for EMBEDDINGS_FILE environment variable
    if let Ok(path) = std::env::var("EMBEDDINGS_FILE")
        && let Some((vecs, loaded_dim)) = load_embeddings(Path::new(&path))
    {
        let actual_n = n.min(vecs.len());
        return (vecs.into_iter().take(actual_n).collect(), loaded_dim);
    }

    // Fall back to random vectors
    println!("Using random vectors (set EMBEDDINGS_FILE to use real embeddings)");
    (generate_vectors(n, dim, seed), dim)
}

/// Generate random normalized vectors simulating embeddings
fn generate_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);

    (0..n)
        .map(|_| {
            let mut vec: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for x in &mut vec {
                    *x /= norm;
                }
            }
            vec
        })
        .collect()
}

/// Compute exact k-nearest neighbors (brute force)
fn exact_knn(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut distances: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dist: f32 = query.iter().zip(v).map(|(a, b)| (a - b).powi(2)).sum();
            (i, dist)
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.truncate(k);
    distances.into_iter().map(|(i, _)| i).collect()
}

/// Compute recall@k
fn compute_recall(predicted: &[usize], ground_truth: &[usize]) -> f32 {
    let predicted_set: std::collections::HashSet<_> = predicted.iter().collect();
    let correct = ground_truth
        .iter()
        .filter(|x| predicted_set.contains(x))
        .count();
    correct as f32 / ground_truth.len() as f32
}

/// Benchmark RaBitQ indexing and search
fn bench_rabitq(c: &mut Criterion) {
    let sizes = [1_000, 10_000, 100_000];
    let k = 10;
    let num_queries = 100;

    for &n in &sizes {
        let (vectors, dim) = get_vectors(n, DEFAULT_DIM, 42);
        let queries = generate_vectors(num_queries, dim, 123);

        // Build index
        let build_start = Instant::now();
        let config = RaBitQConfig::new(dim);
        let index = RaBitQIndex::build(config, &vectors);
        let build_time = build_start.elapsed();

        // Compute ground truth
        let ground_truths: Vec<Vec<usize>> =
            queries.iter().map(|q| exact_knn(q, &vectors, k)).collect();

        // Measure recall and latency
        let mut total_recall = 0.0;
        let search_start = Instant::now();

        for (i, query) in queries.iter().enumerate() {
            let results = index.search(query, k);
            let predicted: Vec<usize> = results
                .iter()
                .map(|(doc_id, _, _)| *doc_id as usize)
                .collect();
            total_recall += compute_recall(&predicted, &ground_truths[i]);
        }

        let search_time = search_start.elapsed();
        let avg_recall = total_recall / num_queries as f32;
        let avg_latency_us = search_time.as_micros() as f64 / num_queries as f64;

        println!("\n=== RaBitQ (n={}, dim={}) ===", n, dim);
        println!("Build time: {:?}", build_time);
        println!("Avg search latency: {:.1} µs", avg_latency_us);
        println!("Recall@{}: {:.2}%", k, avg_recall * 100.0);

        // Criterion benchmark for search
        let bench_name = format!("rabitq_search_n{}", n);
        c.bench_function(&bench_name, |b| {
            b.iter(|| {
                let q = &queries[0];
                black_box(index.search(q, k))
            })
        });
    }
}

/// Benchmark IVF-RaBitQ indexing and search
fn bench_ivf_rabitq(c: &mut Criterion) {
    let n = 100_000;
    let k = 10;
    let num_queries = 100;
    let num_clusters = 256; // sqrt(100K) ≈ 316, use 256

    let (vectors, dim) = get_vectors(n, DEFAULT_DIM, 42);
    let queries = generate_vectors(num_queries, dim, 123);

    // Train centroids
    println!("\n=== IVF-RaBitQ Training (dim={}) ===", dim);
    let train_start = Instant::now();
    let coarse_config = CoarseConfig::new(dim, num_clusters)
        .with_max_iters(10)
        .with_seed(42);
    let centroids = CoarseCentroids::train(&coarse_config, &vectors);
    let train_time = train_start.elapsed();
    println!(
        "Centroid training time ({} clusters): {:?}",
        num_clusters, train_time
    );

    // Create RaBitQ codebook
    let rabitq_config = RaBitQConfig::new(dim);
    let codebook = RaBitQCodebook::new(rabitq_config);

    // Build index
    let build_start = Instant::now();
    let config = IVFRaBitQConfig::new(dim);
    let index = IVFRaBitQIndex::build(config, &centroids, &codebook, &vectors, None);
    let build_time = build_start.elapsed();
    println!("Index build time: {:?}", build_time);

    // Compute ground truth
    let ground_truths: Vec<Vec<usize>> =
        queries.iter().map(|q| exact_knn(q, &vectors, k)).collect();

    // Test different nprobe values
    for nprobe in [8, 16, 32, 64] {
        let mut total_recall = 0.0;
        let search_start = Instant::now();

        for (i, query) in queries.iter().enumerate() {
            let results = index.search(&centroids, &codebook, query, k, Some(nprobe));
            let predicted: Vec<usize> = results.iter().map(|(idx, _, _)| *idx as usize).collect();
            total_recall += compute_recall(&predicted, &ground_truths[i]);
        }

        let search_time = search_start.elapsed();
        let avg_recall = total_recall / num_queries as f32;
        let avg_latency_us = search_time.as_micros() as f64 / num_queries as f64;

        println!("\n--- IVF-RaBitQ (nprobe={}) ---", nprobe);
        println!("Avg search latency: {:.1} µs", avg_latency_us);
        println!("Recall@{}: {:.2}%", k, avg_recall * 100.0);

        // Criterion benchmark
        let bench_name = format!("ivf_rabitq_search_nprobe{}", nprobe);
        c.bench_function(&bench_name, |b| {
            b.iter(|| {
                let q = &queries[0];
                black_box(index.search(&centroids, &codebook, q, k, Some(nprobe)))
            })
        });
    }
}

/// Benchmark IVF merge performance
fn bench_ivf_merge(c: &mut Criterion) {
    let n_per_segment = 10_000;
    let num_segments = 10;
    let num_clusters = 128;

    println!("\n=== IVF Merge Benchmark ===");

    // Train shared centroids
    let (all_vectors, dim) = get_vectors(n_per_segment, DEFAULT_DIM, 42);
    let coarse_config = CoarseConfig::new(dim, num_clusters)
        .with_max_iters(10)
        .with_seed(42);
    let centroids = CoarseCentroids::train(&coarse_config, &all_vectors);

    // Create RaBitQ codebook
    let rabitq_config = RaBitQConfig::new(dim);
    let codebook = RaBitQCodebook::new(rabitq_config);

    // Build multiple segments
    let mut segments = Vec::new();
    for i in 0..num_segments {
        let vectors = generate_vectors(n_per_segment, dim, 100 + i as u64);
        let config = IVFRaBitQConfig::new(dim);
        let index = IVFRaBitQIndex::build(config, &centroids, &codebook, &vectors, None);
        segments.push(index);
    }

    let refs: Vec<&IVFRaBitQIndex> = segments.iter().collect();
    let offsets: Vec<u32> = (0..num_segments)
        .map(|i| (i * n_per_segment) as u32)
        .collect();

    // Benchmark merge
    c.bench_function("ivf_merge_10_segments", |b| {
        b.iter(|| black_box(IVFRaBitQIndex::merge(&refs, &offsets).unwrap()))
    });

    let merge_start = Instant::now();
    let merged = IVFRaBitQIndex::merge(&refs, &offsets).unwrap();
    let merge_time = merge_start.elapsed();

    println!(
        "Merged {} segments ({} vectors each) in {:?}",
        num_segments, n_per_segment, merge_time
    );
    println!("Total vectors in merged index: {}", merged.len());
}

/// Benchmark Matryoshka/MRL dimension trimming recall
/// Tests how recall degrades as we use fewer dimensions for indexing
fn bench_mlr_recall(_c: &mut Criterion) {
    let n = 50_000;
    let full_dim = 1024; // Full embedding dimension
    let k = 10;
    let num_queries = 100;

    // MRL dimensions to test (typical matryoshka dimensions)
    let mrl_dims = [64, 128, 256, 512, 768, 1024];

    println!("\n========================================");
    println!(
        "Matryoshka/MRL Recall Benchmark (n={}, full_dim={})",
        n, full_dim
    );
    println!("========================================");

    // Generate full-dimension vectors
    let vectors = generate_vectors(n, full_dim, 42);
    let queries = generate_vectors(num_queries, full_dim, 123);

    // Ground truth using full dimensions
    let ground_truths_full: Vec<Vec<usize>> =
        queries.iter().map(|q| exact_knn(q, &vectors, k)).collect();

    println!(
        "\n{:<10} {:>12} {:>12} {:>15} {:>12}",
        "mrl_dim", "Recall@10", "vs Full", "Latency (µs)", "Speedup"
    );
    println!("{}", "-".repeat(65));

    let mut full_dim_latency = 0.0;

    for &mrl_dim in &mrl_dims {
        // Trim vectors to mrl_dim
        let trimmed_vectors: Vec<Vec<f32>> =
            vectors.iter().map(|v| v[..mrl_dim].to_vec()).collect();
        let trimmed_queries: Vec<Vec<f32>> =
            queries.iter().map(|q| q[..mrl_dim].to_vec()).collect();

        // Build index with trimmed vectors
        let config = RaBitQConfig::new(mrl_dim);
        let index = RaBitQIndex::build(config, &trimmed_vectors);

        // Ground truth for trimmed dimension (what the index actually returns)
        let ground_truths_trimmed: Vec<Vec<usize>> = trimmed_queries
            .iter()
            .map(|q| exact_knn(q, &trimmed_vectors, k))
            .collect();

        // Measure recall against full-dimension ground truth
        // This shows how much recall we lose by using fewer dimensions
        let mut recall_vs_full = 0.0;
        let mut recall_vs_trimmed = 0.0;
        let search_start = Instant::now();

        for (i, query) in trimmed_queries.iter().enumerate() {
            let results = index.search(query, k);
            let predicted: Vec<usize> = results
                .iter()
                .map(|(doc_id, _, _)| *doc_id as usize)
                .collect();
            recall_vs_full += compute_recall(&predicted, &ground_truths_full[i]);
            recall_vs_trimmed += compute_recall(&predicted, &ground_truths_trimmed[i]);
        }

        let search_time = search_start.elapsed();
        let avg_latency_us = search_time.as_micros() as f64 / num_queries as f64;
        let avg_recall_vs_full = recall_vs_full / num_queries as f32 * 100.0;
        let _avg_recall_vs_trimmed = recall_vs_trimmed / num_queries as f32 * 100.0;

        if mrl_dim == full_dim {
            full_dim_latency = avg_latency_us;
        }

        let speedup = if full_dim_latency > 0.0 && mrl_dim < full_dim {
            full_dim_latency / avg_latency_us
        } else {
            1.0
        };

        let recall_diff = if mrl_dim == full_dim {
            "baseline".to_string()
        } else {
            format!("{:+.1}%", avg_recall_vs_full - 100.0)
        };

        println!(
            "{:<10} {:>11.1}% {:>12} {:>14.1} {:>11.2}x",
            mrl_dim, avg_recall_vs_full, recall_diff, avg_latency_us, speedup
        );
    }

    println!("\nNote: Recall is measured against full-dimension ground truth.");
    println!("Lower mrl_dim = faster search but potentially lower recall.");
}

/// Compare RaBitQ vs IVF-RaBitQ vs ScaNN
fn bench_comparison(_c: &mut Criterion) {
    let n = 50_000;
    let k = 10;
    let num_queries = 100;

    let (vectors, dim) = get_vectors(n, DEFAULT_DIM, 42);
    let queries = generate_vectors(num_queries, dim, 123);

    // Ground truth
    let ground_truths: Vec<Vec<usize>> =
        queries.iter().map(|q| exact_knn(q, &vectors, k)).collect();

    println!("\n========================================");
    println!("Comparison: RaBitQ vs IVF-RaBitQ vs ScaNN (n={})", n);
    println!("========================================");

    // RaBitQ
    let rabitq_config = RaBitQConfig::new(dim);
    let rabitq_build_start = Instant::now();
    let rabitq_index = RaBitQIndex::build(rabitq_config.clone(), &vectors);
    let rabitq_build_time = rabitq_build_start.elapsed();

    let mut rabitq_recall = 0.0;
    let rabitq_search_start = Instant::now();
    for (i, query) in queries.iter().enumerate() {
        let results = rabitq_index.search(query, k);
        let predicted: Vec<usize> = results
            .iter()
            .map(|(doc_id, _, _)| *doc_id as usize)
            .collect();
        rabitq_recall += compute_recall(&predicted, &ground_truths[i]);
    }
    let rabitq_search_time = rabitq_search_start.elapsed();

    println!("\nRaBitQ:");
    println!("  Build time: {:?}", rabitq_build_time);
    println!(
        "  Avg latency: {:.1} µs",
        rabitq_search_time.as_micros() as f64 / num_queries as f64
    );
    println!(
        "  Recall@{}: {:.2}%",
        k,
        rabitq_recall / num_queries as f32 * 100.0
    );

    // IVF-RaBitQ
    let num_clusters = 128;
    let coarse_config = CoarseConfig::new(dim, num_clusters)
        .with_max_iters(10)
        .with_seed(42);
    let centroids = CoarseCentroids::train(&coarse_config, &vectors);
    let rabitq_codebook = RaBitQCodebook::new(rabitq_config);

    let ivf_config = IVFRaBitQConfig::new(dim);
    let ivf_build_start = Instant::now();
    let ivf_index = IVFRaBitQIndex::build(ivf_config, &centroids, &rabitq_codebook, &vectors, None);
    let ivf_build_time = ivf_build_start.elapsed();

    for nprobe in [16, 32] {
        let mut ivf_recall = 0.0;
        let ivf_search_start = Instant::now();
        for (i, query) in queries.iter().enumerate() {
            let results = ivf_index.search(&centroids, &rabitq_codebook, query, k, Some(nprobe));
            let predicted: Vec<usize> = results.iter().map(|(idx, _, _)| *idx as usize).collect();
            ivf_recall += compute_recall(&predicted, &ground_truths[i]);
        }
        let ivf_search_time = ivf_search_start.elapsed();

        println!("\nIVF-RaBitQ (nprobe={}):", nprobe);
        println!(
            "  Build time: {:?} (+ {:?} training)",
            ivf_build_time, ivf_build_time
        );
        println!(
            "  Avg latency: {:.1} µs",
            ivf_search_time.as_micros() as f64 / num_queries as f64
        );
        println!(
            "  Recall@{}: {:.2}%",
            k,
            ivf_recall / num_queries as f32 * 100.0
        );
    }
}

/// Benchmark ScaNN (IVF-PQ) indexing and search
fn bench_scann(c: &mut Criterion) {
    let n = 100_000;
    let k = 10;
    let num_queries = 100;
    let num_clusters = 256;

    let (vectors, dim) = get_vectors(n, DEFAULT_DIM, 42);
    let queries = generate_vectors(num_queries, dim, 123);

    println!("\n=== ScaNN (IVF-PQ) Training (dim={}) ===", dim);

    // Train coarse centroids
    let train_start = Instant::now();
    let coarse_config = CoarseConfig::new(dim, num_clusters)
        .with_max_iters(10)
        .with_seed(42);
    let centroids = CoarseCentroids::train(&coarse_config, &vectors);
    let centroid_time = train_start.elapsed();
    println!(
        "Centroid training time ({} clusters): {:?}",
        num_clusters, centroid_time
    );

    // Train PQ codebook (balanced config: 16 subspaces for 128D = 8 dims each)
    let pq_config = PQConfig::new_balanced(dim);
    println!(
        "PQ config: {} subspaces, {} dims each",
        pq_config.num_subspaces, pq_config.dims_per_block
    );
    let codebook_start = Instant::now();
    let pq_codebook = PQCodebook::train(pq_config, &vectors, 25);
    let codebook_time = codebook_start.elapsed();
    println!("PQ codebook training time: {:?}", codebook_time);

    // Build index
    let build_start = Instant::now();
    let config = IVFPQConfig::new(dim);
    let index = IVFPQIndex::build(config, &centroids, &pq_codebook, &vectors, None);
    let build_time = build_start.elapsed();
    println!("Index build time: {:?}", build_time);

    // Compute ground truth
    let ground_truths: Vec<Vec<usize>> =
        queries.iter().map(|q| exact_knn(q, &vectors, k)).collect();

    // Test different nprobe values
    for nprobe in [8, 16, 32, 64] {
        let mut total_recall = 0.0;
        let search_start = Instant::now();

        for (i, query) in queries.iter().enumerate() {
            let results = index.search(&centroids, &pq_codebook, query, k, Some(nprobe));
            let predicted: Vec<usize> = results.iter().map(|(idx, _, _)| *idx as usize).collect();
            total_recall += compute_recall(&predicted, &ground_truths[i]);
        }

        let search_time = search_start.elapsed();
        let avg_recall = total_recall / num_queries as f32;
        let avg_latency_us = search_time.as_micros() as f64 / num_queries as f64;

        println!("\n--- ScaNN (nprobe={}) ---", nprobe);
        println!("Avg search latency: {:.1} µs", avg_latency_us);
        println!("Recall@{}: {:.2}%", k, avg_recall * 100.0);

        // Criterion benchmark
        let bench_name = format!("scann_search_nprobe{}", nprobe);
        c.bench_function(&bench_name, |b| {
            b.iter(|| {
                let q = &queries[0];
                black_box(index.search(&centroids, &pq_codebook, q, k, Some(nprobe)))
            })
        });
    }
}

/// Benchmark ScaNN merge performance
fn bench_scann_merge(c: &mut Criterion) {
    let n_per_segment = 10_000;
    let num_segments = 10;
    let num_clusters = 128;

    println!("\n=== ScaNN Merge Benchmark ===");

    // Train shared centroids and codebook
    let (all_vectors, dim) = get_vectors(n_per_segment, DEFAULT_DIM, 42);
    let coarse_config = CoarseConfig::new(dim, num_clusters)
        .with_max_iters(10)
        .with_seed(42);
    let centroids = CoarseCentroids::train(&coarse_config, &all_vectors);
    let pq_config = PQConfig::new_balanced(dim);
    let pq_codebook = PQCodebook::train(pq_config, &all_vectors, 25);

    // Build multiple segments
    let mut segments = Vec::new();
    for i in 0..num_segments {
        let vectors = generate_vectors(n_per_segment, dim, 100 + i as u64);
        let config = IVFPQConfig::new(dim);
        let index = IVFPQIndex::build(config, &centroids, &pq_codebook, &vectors, None);
        segments.push(index);
    }

    let refs: Vec<&IVFPQIndex> = segments.iter().collect();
    let offsets: Vec<u32> = (0..num_segments)
        .map(|i| (i * n_per_segment) as u32)
        .collect();

    // Benchmark merge
    c.bench_function("scann_merge_10_segments", |b| {
        b.iter(|| black_box(IVFPQIndex::merge(&refs, &offsets).unwrap()))
    });

    let merge_start = Instant::now();
    let merged = IVFPQIndex::merge(&refs, &offsets).unwrap();
    let merge_time = merge_start.elapsed();

    println!(
        "Merged {} segments ({} vectors each) in {:?}",
        num_segments, n_per_segment, merge_time
    );
    println!("Total vectors in merged index: {}", merged.len());
}

criterion_group!(
    benches,
    bench_rabitq,
    bench_ivf_rabitq,
    bench_ivf_merge,
    bench_scann,
    bench_scann_merge,
    bench_mlr_recall,
    bench_comparison,
);

criterion_main!(benches);
