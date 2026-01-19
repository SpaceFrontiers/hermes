//! Benchmark metrics computation

use std::collections::HashSet;
use std::time::Duration;

/// Latency statistics
#[derive(Clone, Debug)]
pub struct LatencyStats {
    pub avg_us: f64,
}

impl LatencyStats {
    pub fn from_durations(durations: &[Duration]) -> Self {
        if durations.is_empty() {
            return Self { avg_us: 0.0 };
        }

        let micros: Vec<f64> = durations.iter().map(|d| d.as_micros() as f64).collect();
        let avg_us = micros.iter().sum::<f64>() / micros.len() as f64;

        Self { avg_us }
    }
}

/// Compute MRR (Mean Reciprocal Rank) against relevance judgments
pub fn mrr(predicted: &[usize], relevant: &[u32]) -> f32 {
    let relevant_set: HashSet<u32> = relevant.iter().copied().collect();
    for (rank, &idx) in predicted.iter().enumerate() {
        if relevant_set.contains(&(idx as u32)) {
            return 1.0 / (rank + 1) as f32;
        }
    }
    0.0
}

/// Compute NDCG@k (Normalized Discounted Cumulative Gain)
pub fn ndcg_at_k(predicted: &[usize], relevant: &[u32], k: usize) -> f32 {
    let relevant_set: HashSet<u32> = relevant.iter().copied().collect();

    // DCG
    let mut dcg = 0.0f32;
    for (i, &idx) in predicted.iter().take(k).enumerate() {
        if relevant_set.contains(&(idx as u32)) {
            dcg += 1.0 / (i as f32 + 2.0).log2();
        }
    }

    // Ideal DCG (all relevant docs at top)
    let num_relevant = relevant.len().min(k);
    let mut idcg = 0.0f32;
    for i in 0..num_relevant {
        idcg += 1.0 / (i as f32 + 2.0).log2();
    }

    if idcg > 0.0 { dcg / idcg } else { 0.0 }
}
