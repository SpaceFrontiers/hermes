//! Manual experiment for the phase-two dispatch described in
//! docs/bmp-forward-search.md. This adds no retrieval policy or default.
use super::*;
use crate::directories::{FileHandle, OwnedBytes};
use std::hint::black_box;
use std::time::Instant;

#[test]
#[ignore = "manual BMP phase-two forward/inverted crossover experiment"]
fn measure_forward_completion_against_block_scoring() {
    let full_query = std::env::var_os("HERMES_FORWARD_EXPERIMENT_FULL").is_some();
    const BLOCKS: u32 = 64;
    const DIMS: u32 = 4096;
    for size in [8u32, 32, 128] {
        for clustered in [false, true] {
            let mut postings = rustc_hash::FxHashMap::default();
            for doc in 0..BLOCKS * size {
                for k in 0..64 {
                    let dim = if clustered && k < 32 {
                        k
                    } else {
                        32 + (doc * 37 + k * 61) % (DIMS - 32)
                    };
                    postings.entry(dim).or_insert_with(Vec::new).push((
                        doc,
                        0,
                        0.2 + (k % 13) as f32 * 0.1,
                    ));
                }
            }
            let mut bytes = Vec::new();
            crate::segment::build_bmp_blob(
                postings, size, 4, 0.0, None, DIMS, 5.0, 0, true, &mut bytes,
            )
            .unwrap();
            let len = bytes.len() as u64;
            let index = BmpIndex::parse(
                FileHandle::from_bytes(OwnedBytes::new(bytes)),
                0,
                len,
                BLOCKS * size,
                BLOCKS * size,
            )
            .unwrap();
            let forward = index.forward().unwrap();
            for query_len in [8, 32, 64] {
                let terms: Vec<_> = (0..query_len)
                    .map(|i| {
                        let dim = if clustered && i < 32 {
                            i as u32
                        } else {
                            32 + (i as u32 * 61) % (DIMS - 32)
                        };
                        (dim, 0.1 + (i % 7) as f32 * 0.1)
                    })
                    .collect();
                let prepared = prepare_bmp_query(DIMS, &terms, &terms).unwrap().unwrap();
                // Use the production phase-one mask (the three heaviest
                // dimensions), retaining the original quantization for phase two.
                let mask = valid_query_bits(prepared.query_by_dim_u16.len())
                    & if full_query {
                        u64::MAX
                    } else {
                        !prepared.phase1_mask
                    };
                let remaining_query: Vec<_> = prepared
                    .query_by_dim_u16
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &term)| (mask & (1 << i) != 0).then_some(term))
                    .collect();
                // Production receives a grid-derived mask of terms present in
                // this block. Do not charge it for probing absent dimensions.
                let blocks: Vec<_> = (0..BLOCKS)
                    .filter_map(|id| {
                        let block = index.parse_block(id).unwrap();
                        let present = prepared.query_by_dim_u16.iter().enumerate().fold(
                            0,
                            |bits, (i, &(dim, _))| {
                                bits | if block.find_dimension(dim).is_some() {
                                    1u64 << i
                                } else {
                                    0
                                }
                            },
                        ) & mask;
                        (present != 0).then_some((id, block, present))
                    })
                    .collect();
                assert!(!blocks.is_empty());
                for survivors in [1usize, 2, 4, 8, 16, 32, 64, 128]
                    .into_iter()
                    .filter(|&s| s <= size as usize)
                {
                    // Noncontiguous slots include zero partial scores and spread
                    // lookups. Every selected integer score must agree exactly.
                    let slots: Vec<_> = (0..survivors).map(|i| (i * 67) % size as usize).collect();
                    let mut oracle = vec![[0u32; 256]; BLOCKS as usize];
                    for &(block_id, block, present) in &blocks {
                        let mut touched = [0; 4];
                        let faults = score_block_bsearch_int(
                            block,
                            &prepared.query_by_dim_u16,
                            present,
                            remaining_query.len() <= 8,
                            &mut oracle[block_id as usize],
                            &mut touched,
                            size as usize,
                        );
                        assert_eq!(faults.corrupt_terms + faults.dropped_postings, 0);
                        for &slot in &slots {
                            let (doc, ordinal) =
                                index.virtual_to_doc(block_id * size + slot as u32);
                            let id = forward
                                .find(crate::segment::logical_address::LogicalUnit { doc, ordinal })
                                .unwrap();
                            assert_eq!(
                                score_forward_units(forward.vector(id).unwrap(), &remaining_query)
                                    .unwrap(),
                                oracle[block_id as usize][slot]
                            );
                        }
                    }
                    let mut inverted_times = Vec::new();
                    let mut forward_times = Vec::new();
                    // Alternate order to reduce systematic warmup bias.
                    for sample in 0..9 {
                        for use_forward in if sample % 2 == 0 {
                            [false, true]
                        } else {
                            [true, false]
                        } {
                            let start = Instant::now();
                            for _ in 0..5 {
                                for &(block_id, parsed, present) in &blocks {
                                    if use_forward {
                                        for &slot in black_box(&slots) {
                                            let (doc, ordinal) =
                                                index.virtual_to_doc(block_id * size + slot as u32);
                                            let id = forward
                                                .find(
                                                    crate::segment::logical_address::LogicalUnit {
                                                        doc,
                                                        ordinal,
                                                    },
                                                )
                                                .unwrap();
                                            black_box(
                                                score_forward_units(
                                                    forward.vector(id).unwrap(),
                                                    black_box(&remaining_query),
                                                )
                                                .unwrap(),
                                            );
                                        }
                                    } else {
                                        // Phase one already parsed the block. Whole-query
                                        // evaluation still pays the parse cost.
                                        let block = if full_query {
                                            index.parse_block(block_id).unwrap()
                                        } else {
                                            parsed
                                        };
                                        let mut acc = [0; 256];
                                        let mut touched = [0; 4];
                                        black_box(score_block_bsearch_int(
                                            block,
                                            black_box(&prepared.query_by_dim_u16),
                                            present,
                                            remaining_query.len() <= 8,
                                            &mut acc,
                                            &mut touched,
                                            size as usize,
                                        ));
                                        black_box(acc);
                                    }
                                }
                            }
                            let ns =
                                start.elapsed().as_secs_f64() * 1e9 / (blocks.len() * 5) as f64;
                            if use_forward {
                                forward_times.push(ns);
                            } else {
                                inverted_times.push(ns);
                            }
                        }
                    }
                    inverted_times.sort_by(f64::total_cmp);
                    forward_times.sort_by(f64::total_cmp);
                    eprintln!(
                        "forward_experiment full_query={full_query} block_size={size} clustered={clustered} query={query_len} remaining={} active_blocks={} survivors={survivors}: inverted_ns={:.1} forward_ns={:.1} ratio={:.3}",
                        remaining_query.len(),
                        blocks.len(),
                        inverted_times[4],
                        forward_times[4],
                        forward_times[4] / inverted_times[4]
                    );
                }
            }
        }
    }
}
