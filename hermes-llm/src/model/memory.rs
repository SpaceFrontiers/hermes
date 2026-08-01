//! Fast-to-slow continuum memory for sleep-capable models.
//!
//! A memory chain is opt-in through MAL and replaces a block's ordinary FFN
//! branch. Each tier remains a residual sublayer, so a zero-output slower tier
//! is an exact no-op. Low-rank reserve experts are allocated up front and are
//! routed only after their checkpointed activation bit is set.

use anyhow::Result;
use burn::module::{ModuleVisitor, Param, ParamId};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::{Bool, DType, Int, TensorData};
use burn_nn::{Linear, LinearConfig};

use crate::mal::{BlockDef, MemoryDef, MemoryTierInit, ModelDef};

use super::FeedForward;
use super::ffn::host_route_plan;
#[cfg(feature = "cuda")]
use super::grouped_linear::{grouped_linear, is_cuda_device};
use super::matmul::matmul_2;
#[cfg(feature = "cuda")]
use super::matmul::matmul_input;
#[cfg(feature = "cuda")]
use super::moe_dispatch::{route_combine, route_gather};
#[cfg(feature = "cuda")]
use super::moe_route::route_plan;
use super::row_permute::row_permute;

/// Expert-routing behavior for ordinary wake execution and dream generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryRouting {
    #[default]
    Wake,
    /// Add one deterministic random expert to every MoE router. This must only
    /// be selected while generating synthetic dreams.
    Dream { seed: u64 },
}

impl MemoryRouting {
    pub(crate) fn salted(self, salt: u64) -> Self {
        match self {
            Self::Wake => Self::Wake,
            Self::Dream { seed } => Self::Dream {
                seed: mix_seed(seed, salt),
            },
        }
    }

    pub(crate) fn random_index(self, count: usize) -> Option<usize> {
        match self {
            Self::Wake => None,
            Self::Dream { seed } if count > 0 => {
                Some(mix_seed(seed, count as u64) as usize % count)
            }
            Self::Dream { .. } => None,
        }
    }
}

fn mix_seed(mut value: u64, salt: u64) -> u64 {
    value ^= salt.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Checkpointed state and trainable parameter IDs for one reserve slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemorySlotStatus {
    pub layer: usize,
    pub tier: usize,
    pub tier_name: String,
    pub slot: usize,
    pub active: bool,
    pub generation: u64,
    pub parameter_ids: Vec<ParamId>,
}

#[derive(Module, Debug)]
struct LowRankExpertSlot {
    router: Linear,
    a: Param<Tensor<2>>,
    b: Param<Tensor<2>>,
    // Bool/int parameters are serialized by Burn but never visited by a float
    // optimizer. The runtime mirror avoids a host synchronization per token.
    active: Param<Tensor<1, Bool>>,
    generation: Param<Tensor<1, Int>>,
    #[module(skip)]
    runtime_active: bool,
}

impl LowRankExpertSlot {
    fn new(hidden: usize, rank: usize, slot: usize, device: &Device) -> Self {
        let a_values = deterministic_values(hidden * rank, mix_seed(slot as u64, 0x41));
        Self {
            router: LinearConfig::new(hidden, 1).with_bias(false).init(device),
            a: Param::from_tensor(Tensor::from_data(
                TensorData::new(a_values, [hidden, rank]),
                device,
            )),
            // A zero B is both a strict residual no-op and the standard LoRA
            // initialization: B receives gradients on the first update.
            b: Param::from_tensor(Tensor::zeros([rank, hidden], device)),
            active: Param::initialized(
                ParamId::new(),
                Tensor::<1, Bool>::from_data([false], device),
            ),
            generation: Param::initialized(
                ParamId::new(),
                Tensor::<1, Int>::from_data([0_i64], device),
            ),
            runtime_active: false,
        }
    }

    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        let dtype = input.dtype();
        matmul_2(matmul_2(input.cast(DType::F32), self.a.val()), self.b.val()).cast(dtype)
    }

    fn generation(&self) -> u64 {
        self.generation
            .val()
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .expect("memory generation must be readable")[0] as u64
    }

    fn checkpoint_active(&self) -> bool {
        self.active
            .val()
            .into_data()
            .to_vec::<bool>()
            .expect("memory activation mask must be readable")[0]
    }

    fn set_active(&mut self, active: bool) {
        self.active = self.active.clone().map(|value| {
            let device = value.device();
            Tensor::<1, Bool>::from_data([active], &device)
        });
        self.runtime_active = active;
    }

    fn sync_state(&mut self) {
        self.runtime_active = self.checkpoint_active();
    }

    fn reset(&mut self, seed: u64) {
        let [hidden, rank] = self.a.shape().dims();
        let a_values = deterministic_values(hidden * rank, seed);
        self.a = self.a.clone().map(|value| {
            let device = value.device();
            Tensor::from_data(TensorData::new(a_values, [hidden, rank]), &device)
        });
        self.b = self.b.clone().map(|value| Tensor::zeros_like(&value));
        self.router.weight = self
            .router
            .weight
            .clone()
            .map(|value| Tensor::zeros_like(&value));
        let generation = self.generation().saturating_add(1) as i64;
        self.generation = self.generation.clone().map(|value| {
            let device = value.device();
            Tensor::<1, Int>::from_data([generation], &device)
        });
        self.set_active(false);
    }

    fn parameter_ids(&self) -> Vec<ParamId> {
        vec![self.router.weight.id, self.a.id, self.b.id]
    }
}

fn deterministic_values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.max(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 40) as f32 / (1_u32 << 24) as f32;
            (unit * 2.0 - 1.0) * 0.02
        })
        .collect()
}

#[derive(Module, Debug)]
struct LowRankExpertReserve {
    slots: Vec<LowRankExpertSlot>,
    #[module(skip)]
    top_k: usize,
}

impl LowRankExpertReserve {
    fn new(hidden: usize, capacity: usize, rank: usize, top_k: usize, device: &Device) -> Self {
        Self {
            slots: (0..capacity)
                .map(|slot| LowRankExpertSlot::new(hidden, rank, slot, device))
                .collect(),
            top_k,
        }
    }

    fn forward<const D: usize>(
        &self,
        input: Tensor<D>,
        routing: MemoryRouting,
    ) -> Option<Tensor<D>> {
        let active = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.runtime_active)
            .collect::<Vec<_>>();
        if active.is_empty() {
            // Do not put a zero allocation or residual add on the accelerator
            // graph. Besides avoiding wake-time kernels, this is what makes a
            // dormant reserve completely absent from forward and backward.
            return None;
        }

        let shape = input.dims();
        let hidden = shape[D - 1];
        let tokens = shape[..D - 1].iter().product::<usize>();
        let flat = input.reshape([tokens, hidden]);
        if active.len() == 1 {
            // A one-candidate top-1 router is identically one. Avoid a device
            // synchronization and all dispatch/permutation kernels in the
            // common first-active-slot case.
            return Some(active[0].1.forward(flat).reshape(shape));
        }

        // Concatenate only active router rows, then evaluate them with one
        // projection. Dormant rows stay off the graph and acquire no gradient
        // or optimizer state.
        let router_weight = Tensor::cat(
            active
                .iter()
                .map(|(_, slot)| slot.router.weight.val())
                .collect(),
            1,
        );
        let logits = matmul_2(flat.clone().cast(DType::F32), router_weight);
        let routed_k = self.top_k.min(active.len());
        let (top_logits, top_indices) = logits.clone().topk_with_indices(routed_k, 1);
        let extra = routing.random_index(active.len());
        let all_weights = match extra {
            Some(index) => softmax(
                Tensor::cat(
                    vec![
                        top_logits,
                        logits.clone().slice([0..tokens, index..index + 1]),
                    ],
                    1,
                ),
                1,
            ),
            None => softmax(top_logits, 1),
        };
        let top_weights = all_weights.clone().slice([0..tokens, 0..routed_k]);
        let routes = tokens * routed_k;
        let device = flat.device();
        #[cfg(feature = "cuda")]
        let (route_order, inverse_order, mut counts, device_counts) = if is_cuda_device(&device) {
            let (order, inverse, counts) = route_plan(top_indices.clone(), active.len());
            (order, inverse, Vec::new(), Some(counts))
        } else {
            let (order, inverse, counts) =
                host_route_plan(top_indices.clone(), active.len(), routes, &device);
            (order, inverse, counts, None)
        };
        #[cfg(not(feature = "cuda"))]
        let (route_order, inverse_order, counts) =
            host_route_plan(top_indices, active.len(), routes, &device);

        #[cfg(feature = "cuda")]
        let routed_input = if is_cuda_device(&device) {
            route_gather(
                flat.clone(),
                route_order.clone(),
                inverse_order.clone(),
                routed_k,
            )
        } else {
            let repeated = flat
                .clone()
                .unsqueeze_dim::<3>(1)
                .repeat_dim(1, routed_k)
                .reshape([routes, hidden]);
            row_permute(repeated, route_order.clone(), inverse_order.clone())
        };
        #[cfg(not(feature = "cuda"))]
        let routed_input = {
            let repeated = flat
                .clone()
                .unsqueeze_dim::<3>(1)
                .repeat_dim(1, routed_k)
                .reshape([routes, hidden]);
            row_permute(repeated, route_order.clone(), inverse_order.clone())
        };

        #[cfg(feature = "cuda")]
        if let Some(device_counts) = device_counts {
            // Keep the route plan on-device. Only the small per-slot count
            // vector crosses to the host to define grouped-GEMM descriptors.
            counts = device_counts
                .into_data()
                .convert::<i64>()
                .to_vec::<i64>()
                .expect("memory route counts must be readable")
                .into_iter()
                .map(|count| count as usize)
                .collect();
        }

        #[cfg(feature = "cuda")]
        let routed_output = if is_cuda_device(&device) {
            let a = Tensor::stack::<3>(active.iter().map(|(_, slot)| slot.a.val()).collect(), 0);
            let b = Tensor::stack::<3>(active.iter().map(|(_, slot)| slot.b.val()).collect(), 0);
            let projected = grouped_linear(routed_input, matmul_input(a), &counts);
            grouped_linear(projected, matmul_input(b), &counts).cast(flat.dtype())
        } else {
            Self::forward_compact(&active, routed_input, &counts, hidden)
        };
        #[cfg(not(feature = "cuda"))]
        let routed_output = Self::forward_compact(&active, routed_input, &counts, hidden);

        #[cfg(feature = "cuda")]
        let mut output = if is_cuda_device(&device) {
            route_combine(routed_output, top_weights, inverse_order, routed_k)
        } else {
            (row_permute(routed_output, inverse_order, route_order)
                .reshape([tokens, routed_k, hidden])
                * top_weights
                    .cast(flat.dtype())
                    .reshape([tokens, routed_k, 1]))
            .sum_dim(1)
            .reshape([tokens, hidden])
        };
        #[cfg(not(feature = "cuda"))]
        let mut output = (row_permute(routed_output, inverse_order, route_order)
            .reshape([tokens, routed_k, hidden])
            * top_weights
                .cast(flat.dtype())
                .reshape([tokens, routed_k, 1]))
        .sum_dim(1)
        .reshape([tokens, hidden]);
        if let Some(index) = extra {
            let weight = all_weights
                .slice([0..tokens, routed_k..routed_k + 1])
                .cast(flat.dtype());
            output = output + active[index].1.forward(flat.clone()) * weight;
        }
        Some(output.reshape(shape))
    }

    fn forward_compact(
        active: &[(usize, &LowRankExpertSlot)],
        routed_input: Tensor<2>,
        counts: &[usize],
        hidden: usize,
    ) -> Tensor<2> {
        let mut offset = 0;
        let mut outputs = Vec::new();
        for (local, count) in counts.iter().copied().enumerate() {
            if count == 0 {
                continue;
            }
            outputs.push(
                active[local].1.forward(
                    routed_input
                        .clone()
                        .slice([offset..offset + count, 0..hidden]),
                ),
            );
            offset += count;
        }
        debug_assert_eq!(offset, routed_input.dims()[0]);
        Tensor::cat(outputs, 0)
    }

    fn sync_state(&mut self) {
        for slot in &mut self.slots {
            slot.sync_state();
        }
    }
}

#[derive(Module, Debug)]
struct MemoryTier {
    feed_forward: FeedForward,
    reserve: LowRankExpertReserve,
    #[module(skip)]
    name: String,
}

impl MemoryTier {
    fn forward<const D: usize>(
        &self,
        input: Tensor<D>,
        collect_auxiliary: bool,
        routing: MemoryRouting,
    ) -> (Tensor<D>, Option<Tensor<1>>) {
        let reserve = self.reserve.forward(input.clone(), routing);
        let (base, auxiliary) = self.feed_forward.forward_internal_with_routing(
            input.clone(),
            collect_auxiliary,
            routing,
        );
        (
            match reserve {
                Some(reserve) => base + reserve,
                None => base,
            },
            auxiliary,
        )
    }
}

/// Ordered residual chain of fast-to-slow FFN/MoE tiers.
#[derive(Module, Debug)]
pub struct MemoryChain {
    tiers: Vec<MemoryTier>,
}

impl MemoryChain {
    pub(crate) fn new(
        config: &ModelDef,
        block: &BlockDef,
        definition: &MemoryDef,
        device: &Device,
    ) -> Self {
        let tiers = definition
            .tiers
            .iter()
            .map(|definition| {
                let mut tier_block = block.clone();
                tier_block.memory = None;
                tier_block.ffn = definition.ffn.clone();
                let mut feed_forward = FeedForward::new(config, &tier_block, device);
                if matches!(definition.residual_init, MemoryTierInit::ResidualZero) {
                    feed_forward.zero_output();
                }
                MemoryTier {
                    feed_forward,
                    reserve: LowRankExpertReserve::new(
                        config.hidden_size,
                        definition.reserve_experts.capacity,
                        definition.reserve_experts.rank,
                        definition.reserve_experts.top_k,
                        device,
                    ),
                    name: definition.name.clone(),
                }
            })
            .collect();
        Self { tiers }
    }

    pub(crate) fn forward_with<const D: usize>(
        &self,
        mut state: Tensor<D>,
        collect_auxiliary: bool,
        routing: MemoryRouting,
        mut normalize: impl FnMut(Tensor<D>) -> Tensor<D>,
        mut residual: impl FnMut(Tensor<D>, Tensor<D>) -> Tensor<D>,
    ) -> (Tensor<D>, Option<Tensor<1>>) {
        let mut auxiliary = None;
        for (index, tier) in self.tiers.iter().enumerate() {
            let (branch, tier_auxiliary) = tier.forward(
                normalize(state.clone()),
                collect_auxiliary,
                routing.salted(index as u64),
            );
            state = residual(state, branch);
            if let Some(loss) = tier_auxiliary {
                auxiliary = Some(match auxiliary {
                    Some(total) => total + loss,
                    None => loss,
                });
            }
        }
        (state, auxiliary)
    }

    pub(crate) fn prepare_inference(&mut self) {
        self.sync_state();
        for tier in &mut self.tiers {
            tier.feed_forward.prepare_inference();
        }
    }

    pub(crate) fn sync_state(&mut self) {
        for tier in &mut self.tiers {
            tier.reserve.sync_state();
        }
    }

    pub(crate) fn prepare_upgrade_state(&mut self) {
        for (index, tier) in self.tiers.iter_mut().enumerate() {
            if index > 0 {
                tier.feed_forward.zero_output();
            }
            for slot in &mut tier.reserve.slots {
                slot.set_active(false);
            }
        }
    }

    pub(crate) fn activate_slot(&mut self, tier: usize, slot: usize) -> Result<()> {
        let tier_index = tier;
        let slot_index = slot;
        let tier = self
            .tiers
            .get_mut(tier_index)
            .ok_or_else(|| anyhow::anyhow!("memory tier {tier_index} does not exist"))?;
        let slot = tier
            .reserve
            .slots
            .get_mut(slot_index)
            .ok_or_else(|| anyhow::anyhow!("memory slot {slot_index} does not exist"))?;
        if slot.runtime_active {
            anyhow::bail!("memory slot {slot_index} in tier {tier_index} is already active");
        }
        slot.set_active(true);
        Ok(())
    }

    pub(crate) fn deactivate_slot(&mut self, tier: usize, slot: usize) -> Result<()> {
        let tier = self
            .tiers
            .get_mut(tier)
            .ok_or_else(|| anyhow::anyhow!("memory tier {tier} does not exist"))?;
        let slot = tier
            .reserve
            .slots
            .get_mut(slot)
            .ok_or_else(|| anyhow::anyhow!("memory slot {slot} does not exist"))?;
        slot.set_active(false);
        Ok(())
    }

    pub(crate) fn reset_slot(&mut self, tier: usize, slot: usize, seed: u64) -> Result<()> {
        let tier = self
            .tiers
            .get_mut(tier)
            .ok_or_else(|| anyhow::anyhow!("memory tier {tier} does not exist"))?;
        let slot = tier
            .reserve
            .slots
            .get_mut(slot)
            .ok_or_else(|| anyhow::anyhow!("memory slot {slot} does not exist"))?;
        slot.reset(seed);
        Ok(())
    }

    pub(crate) fn statuses(&self, layer: usize) -> Vec<MemorySlotStatus> {
        self.tiers
            .iter()
            .enumerate()
            .flat_map(|(tier_index, tier)| {
                tier.reserve
                    .slots
                    .iter()
                    .enumerate()
                    .map(move |(slot_index, slot)| MemorySlotStatus {
                        layer,
                        tier: tier_index,
                        tier_name: tier.name.clone(),
                        slot: slot_index,
                        active: slot.runtime_active,
                        generation: slot.generation(),
                        parameter_ids: slot.parameter_ids(),
                    })
            })
            .collect()
    }

    pub(crate) fn tier_parameter_ids(&self, tier: usize) -> Result<Vec<ParamId>> {
        let tier = self
            .tiers
            .get(tier)
            .ok_or_else(|| anyhow::anyhow!("memory tier {tier} does not exist"))?;
        #[derive(Default)]
        struct FloatIds(Vec<ParamId>);
        impl ModuleVisitor for FloatIds {
            fn visit_float<const D: usize>(&mut self, parameter: &Param<Tensor<D>>) {
                self.0.push(parameter.id);
            }
        }
        let mut ids = FloatIds::default();
        tier.visit(&mut ids);
        Ok(ids.0)
    }

    pub(crate) fn non_muon_parameter_ids(&self) -> Vec<ParamId> {
        let mut ids = Vec::new();
        for tier in &self.tiers {
            ids.extend(tier.feed_forward.router_parameter_id());
            // Reserve factors stay out of the always-on wake Muon group. The
            // sleep optimizer selects an active slot by its explicit IDs.
            ids.extend(
                tier.reserve
                    .slots
                    .iter()
                    .flat_map(LowRankExpertSlot::parameter_ids),
            );
        }
        ids
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;

    use super::*;

    fn memory_config() -> ModelDef {
        crate::mal::parse_mal(
            r#"
            ffn base { hidden_dim: 12 activation: swiglu }
            memory cms {
                tier fast {
                    ffn: base
                    reserve_experts { capacity: 2 rank: 3 top_k: 1 }
                }
            }
            model sleeper {
                vocab_size: 16 max_seq_len: 8 hidden_size: 8 num_layers: 1
                block: { attention: { num_heads: 1 } memory: cms }
            }
            "#,
        )
        .unwrap()
    }

    #[test]
    fn dormant_slots_have_no_forward_or_backward_path() {
        let config = memory_config();
        let device = Device::ndarray().autodiff();
        let block = config.block.clone();
        let mut memory = MemoryChain::new(&config, &block, block.memory.as_ref().unwrap(), &device);
        let run = |memory: &MemoryChain| {
            let input = Tensor::<2>::random([5, 8], Distribution::Default, &device);
            memory
                .forward_with(
                    input,
                    false,
                    MemoryRouting::Wake,
                    |state| state,
                    |state, branch| state + branch,
                )
                .0
                .square()
                .mean()
                .backward()
        };

        let dormant_gradients = run(&memory);
        for slot in &memory.tiers[0].reserve.slots {
            assert!(slot.a.grad(&dormant_gradients).is_none());
            assert!(slot.b.grad(&dormant_gradients).is_none());
            assert!(slot.router.weight.grad(&dormant_gradients).is_none());
        }

        memory.activate_slot(0, 0).unwrap();
        let active_gradients = run(&memory);
        let active = &memory.tiers[0].reserve.slots[0];
        let dormant = &memory.tiers[0].reserve.slots[1];
        assert!(active.a.grad(&active_gradients).is_some());
        assert!(active.b.grad(&active_gradients).is_some());
        // With exactly one candidate the top-1 decision is constant, so the
        // router row is intentionally absent until there is a real choice.
        assert!(active.router.weight.grad(&active_gradients).is_none());
        assert!(dormant.a.grad(&active_gradients).is_none());
        assert!(dormant.b.grad(&active_gradients).is_none());
        assert!(dormant.router.weight.grad(&active_gradients).is_none());

        let error = memory.activate_slot(0, 0).unwrap_err().to_string();
        assert!(error.contains("already active"), "{error}");
        memory.activate_slot(0, 1).unwrap();
        let routed_gradients = run(&memory);
        assert!(
            memory.tiers[0].reserve.slots[0]
                .router
                .weight
                .grad(&routed_gradients)
                .is_some()
        );
        assert!(
            memory.tiers[0].reserve.slots[1]
                .router
                .weight
                .grad(&routed_gradients)
                .is_some()
        );
    }

    #[test]
    fn reset_is_dormant_noop_and_advances_generation() {
        let config = memory_config();
        let device = Device::ndarray();
        let block = config.block.clone();
        let mut memory = MemoryChain::new(&config, &block, block.memory.as_ref().unwrap(), &device);
        memory.activate_slot(0, 0).unwrap();
        assert!(memory.statuses(0)[0].active);
        memory.reset_slot(0, 0, 17).unwrap();
        let status = &memory.statuses(0)[0];
        assert!(!status.active);
        assert_eq!(status.generation, 1);
        let b: f32 = memory.tiers[0].reserve.slots[0]
            .b
            .val()
            .abs()
            .sum()
            .into_scalar();
        assert_eq!(b, 0.0);
    }
}
