# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Longest-processing-time-first expert placement over decode and prefill load.

Per layer, with `V[k, e]` the normalised load of logical expert `e` under
objective `k` (decode-only steps and prefill-carrying steps, or their sum when
the step kinds are unknown):

1. Replicas: every expert gets one slot; each extra slot goes to the expert
   with the largest per-replica load `max_k V[k, e] / nrep[e]`, at most one
   replica per GPU (ties: lower expert id).
2. Placement: replicas in descending per-replica load (ties: lower expert id),
   each to the GPU with a free slot and no replica of that expert that
   minimises `max_k (load[gpu, k] + V[k, e] / nrep[e])` (ties: smaller sum,
   lower GPU).
3. Slots: with no previous placement, a GPU's slots hold its experts in
   ascending id; otherwise experts that stay on a GPU keep their slot and only
   the experts new to it take the freed slots, in ascending id.

Balancing decode and prefill separately matters when they route to different
hot experts: their sum can look balanced while each kind of step is not.

The greedy steps run for all layers at once (layers are independent), so a
rebalance costs one pass over the replicas rather than one per layer.
"""

import numpy as np
import torch

from .abstract import AbstractEplbPolicy, EplbStats, EplbStepKindStats


def _normalise(v: np.ndarray) -> np.ndarray:
    """Normalise the last axis to sum 1; an empty objective counts as uniform."""
    v = np.asarray(v, dtype=np.float64)
    tot = v.sum(-1, keepdims=True)
    return np.where(tot > 0, v / np.where(tot > 0, tot, 1), 1.0 / v.shape[-1])


def objectives(decode: np.ndarray, prefill: np.ndarray | None) -> np.ndarray:
    """Normalised objectives `V [..., K, E]` from loads `[..., E]`; one
    objective (the load as given) without the prefill split."""
    if prefill is None:
        return _normalise(decode)[..., None, :]
    return _normalise(np.stack([decode, prefill], axis=-2))


def replicate_layers(V: np.ndarray, n_slots: int, num_gpus: int) -> np.ndarray:
    """Replica count per expert `[L, E]` for `n_slots` slots (step 1)."""
    num_layers, _, n = V.shape
    if not n <= n_slots <= num_gpus * n:
        raise ValueError(f"{n_slots} slots for {n} experts on {num_gpus} ranks")
    m = V.max(1)
    nrep = np.ones((num_layers, n), dtype=np.int64)
    rows = np.arange(num_layers)
    for _ in range(n_slots - n):
        score = np.where(nrep < num_gpus, m / nrep, -1.0)
        nrep[rows, score.argmax(1)] += 1
    return nrep


def place_layers(
    V: np.ndarray, nrep: np.ndarray, slots: int, num_gpus: int
) -> np.ndarray:
    """Whether GPU `g` holds expert `e`, `[L, num_gpus, E]` (step 2)."""
    num_layers, k, n = V.shape
    rows = np.arange(num_layers)
    per_replica = V.max(1) / nrep
    # every layer has the same number of replicas, so they stack
    experts = np.stack([np.repeat(np.arange(n), r) for r in nrep])
    keys = np.take_along_axis(per_replica, experts, 1)
    order = np.lexsort((experts, -keys), axis=-1)
    experts = np.take_along_axis(experts, order, 1)

    load = np.zeros((num_layers, num_gpus, k))
    held = np.zeros((num_layers, num_gpus, n), dtype=bool)
    used = np.zeros((num_layers, num_gpus), dtype=np.int64)
    for x in experts.T:
        w = V[rows, :, x] / nrep[rows, x][:, None]
        cand = load + w[:, None, :]
        ok = (used < slots) & ~held[rows, :, x]
        worst = np.where(ok, cand.max(2), np.inf)
        ok &= worst == worst.min(1, keepdims=True)
        total = np.where(ok, cand.sum(2), np.inf)
        ok &= total == total.min(1, keepdims=True)
        gpu = ok.argmax(1)
        held[rows, gpu, x] = True
        used[rows, gpu] += 1
        load[rows, gpu] += w
    return held


def assign_slots_layers(
    held: np.ndarray, slots: int, old: np.ndarray | None
) -> np.ndarray:
    """`phys2log [L, num_gpus * slots]` from `held [L, num_gpus, E]` (step 3)."""
    num_layers, num_gpus, n = held.shape
    # each GPU holds exactly `slots` experts; nonzero lists them per GPU in
    # ascending id
    fresh = np.nonzero(held)[2].reshape(num_layers, num_gpus * slots)
    if old is None:
        return fresh
    old = old.reshape(num_layers, num_gpus, slots)
    keep = np.take_along_axis(held, old, 2)
    was_there = np.zeros_like(held)
    np.put_along_axis(was_there, old, True, 2)
    incoming = np.nonzero(held & ~was_there)[2]
    row = old.copy()
    # both sides list GPUs in order, and per GPU as many freed slots as
    # incoming experts
    row[~keep] = incoming
    return row.reshape(num_layers, num_gpus * slots)


def plan_layers(
    decode: np.ndarray,
    prefill: np.ndarray | None,
    num_replicas: int,
    num_gpus: int,
    old: np.ndarray | None = None,
) -> np.ndarray:
    """`phys2log [L, num_replicas]` from per-layer loads `[L, E]`."""
    V = objectives(decode, prefill)
    slots = num_replicas // num_gpus
    held = place_layers(V, replicate_layers(V, num_replicas, num_gpus), slots, num_gpus)
    return assign_slots_layers(held, slots, old)


# Single-layer forms (the DeepSeek-V4.1 amd_kernels plan file builder).
def replicate(V: np.ndarray, n_slots: int, num_gpus: int) -> np.ndarray:
    """Replica count per expert of one layer's `V [K, E]` (step 1)."""
    return replicate_layers(V[None], n_slots, num_gpus)[0]


def place(V: np.ndarray, nrep: np.ndarray, slots: int, num_gpus: int) -> np.ndarray:
    """One layer's `phys2log [num_gpus * slots]` (steps 2 and 3, no previous
    placement)."""
    held = place_layers(V[None], nrep[None], slots, num_gpus)
    return assign_slots_layers(held, slots, None)[0]


class LptEplbPolicy(AbstractEplbPolicy):
    """LPT placement balancing decode and prefill load (module docstring).

    Uses the per-step-kind load windows (`EplbStepKindStats`) when EPLB
    provides them, the summed window otherwise. Expert groups and nodes are
    not considered: the placement is flat over all GPUs.
    """

    uses_step_kind = True

    def rebalance_experts(
        self,
        stats: EplbStats,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if stats.num_replicas % stats.num_gpus:
            raise ValueError(
                f"{stats.num_replicas} physical experts do not split over "
                f"{stats.num_gpus} GPUs"
            )
        if isinstance(stats, EplbStepKindStats):
            decode = stats.decode_load_window.double().numpy()
            prefill = stats.prefill_load_window.double().numpy()
        else:
            decode = stats.global_expert_load_window.double().numpy()
            prefill = None
        old = (
            None
            if old_global_expert_indices is None
            else old_global_expert_indices.numpy().astype(np.int64)
        )
        return torch.from_numpy(
            plan_layers(decode, prefill, stats.num_replicas, stats.num_gpus, old)
        )
