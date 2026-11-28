# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Longest-processing-time-first expert placement over decode and prefill load.

Per layer, with `V[k, e]` the normalised load of logical expert `e` under
objective `k` (decode-only steps and prefill-carrying steps, or their sum when
the step kinds are unknown):

1. Replicas: every expert gets one slot; each extra slot goes to the expert
   with the largest per-replica load `max_k V[k, e] / nrep[e]`, at most one
   replica per GPU (ties: lower expert id).
2. Placement: replicas in descending per-replica load, each to the GPU with a
   free slot and no replica of that expert that minimises
   `max_k (load[gpu, k] + V[k, e] / nrep[e])` (ties: smaller sum, lower GPU).
3. Slots: with no previous placement, a GPU's slots hold its experts in
   ascending id; otherwise experts that stay on a GPU keep their slot and only
   the experts new to it take the freed slots.

Balancing decode and prefill separately matters when they route to different
hot experts: their sum can look balanced while each kind of step is not.
"""

import numpy as np
import torch

from .abstract import AbstractEplbPolicy, EplbStats, EplbStepKindStats


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.atleast_2d(np.asarray(v, dtype=np.float64))
    tot = v.sum(1, keepdims=True)
    # an empty objective counts as uniform
    return np.where(tot > 0, v / np.where(tot > 0, tot, 1), 1.0 / v.shape[1])


def objectives(decode: np.ndarray, prefill: np.ndarray | None) -> np.ndarray:
    """One layer's normalised objectives `V [K, E]`; one objective (the load
    as given) without the prefill split."""
    if prefill is None:
        return _normalise(decode)
    return _normalise(np.stack([decode, prefill]))


def replicate(V: np.ndarray, n_slots: int, num_gpus: int) -> np.ndarray:
    """Replica count per expert for `n_slots` slots (step 1)."""
    n = V.shape[1]
    if not n <= n_slots <= num_gpus * n:
        raise ValueError(f"{n_slots} slots for {n} experts on {num_gpus} ranks")
    m = V.max(0)
    nrep = np.ones(n, dtype=np.int64)
    for _ in range(n_slots - n):
        nrep[int(np.argmax(np.where(nrep < num_gpus, m / nrep, -1.0)))] += 1
    return nrep


def place(V: np.ndarray, nrep: np.ndarray, slots: int, num_gpus: int) -> list[set[int]]:
    """The experts each GPU holds (step 2)."""
    k, n = V.shape
    m = V.max(0)
    order = sorted((-m[x] / nrep[x], x) for x in range(n) for _ in range(nrep[x]))
    load = np.zeros((num_gpus, k))
    held: list[set[int]] = [set() for _ in range(num_gpus)]
    for _, x in order:
        w = V[:, x] / nrep[x]
        free = [r for r in range(num_gpus) if len(held[r]) < slots and x not in held[r]]
        r = min(free, key=lambda i: ((load[i] + w).max(), (load[i] + w).sum(), i))
        held[r].add(x)
        load[r] += w
    return held


def assign_slots(
    held: list[set[int]], slots: int, old_row: np.ndarray | None
) -> np.ndarray:
    """`phys2log [num_gpus * slots]` from each GPU's experts (step 3)."""
    row = np.empty(len(held) * slots, dtype=np.int64)
    for r, experts in enumerate(held):
        if old_row is None:
            row[r * slots : (r + 1) * slots] = sorted(experts)
            continue
        old = old_row[r * slots : (r + 1) * slots]
        keep = [e in experts for e in old]
        incoming = iter(sorted(experts - set(old[keep].tolist())))
        row[r * slots : (r + 1) * slots] = [
            e if kept else next(incoming) for e, kept in zip(old.tolist(), keep)
        ]
    return row


def plan_layer(
    decode: np.ndarray,
    prefill: np.ndarray | None,
    num_replicas: int,
    num_gpus: int,
    old_row: np.ndarray | None = None,
) -> np.ndarray:
    V = objectives(decode, prefill)
    slots = num_replicas // num_gpus
    held = place(V, replicate(V, num_replicas, num_gpus), slots, num_gpus)
    return assign_slots(held, slots, old_row)


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
        num_gpus = stats.num_gpus
        if stats.num_replicas % num_gpus:
            raise ValueError(
                f"{stats.num_replicas} physical experts do not split over "
                f"{num_gpus} GPUs"
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
            else old_global_expert_indices.numpy()
        )
        rows = [
            plan_layer(
                decode[layer],
                None if prefill is None else prefill[layer],
                stats.num_replicas,
                num_gpus,
                None if old is None else old[layer],
            )
            for layer in range(decode.shape[0])
        ]
        return torch.from_numpy(np.stack(rows))
