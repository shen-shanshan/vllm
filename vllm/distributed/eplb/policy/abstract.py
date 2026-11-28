# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class EplbStats:
    """Load and topology snapshot for one EPLB planning call.

    Policies may accept subclasses with additional planning inputs."""

    global_expert_load_window: torch.Tensor
    """
    Expert load window, converted to CPU before policy invocation.
    Built-in policies use shape (num_moe_layers, num_logical_experts).
    """
    num_replicas: int
    """
    Number of physical experts.
    """
    num_groups: int
    """
    Number of expert groups.
    """
    num_nodes: int
    """
    Number of nodes.
    """
    num_gpus: int
    """
    Number of GPUs.
    """


@dataclass(slots=True)
class EplbStepKindStats(EplbStats):
    """`EplbStats` with the load window split by step kind, for policies that
    set `uses_step_kind`. Each rank counts its own steps: a step is a prefill
    step on a rank when that rank's batch carried prefill tokens."""

    decode_load_window: torch.Tensor
    """Load of the decode-only steps, same shape as the summed window."""
    prefill_load_window: torch.Tensor
    """Load of the prefill-carrying steps, same shape as the summed window."""


class AbstractEplbPolicy(ABC):
    uses_step_kind: bool = False
    """Whether the policy wants `EplbStepKindStats`: EPLB then records the
    step kind of every load-window entry."""

    @abstractmethod
    def rebalance_experts(
        self,
        stats: EplbStats,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Entry point for expert-parallelism load balancer.

        Args:
            stats: Load/topology snapshot with a CPU load tensor.
                Built-in policies use `[layers, num_logical_experts]` loads.
                Policy-specific subclasses may provide additional inputs.
            old_global_expert_indices: [layers, num_replicas], the old global
                expert indices. Used to avoid unnecessary weight copying
                for experts moving within one rank.

        Returns:
            physical_to_logical_map: [layers, num_replicas], the expert
                index of each replica, on CPU.

        """
        raise NotImplementedError
