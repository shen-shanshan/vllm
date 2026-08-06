# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ROCm-specific multi-stream helpers for DeepSeek-V4 CSA overlap.

Uses ``Stream.wait_stream()`` fork-join (validated in ATOM's
``maybe_compressors_async``) instead of CUDA Event cross-stream ordering,
which is unreliable on HIP and can deadlock or hang under multistream overlap.
"""

from collections.abc import Callable
from typing import Any

import torch


def rocm_execute_in_parallel(
    default_fn: Callable[[], Any],
    aux_fns: list[Callable[[], Any] | None],
    aux_streams: list[torch.cuda.Stream] | None,
    *,
    queue_aux_before_default: bool = True,
) -> tuple[Any, list[Any]]:
    """Run default_fn and aux_fns on separate HIP streams with wait_stream."""
    if aux_streams is None:
        default_result = default_fn()
        aux_results = [fn() if fn is not None else None for fn in aux_fns]
        return default_result, aux_results

    assert len(aux_fns) == len(aux_streams)
    current_stream = torch.cuda.current_stream()
    aux_results: list[Any] = [None] * len(aux_fns)
    pending: list[torch.cuda.Stream] = []

    def _launch_aux() -> None:
        for i, fn in enumerate(aux_fns):
            if fn is None:
                continue
            aux_stream = aux_streams[i]
            aux_stream.wait_stream(current_stream)
            with torch.cuda.stream(aux_stream):
                aux_results[i] = fn()
            pending.append(aux_stream)

    if queue_aux_before_default:
        _launch_aux()
        default_result = default_fn()
    else:
        default_result = default_fn()
        _launch_aux()

    for aux_stream in pending:
        current_stream.wait_stream(aux_stream)

    return default_result, aux_results


def rocm_maybe_execute_in_parallel(
    fn0: Callable[[], Any],
    fn1: Callable[[], Any],
    aux_stream: torch.cuda.Stream | None,
) -> tuple[Any, Any]:
    """Two-way overlap: fn1 on aux stream in parallel with fn0 on default."""
    if aux_stream is None:
        return fn0(), fn1()

    current_stream = torch.cuda.current_stream()
    aux_stream.wait_stream(current_stream)
    with torch.cuda.stream(aux_stream):
        result1 = fn1()
    result0 = fn0()
    current_stream.wait_stream(aux_stream)
    return result0, result1
