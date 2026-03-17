# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data transfer objects for encoder CUDA graph management."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class EncoderCudaGraphConfig:
    """Configuration for encoder CUDA graph management.

    Provided by the model at init time via
    ``get_encoder_cudagraph_config()``. Values are fixed for the
    lifetime of the manager.
    """

    modalities: list[str]
    """Supported modalities (e.g. ["image"] or ["image", "video"])."""

    input_key: str
    """Key in mm_kwargs for the input tensor (e.g. "pixel_values").
    Used as fallback when modality_input_keys is not set."""

    buffer_keys: list[str]
    """Keys for the tensor buffers recorded into the CUDA graph.
    Before replay the manager zeros then slice-copies new data
    into these buffers."""

    out_hidden_size: int
    """Output hidden dim of the vision encoder.
    Used for DP gather buffer allocation."""

    modality_input_keys: dict[str, str] | None = None
    """Per-modality input key mapping (e.g. {"image": "pixel_values",
    "video": "pixel_values_videos"}).  When set, the manager uses
    this mapping instead of ``input_key`` to identify the main input
    tensor for each modality."""


@dataclass
class EncoderCudaGraphCaptureInputs:
    """Everything needed for one CUDA graph capture.

    Returned by ``prepare_encoder_cudagraph_capture_inputs()``.
    """

    mm_kwargs: dict[str, Any]
    """Dummy forward inputs (model-specific keys).
    For Qwen3-VL images this contains pixel_values and image_grid_thw;
    for videos it contains pixel_values_videos and video_grid_thw."""

    buffers: dict[str, torch.Tensor]
    """Precomputed tensor buffers that will be recorded into the
    CUDA graph.  The manager stores references to these exact
    tensor objects and copies new data into them before each
    ``graph.replay()`` call (buffer identity invariant)."""

    max_num_seqs: int = 0
    """Total number of attention sequences (cu_seqlens segments) used
    during this capture.  For images this equals the number of dummy
    images (t=1 always).  For videos it is
    ``max_batch_size * t_capture`` where ``t_capture`` is the fixed
    temporal grid dimension used for capture.  Stored in
    ``BudgetGraphMetadata`` and checked against actual replay batches
    to prevent cu_seqlens buffer overflow."""


@dataclass
class EncoderCudaGraphReplayBuffers:
    """New buffer values for graph replay, computed by the model from
    actual batch inputs.

    Returned by ``prepare_encoder_cudagraph_replay_buffers()``.
    Keys match ``EncoderCudaGraphConfig.buffer_keys``.
    """

    buffers: dict[str, torch.Tensor | None]
    """Data to copy into the captured buffers before replay.
    ``None`` values leave the corresponding captured buffer
    unchanged."""

    fits: bool = True
    """Whether the current batch fits within the captured graph
    constraints (e.g. temporal frame count does not exceed the
    budget captured at capture time).  When ``False``, the manager
    will skip graph replay and fall back to eager execution."""
