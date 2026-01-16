# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from pydantic.dataclasses import dataclass


@dataclass
class PAEConfig:
    role: Literal["proj", "attn", "expert"] = "proj"

    host: str = "127.0.0.1"
    port: int = 8000

    num_proj_servers: int = 1
    num_attn_servers: int = 1
    num_expert_servers: int = 1

    server_rank: int = 0
