# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from elastic_vllm.pae_config import PAEConfig


class PAEConnector:
    """
    Description...

    Comm pipeline (within one forward):
    P: qkv_proj()
    P: send_qkv_proj_output() --> A: recv_qkv_proj_output()
                                  A: attn()
    P: recv_attn_output() <------ A: send_attn_output()
    P: out_proj()
    P: send_out_proj_output() --> E: recv_out_proj_output()
                                  E: moe()
    P: recv_expert_output() <---- E: send_expert_output()
    """

    def __init__(self, rank: int, config: "PAEConfig"):
        self.rank = rank
        self.config = config
        self._initialized: bool = False

    def init_pae_connector(self):
        ...
        self._initialized = True

    def _send_data(self): ...

    def _recv_data(self): ...

    ############################################################################
    #                          Interfaces for P node                           #
    ############################################################################
    def send_qkv_proj_output(self, hidden_states: torch.Tensor):
        """Send the output of qkv_proj to A node."""
        ...

    def recv_attn_output(self):
        """Receive the output of attn from A node"""
        ...

    def send_out_proj_output(self, hidden_states: torch.Tensor):
        """Send the output of out_proj to E node."""
        ...

    def recv_expert_output(self):
        """Receive the output of expert from E node"""
        ...

    ############################################################################
    #                          Interfaces for A node                           #
    ############################################################################
    def recv_qkv_proj_output(self):
        """Receive the output of qkv_proj from P node."""
        ...

    def send_attn_output(self, hidden_states: torch.Tensor):
        """Send the output of attn to P node"""
        ...

    ############################################################################
    #                          Interfaces for E node                           #
    ############################################################################
    def recv_out_proj_output(self):
        """Receive the output of out_proj from P node."""
        ...

    def send_expert_output(self, hidden_states: torch.Tensor):
        """Send the output of expert to P node"""
        ...
