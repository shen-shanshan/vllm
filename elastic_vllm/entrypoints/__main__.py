# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Elastic-vLLM A/E Server Entry Point.

This script provides a standalone entry point for running attn/expert servers in an PAE
(Proj-Attn-Expert) disaggregation setup.

Launch attn server:
python -m elastic_vllm.entrypoints \
/path/to/model \
'{"role": "attn"}' \
--enforce-eager

Launch expert server:
...
"""

import json

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)


def json_dict(s: str) -> dict:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
        # raise argparse.ArgumentTypeError(f"Invalid JSON: {e}")


parser = FlexibleArgumentParser()
# Add elastic-vllm custom arguments.
parser.add_argument("model", type=str)
parser.add_argument("pae_config", type=json_dict)
# Add vllm native engine arguments.
parser = AsyncEngineArgs.add_cli_args(parser)
args = parser.parse_args()
# Set the model from positional argument.
args.model = args.model

pae_config = args.pae_config
if pae_config["role"] == "attn" or pae_config["role"] == "expert":
    pass
else:
    # raise ValueError(
    #     "Invalid PAE role: %s, please use 'attn' or 'expert'", pae_config["role"]
    # )
    pass
