# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from torchtitan.tools.logging import logger


def get_nparams_and_flops(model: nn.Module, model_config, seq_len: int) -> tuple[int, int]:
    nparams = sum(p.numel() for p in model.parameters())
    nparams_embedding = sum(
        sum(p.numel() for p in m.parameters())
        for m in model.children()
        if isinstance(m, nn.Embedding)
    )
    
    if hasattr(model_config, "num_heads"):
        num_heads = model_config.num_heads
    elif hasattr(model_config, "num_attention_heads"):
        num_heads = model_config.num_attention_heads
    else:
        num_heads = 1
        logger.warning("num_heads not found in model_config, defaulting to 1. ")

    l, h, q, t = (
        model_config.num_hidden_layers,
        num_heads,
        model_config.hidden_size // num_heads,
        seq_len,
    )
    # Reasoning behind the factor of 12 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t

    return nparams, num_flops_per_token
