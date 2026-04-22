# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Pipeline parallelism for FLA models. The torchtitan 0.2+ signature is
#   pipelining_fn(model, parallel_dims, job_config, device, model_args,
#                 parallelize_fn, loss_fn)
# Flame's pipeline split currently operates on HF's AutoModelForCausalLM
# structure, which doesn't round-trip cleanly through the PP wrapper yet, so
# this entry point raises NotImplementedError. The split helper below is kept
# as a reference for the migration.

from typing import Callable, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from torchtitan.config import JobConfig
from torchtitan.distributed.parallel_dims import ParallelDims

DeviceType = Union[int, str, torch.device]


def pipeline_fla(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: DeviceType,
    model_args: PretrainedConfig,
    parallelize_fn: Callable[..., nn.Module],
    loss_fn: Callable[..., torch.Tensor],
):
    """Placeholder pipelining entry point.

    TODO[flame]:
      - match the key of models' components with the actual HF naming
      - fix post_init / tie_embedding for PP; HF auto-ties when head is None
    """
    raise NotImplementedError(
        "Pipeline parallelism is not supported in this version of flame. "
        "Set --experimental.pipeline_parallel_degree=1 (the default)."
    )
