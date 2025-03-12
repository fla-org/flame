# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any, Callable, Dict, Generic, List, TypeVar, Union

import functools
import math
from functools import partial
import torch
import torch.nn as nn

from torch.optim import Optimizer
from torchtitan.components.optimizer import (
    LRSchedulersContainer,
    OptimizersContainer,
    OptimizersInBackwardContainer,
    FTOptimizersContainer,
)
from torchtitan.config_manager import JobConfig
from torchtitan.components.ft import FTManager, has_torchft


__all__ = [
    "build_optimizers",
    "build_lr_schedulers",
]


def build_optimizers(
    model_parts: List[nn.Module],
    job_config: JobConfig,
    ft_manager: FTManager,
) -> OptimizersContainer:
    """Create a OptimizersContainer for the given model parts and job config.

    This function creates a ``OptimizersContainer`` for the given model parts.
    ``job_config`` should define the correct optimizer name and parameters.
    This function currently supports creating ``OptimizersContainer`` and
    ``OptimizersInBackwardContainer``.

    **Note**
    Users who want to customize the optimizer behavior can create their own
    ``OptimizersContainer`` subclass and ``build_optimizers``. Passing the
    customized ``build_optimizers`` to ``TrainSpec`` will create the customized
    ``OptimizersContainer``.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        job_config (JobConfig): Job config containing the optimizer name and parameters.
    """
    optim_in_bwd = job_config.optimizer.early_step_in_backward
    if optim_in_bwd and job_config.experimental.pipeline_parallel_degree > 1:
        raise NotImplementedError(
            "Optimizers in backward is not supported with pipeline parallelism."
        )
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    eps = job_config.optimizer.eps

    optim_implementation = job_config.optimizer.implementation
    assert optim_implementation in ["fused", "foreach", "for-loop"]

    fused = optim_implementation == "fused"
    foreach = optim_implementation == "foreach"
    weight_decay = job_config.optimizer.weight_decay

    optimizer_kwargs = {
        "lr": lr,
        "eps": eps,
        "betas": (0.9, 0.95),
        "weight_decay": weight_decay,
        "fused": fused,
        "foreach": foreach,
    }

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    if name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {name} not added.")
    optimizer_cls = optimizer_classes[name]

    if optim_in_bwd and ft_manager.enabled:
        raise ValueError("TorchFT is not supported with optimizers in backward.")
    elif optim_in_bwd:
        return OptimizersInBackwardContainer(
            model_parts, optimizer_cls, optimizer_kwargs
        )
    elif ft_manager.enabled:
        return FTOptimizersContainer(
            model_parts, optimizer_cls, optimizer_kwargs, ft_manager.manager
        )
    else:
        return OptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)


def linear_scheduler_lambda(
    current_step: int,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    ratio = max(
        0.0,
        float(num_training_steps - current_step)
        / float(max(1, num_training_steps - num_warmup_steps)),
    )
    return ratio * (1 - min_lr_ratio) + min_lr_ratio


def cosine_scheduler_lambda(
    current_step: int,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.1,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1 - min_lr_ratio) + min_lr_ratio
    return max(0, factor)


def wsd_scheduler_lambda(
    current_step: int,
    num_warmup_steps: int,
    num_training_steps: int,
    decay_ratio: float = 0.1,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.1,
    decay_type: str = "sqrt",
):
    num_stable_steps = num_training_steps * (1 - decay_ratio)
    num_decay_steps = num_training_steps - num_stable_steps - num_warmup_steps
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    if current_step < num_warmup_steps + num_stable_steps:
        return 1.0
    if current_step < num_warmup_steps + num_stable_steps + num_decay_steps:
        progress = float(current_step - num_warmup_steps - num_stable_steps) / float(
            max(1, num_decay_steps)
        )
        if decay_type == "linear":
            return min_lr_ratio + (1 - min_lr_ratio) * (1 - progress)
        elif decay_type == "exp":
            return min_lr_ratio**progress
        elif decay_type == "cosine":
            return (
                min_lr_ratio
                + (1 - min_lr_ratio)
                * (1 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
                * 0.5
            )
        elif decay_type == "square":
            return min_lr_ratio + (1 - min_lr_ratio) * (1 - progress**2)
        elif decay_type == "sqrt":
            return min_lr_ratio + (1 - min_lr_ratio) * (1 - math.sqrt(progress))
        else:
            raise ValueError(
                f"decay type {decay_type} is not in ['cosine','miror_cosine','linear','exp','square','sqrt']"
            )
    return min_lr_ratio


def build_lr_schedulers(
    optimizers: OptimizersContainer, job_config: JobConfig
) -> LRSchedulersContainer:
    """Create a LRSchedulerContainer for the given optimizers and job config.

    This function creates a ``LRSchedulersContainer`` for the given optimizers.
    ``job_config`` should define the correct lr scheduler parameters.

    **Note**
    Users who want to customize the lr scheduler behavior can create their own
    ``LRSchedulersContainer`` subclass and ``build_lr_scheduler``. Passing the
    customized ``build_lr_schedulers`` to ``TrainSpec`` will create the customized
    ``LRSchedulersContainer``.


    Args:
        optimizers (OptimizersContainer): The corresponding optimizers for the
            lr_schedulers.
    """
    warmup_steps = int(job_config.training.warmup_steps)
    if job_config.optimizer.scheduler == "linear":
        lr_lambda = partial(
            linear_scheduler_lambda,
            num_warmup_steps=warmup_steps,
            num_training_steps=job_config.training.steps,
            min_lr_ratio=job_config.optimizer.min_lr_ratio,
        )
    elif job_config.optimizer.scheduler == "cosine":
        lr_lambda = partial(
            cosine_scheduler_lambda,
            num_warmup_steps=warmup_steps,
            num_training_steps=job_config.training.steps,
            min_lr_ratio=job_config.optimizer.min_lr_ratio,
        )
    elif job_config.optimizer.scheduler == "wsd":
        lr_lambda = partial(
            wsd_scheduler_lambda,
            num_warmup_steps=warmup_steps,
            num_training_steps=job_config.training.steps,
            min_lr_ratio=job_config.optimizer.min_lr_ratio,
        )
    else:
        raise ValueError(f"Scheduler {job_config.optimizer.scheduler} not supported")
    return LRSchedulersContainer(optimizers, lr_lambda)
