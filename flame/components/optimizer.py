# -*- coding: utf-8 -*-
"""
Optimizers that exclude the parameters FLA models mark with `_no_weight_decay`, e.g., `A_log`, `D` and `dt_bias`, from
weight decay.

Parallelisms may replace parameter objects, so the marked names are recorded on the model before they are applied
(`record_no_weight_decay`), and parameters still carrying the attribute afterwards are excluded as well.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

# module wrappers that parallelisms insert into parameter names
WRAPPER_PREFIXES = ('_checkpoint_wrapped_module.', '_orig_mod.', '_fsdp_wrapped_module.')


def _unwrap_name(name: str) -> str:
    for prefix in WRAPPER_PREFIXES:
        name = name.replace(prefix, '')
    return name


def record_no_weight_decay(model: nn.Module) -> None:
    """Record the names of the parameters marked with `_no_weight_decay` before parallelisms are applied."""
    model._no_weight_decay_names = {
        _unwrap_name(name) for name, param in model.named_parameters() if getattr(param, '_no_weight_decay', False)
    }


def no_weight_decay_params(model_parts: list[nn.Module]) -> set[int]:
    """The ids of the parameters to exclude from weight decay."""
    ids = set()
    for model in model_parts:
        names = getattr(model, '_no_weight_decay_names', set())
        for name, param in model.named_parameters():
            if getattr(param, '_no_weight_decay', False) or _unwrap_name(name) in names:
                ids.add(id(param))
    return ids


def with_param_groups(optimizer_cls: type[torch.optim.Optimizer], no_decay: set[int]) -> Callable:
    """Wrap `optimizer_cls` to put the parameters in `no_decay` into a parameter group without weight decay."""
    def build(params, **kwargs: Any) -> torch.optim.Optimizer:
        params = list(params)
        groups = [
            {'params': [p for p in params if id(p) not in no_decay]},
            {'params': [p for p in params if id(p) in no_decay], 'weight_decay': 0.0},
        ]
        return optimizer_cls([group for group in groups if group['params']], **kwargs)
    return build


def build_optimizers(model_parts: list[nn.Module], job_config, ft_manager):
    """`torchtitan.components.optimizer.build_optimizers`, honoring `_no_weight_decay` with
    `--optimizer.respect_no_weight_decay`."""
    from torchtitan.components import optimizer as titan_optimizer

    if not job_config.optimizer.respect_no_weight_decay:
        return titan_optimizer.build_optimizers(model_parts, job_config, ft_manager)

    optim_in_bwd = job_config.optimizer.early_step_in_backward
    if optim_in_bwd and job_config.parallelism.pipeline_parallel_degree > 1:
        raise NotImplementedError("Optimizers in backward is not supported with pipeline parallelism.")
    optimizer_classes = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}
    if job_config.optimizer.name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {job_config.optimizer.name} not added.")
    implementation = job_config.optimizer.implementation
    assert implementation in ['fused', 'foreach', 'for-loop']
    optimizer_kwargs = {
        'lr': job_config.optimizer.lr,
        'betas': (job_config.optimizer.beta1, job_config.optimizer.beta2),
        'eps': job_config.optimizer.eps,
        'weight_decay': job_config.optimizer.weight_decay,
        'fused': implementation == 'fused',
        'foreach': implementation == 'foreach',
    }
    optimizer_cls = with_param_groups(optimizer_classes[job_config.optimizer.name], no_weight_decay_params(model_parts))

    if optim_in_bwd and ft_manager.enabled:
        raise ValueError("TorchFT is not supported with optimizers in backward.")
    if optim_in_bwd:
        return titan_optimizer.OptimizersInBackwardContainer(model_parts, optimizer_cls, optimizer_kwargs)
    if ft_manager.enabled:
        return titan_optimizer.FTOptimizersContainer(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            ft_manager.manager,
            use_ft_optimizer=job_config.fault_tolerance.semi_sync_method is None,
        )
    return titan_optimizer.OptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)
