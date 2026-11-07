# -*- coding: utf-8 -*-
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from flame.components.optimizer import build_optimizers, no_weight_decay_params, record_no_weight_decay, with_param_groups


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.A_log = nn.Parameter(torch.ones(4))
        self.A_log._no_weight_decay = True


class CheckpointWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self._checkpoint_wrapped_module = module


def decayed(params, optimizer):
    """Which parameters one step with zero gradients changes, i.e., which ones weight decay reaches."""
    before = [p.detach().clone() for p in params]
    for p in params:
        p.grad = torch.zeros_like(p)
    optimizer.step()
    return [not torch.equal(b, p) for b, p in zip(before, params)]


def test_marked_parameters_skip_weight_decay():
    model = Layer()
    params = list(model.parameters())
    optimizer = with_param_groups(torch.optim.AdamW, no_weight_decay_params([model]))(params, lr=0.1, weight_decay=0.5)
    assert decayed(params, optimizer) == [False if p is model.A_log else True for p in params]


def test_recorded_names_survive_replaced_parameters():
    model = Layer()
    record_no_weight_decay(model)
    # a parallelism replaces the parameter without its attribute and wraps the module
    model.A_log = nn.Parameter(torch.ones(4))
    wrapped = CheckpointWrapper(model)
    wrapped._no_weight_decay_names = model._no_weight_decay_names
    assert no_weight_decay_params([wrapped]) == {id(model.A_log)}


def job_config(respect):
    return SimpleNamespace(
        optimizer=SimpleNamespace(
            name='AdamW', lr=0.1, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.5, implementation='for-loop',
            early_step_in_backward=False, respect_no_weight_decay=respect,
        ),
        parallelism=SimpleNamespace(pipeline_parallel_degree=1),
    )


@pytest.mark.parametrize('respect', [False, True])
def test_build_optimizers(respect):
    pytest.importorskip('torchtitan.components.optimizer')
    model = Layer()
    record_no_weight_decay(model)
    optimizers = build_optimizers([model], job_config(respect), SimpleNamespace(enabled=False))
    params = list(model.parameters())
    assert decayed(params, optimizers) == [not (respect and p is model.A_log) for p in params]
