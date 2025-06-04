# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D pipeline parallelism to the Llama model.

import copy
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import ScheduleZBVZeroBubble, _PipelineSchedule, get_schedule_class
from transformers import PretrainedConfig

from flame.models.parallelize_fla import get_blocks, get_components_name, get_model
from torchtitan.config_manager import JobConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.pipeline import build_pipeline_schedule, generate_split_points, stage_ids_this_rank
from torchtitan.tools.logging import logger

DeviceType = Union[int, str, torch.device]


def pipeline_fla(
    model: nn.Module,
    pp_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: DeviceType,
    model_config: PretrainedConfig,
    loss_fn: Callable[..., torch.Tensor],
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    stages, models = pipeline_fla_manual_split(
        model, pp_mesh, parallel_dims, job_config, device, model_config
    )

    pp_schedule = build_pipeline_schedule(job_config, stages, loss_fn)

    # This is used in the train loop to determine whether to pass in the input_ids and labels
    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, models, has_first_stage, has_last_stage


def pipeline_fla_manual_split(
    whole_model: nn.Module,
    pp_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: DeviceType,
    model_config: PretrainedConfig,
) -> tuple[list[PipelineStage], list[nn.Module]]:
    """
    This API extracts one torch.nn.Module objects for the part of the model configured to run inside this stage.

    It wraps the model chunk in a ManualPipelineStage object and returns both the stage and model objects.

    The stage object is used to create a pipeline schedule, and the model object can be used for applying SPMD
    parallelism.
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_size = pp_mesh.size()

    splits = (
        job_config.experimental.pipeline_parallel_split_points
        or generate_split_points(
            job_config, parallel_dims.pp, model_config.num_hidden_layers
        )
    )

    def _build_stage(
        stage_idx: int,
        start_layer: Optional[str],
        stop_layer: Optional[str],
        is_first: bool = False,
        is_last: bool = False,
    ) -> tuple[PipelineStage, nn.Module]:
        model = copy.deepcopy(whole_model)
        if not is_first:
            # we do `model.tok_embeddings = None` here
            real_model = get_model(model)
            tok_embeddings_name = get_components_name(real_model, "tok_embeddings")
            setattr(real_model, tok_embeddings_name, None)

        drop_layers = start_layer is not None
        # Get module dictionary from get_blocks(model)
        # and Create a list of keys before modifying dictionary
        module_dict = get_blocks(model)._modules  # Store reference
        layer_names = list(module_dict.keys())

        # Iterate over the list of keys instead of `_modules.items()`
        for name in layer_names:
            # Dynamically determine prefix (blocks.* or layers.*)
            prefix = start_layer.split(".")[0] if start_layer else "layers"
            layer_name = f"{prefix}.{name}"  # Construct the correct name format

            # Ensure `drop_layers` activation is based on actual naming
            if layer_name == start_layer:
                drop_layers = False
            if layer_name == stop_layer:
                drop_layers = True

            # Delete layer if drop_layers is active
            if drop_layers:
                del module_dict[name]  # Safe deletion from stored dictionary

        if not is_last:
            # we do `model.norm = None` and `model.output = None`
            real_model = get_model(model)
            norm_name = get_components_name(real_model, "norm")
            setattr(real_model, norm_name, None)

            head_name = get_components_name(model, "lm_head")
            setattr(model, head_name, None)

        stage = PipelineStage(
            model,
            stage_idx,
            num_stages,
            device,
            group=pp_mesh.get_group("pp"),
        )
        return stage, model

    num_stages = len(splits) + 1
    stage_idx = pp_rank

    stages = []
    models = []

    schedule_class = get_schedule_class(
        job_config.experimental.pipeline_parallel_schedule
    )
    style = "v" if schedule_class == ScheduleZBVZeroBubble else "loop"

    for stage_idx in stage_ids_this_rank(pp_rank, pp_size, num_stages, style=style):
        start_layer = splits[stage_idx - 1] if stage_idx > 0 else None
        stop_layer = splits[stage_idx] if stage_idx < num_stages - 1 else None
        stage, model_chunk = _build_stage(
            stage_idx,
            start_layer,
            stop_layer,
            is_first=stage_idx == 0,
            is_last=stage_idx == num_stages - 1,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx}"
            f" with start_layer {start_layer}, stop_layer {stop_layer}"
        )
        stages.append(stage)
        models.append(model_chunk)
    return stages, models
