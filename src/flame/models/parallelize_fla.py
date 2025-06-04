# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.

from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed._composable.replicate import replicate
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module
)

from fla.modules.fused_linear_cross_entropy import LinearLossParallel
from fla.modules.mlp import SwiGLULinearParallel
from fla.modules.parallel import PrepareModuleWeight
from torchtitan.config_manager import TORCH_DTYPE_MAP, JobConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.tools.logging import logger


def parallelize_fla(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    if parallel_dims.tp_enabled:
        if (
            job_config.experimental.enable_async_tensor_parallel
            and not job_config.training.compile
        ):
            raise RuntimeError("Async TP requires --training.compile")
        enable_float8_linear = "float8" in job_config.model.converters
        apply_tp(
            model,
            world_mesh["tp"],
            loss_parallel=parallel_dims.loss_parallel_enabled,
            enable_float8=enable_float8_linear,
            enable_async_tp=job_config.experimental.enable_async_tensor_parallel,
        )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    # turn on per-block compile after AC wrapping and before FSDP
    if job_config.training.compile:
        apply_compile(model)

    if (
        parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled
    ):  # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)

        apply_fsdp(
            model,
            world_mesh[tuple(dp_mesh_dim_names)],
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.training.fsdp_reshard_after_forward,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            world_mesh,
            enable_compile=job_config.training.compile,
            enable_compiled_autograd=job_config.experimental.enable_compiled_autograd,
        )


class TPPlan:
    def __init__(
        self,
        model=None,
        loss_parallel=False,
        enable_float8=False,
    ):
        self.model = model
        self.loss_parallel = loss_parallel
        self.enable_float8 = enable_float8
        self.base_model_prefix = getattr(model, "base_model_prefix", "model")

        # TODO(vkuzo): once float8 configuration supports delayed scaling,
        # add a check here to enforce supported float8 all-gather configurations
        # TODO(vkuzo): add the items below to __init__.py of torchao.float8 and import from there
        try:
            from torchao.float8.float8_tensor_parallel import (
                Float8ColwiseParallel,
                Float8RowwiseParallel,
                PrepareFloat8ModuleInput
            )
        except ImportError:
            Float8ColwiseParallel = None
            Float8RowwiseParallel = None
            PrepareFloat8ModuleInput = None
        if self.enable_float8 and Float8ColwiseParallel is not None:
            self.rowwise_parallel = Float8RowwiseParallel
            self.colwise_parallel = Float8ColwiseParallel
            self.prepare_module_input = PrepareFloat8ModuleInput
            self.prepare_module_output = PrepareModuleOutput
        else:
            self.rowwise_parallel = RowwiseParallel
            self.colwise_parallel = ColwiseParallel
            self.prepare_module_input = PrepareModuleInput
            self.prepare_module_output = PrepareModuleOutput

    @property
    def model_plan(self):
        plans = {
            f"{self.base_model_prefix}.embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            f"{self.base_model_prefix}.norm": SequenceParallel(),
        }
        if self.loss_parallel:
            plans.update(
                {
                    "lm_head": ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Shard(-1) if self.loss_parallel else Replicate(),
                        use_local_output=not self.loss_parallel,
                    ),
                }
            )
        else:
            plans.update(
                {
                    "lm_head": PrepareModuleWeight(layouts=Replicate()),
                    "criterion": LinearLossParallel(),
                }
            )
        return plans

    @property
    def layer_plan(self):
        return {
            "attn_norm": SequenceParallel(),
            **self.attn_plan,
            "mlp_norm": SequenceParallel(),
            **self.mlp_plan,
        }

    @property
    def attn_plan(self):
        raise NotImplementedError(
            f"TP plans for token mixing layers of {self.model.config.model_type} not implemented"
        )

    @property
    def mlp_plan(self):
        return {
            "mlp": self.prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "mlp.gate_proj": self.colwise_parallel(),
            "mlp.up_proj": self.colwise_parallel(),
            "mlp.down_proj": self.rowwise_parallel(output_layouts=Shard(1)),
            "mlp.swiglu_linear": SwiGLULinearParallel(output_layouts=Shard(1)),
        }


class TransformerTPPlan(TPPlan):

    @property
    def attn_plan(self):
        return {
            "attn": self.prepare_module_input(
                input_kwarg_layouts={"hidden_states": Shard(1)},
                desired_input_kwarg_layouts={"hidden_states": Replicate()},
            ),
            "attn.q_proj": self.colwise_parallel(),
            "attn.k_proj": self.colwise_parallel(),
            "attn.v_proj": self.colwise_parallel(),
            "attn.o_proj": self.rowwise_parallel(output_layouts=Shard(1)),
        }


class GLATPPlan(TPPlan):

    @property
    def attn_plan(self):
        return {
            "attn": self.prepare_module_input(
                input_kwarg_layouts={"hidden_states": Shard(1)},
                desired_input_kwarg_layouts={"hidden_states": Replicate()},
            ),
            "attn.q_proj": self.colwise_parallel(),
            "attn.k_proj": self.colwise_parallel(),
            "attn.v_proj": self.colwise_parallel(),
            "attn.g_proj": self.colwise_parallel(),
            "attn.gk_proj.0": PrepareModuleWeight(layouts=Replicate()),
            "attn.gk_proj.1": self.colwise_parallel(),
            "attn.g_norm": SequenceParallel(sequence_dim=-1),
            "attn.o_proj": self.rowwise_parallel(output_layouts=Shard(1)),
        }


TP_PLAN_MAP = {"transformer": TransformerTPPlan, "gla": GLATPPlan}


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8: bool,
    enable_async_tp: bool,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    tp_plan = TP_PLAN_MAP[model.config.model_type](
        model, loss_parallel=loss_parallel, enable_float8=enable_float8
    )
    parallelize_module(model, tp_mesh, tp_plan.model_plan)

    blocks = get_blocks(model)
    if blocks is None:
        logger.warning("No block found for tensor parallelism")
    else:
        for _, block in enumerate(blocks):
            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=tp_plan.layer_plan,
            )

    if enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info(
        f"Applied {'Float8 ' if enable_float8 else ''}{'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the model"
    )


# for selective op activation checkpointing
_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
}


def _apply_ac_to_block(module: nn.Module, ac_config):
    valid_ac_modes = ("full", "selective")
    if ac_config.mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode: {ac_config.mode}. Valid modes: {valid_ac_modes}"
        )

    if ac_config.mode == "full":
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)

    assert ac_config.mode == "selective", f"{ac_config.mode}"
    use_op_sac = ac_config.selective_ac_option == "op"
    use_layer_sac = ac_config.selective_ac_option.isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(
            f"Invalid selective AC option: {ac_config.selective_ac_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )
    if use_op_sac:
        from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm
                to_save = func in _save_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )
                return (
                    CheckpointPolicy.MUST_SAVE
                    if to_save
                    else CheckpointPolicy.PREFER_RECOMPUTE
                )

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=False,
        )
    elif use_layer_sac:
        # Checkpoint every `ac_freq` of the modules passed to this function
        ac_freq = int(ac_config.selective_ac_option)
        ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
        ptd_checkpoint_wrapper._count += 1
        if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
            return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
        else:
            return module


def apply_ac(model: nn.Module, ac_config):
    """Apply activation checkpointing to the model."""
    blocks = get_blocks(model)
    if blocks is None:
        logger.warning("No block found for activation checkpointing")
        return

    for layer_id, block in blocks.named_children():
        block = _apply_ac_to_block(block, ac_config)
        blocks.register_module(layer_id, block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")


def apply_compile(model: nn.Module):
    """
    Apply torch.compile to each block, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """

    blocks = get_blocks(model)
    if blocks is None:
        logger.warning("No block found for torch.compile")
    else:
        for layer_id, block in blocks.named_children():
            block = torch.compile(block)
            blocks.register_module(layer_id, block)
        logger.info("Compiling each block with torch.compile")

    real_model = get_model(model)

    logger.info("Compiling the embedding, norm, and lm_head layers with torch.compile")
    embeddings_key = get_components_name(real_model, "tok_embeddings")
    if embeddings_key is not None:
        embeddings = torch.compile(getattr(real_model, embeddings_key), fullgraph=True)
        real_model.register_module(embeddings_key, embeddings)

    norm_key = get_components_name(real_model, "norm")
    if norm_key is not None:
        norm = torch.compile(getattr(real_model, norm_key), fullgraph=True)
        real_model.register_module(norm_key, norm)

    lm_head_key = get_components_name(model, "lm_head")
    if lm_head_key is not None:
        lm_head = torch.compile(getattr(model, lm_head_key), fullgraph=True)
        model.register_module(lm_head_key, lm_head)

    logger.info("Compiling the entire model with torch.compile")
    model = torch.compile(model)


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional):
            The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    blocks = get_blocks(model)
    if blocks is None:
        logger.warning("No block found for FSDP")
    else:
        total_blocks = len(blocks)
        for layer_id, block in enumerate(blocks):
            if reshard_after_forward_policy == "always":
                reshard_after_forward = True
            elif reshard_after_forward_policy == "never":
                reshard_after_forward = False
            elif reshard_after_forward_policy == "default":
                if pp_enabled:
                    # For PP, do not reshard after forward to avoid per-microbatch
                    # all-gathers, which can be expensive and non-overlapped
                    reshard_after_forward = False
                else:
                    # As an optimization, do not reshard after forward for the last
                    # transformer block since FSDP would prefetch it immediately
                    reshard_after_forward = int(layer_id) < total_blocks - 1
            else:
                raise ValueError(
                    f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
                )
            fully_shard(
                block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

    fully_shard(model, **fsdp_config, reshard_after_forward=not pp_enabled)


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
):
    if enable_compile:
        if enable_compiled_autograd:
            torch._dynamo.config.optimize_ddp = (
                "python_reducer_without_compiled_forward"
            )
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")


def get_model(model):
    base_model_prefix = getattr(model, "base_model_prefix", "model")
    if not hasattr(model, base_model_prefix):
        return None
    model = getattr(model, base_model_prefix)
    return model


def get_blocks(model):
    # TODO[flame]: adapt for network not using 'layers' attribute
    model = get_model(model)
    if not hasattr(model, "layers"):
        logger.warning('no "layers" in model can be found')
        return None
    return model.layers


def get_components_name(model, component_name):
    """
    We try to catch tok_embeddings, norm layers and lm_head layers
    We do not catch the layer names in the blocks, for blocks see `get_blocks`
    We assume the model has the following structure:
    LlamaForCausalLM:
        Model:
            embed_tokens,
            layers,
            norm,
        lm_head
    ***
    so, to search 'tok_embeddings' and 'norm' we need to pass `get_model(model)`
    and for 'lm_head' we need to pass `model`
    ***
    """

    if component_name == "tok_embeddings":
        if hasattr(model, "tok_embeddings"):
            return "tok_embeddings"
        elif hasattr(model, "embed_tokens"):
            return "embed_tokens"
        elif hasattr(model, "embeddings"):
            return "embeddings"
        else:
            logger.warning("No tok_embeddings found in model")
            return None

    elif component_name == "norm":
        if hasattr(model, "norm"):
            return "norm"
        elif hasattr(model, "norms"):
            return "norms"
        elif hasattr(model, "layernorm"):
            return "layernorm"
        else:
            logger.warning("No norm found in model")
            return None

    elif component_name == "lm_head":
        if hasattr(model, "lm_head"):
            return "lm_head"
        else:
            logger.warning("No lm_head found in model")
            return None
