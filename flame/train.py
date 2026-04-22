# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import json
import os
import time
from datetime import timedelta

import flame  # noqa: F401  (registers torchtitan compat shims; keep this first)

import fla  # noqa: F401
import torch
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.ops.utils import prepare_position_ids
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.metrics import build_metrics_processor, ensure_pp_loss_visible
from torchtitan.components.optimizer import build_optimizers
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import custom_models  # noqa: F401
from flame.components.checkpoint import TrainState
from flame.config_manager import JobConfig
from flame.data import build_dataloader, build_dataset
from flame.models.parallelize_fla import parallelize_fla
from flame.tools.utils import get_nparams_and_flops


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    logger.info(f"Starting job: {job_config.job.description}")

    if job_config.experimental.custom_import:
        importlib.import_module(job_config.experimental.custom_import)

    # used for colorful printing
    color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color

    if job_config.job.print_args:
        logger.info(
            f"{color.green}{json.dumps(job_config.to_dict(), indent=2, sort_keys=True)}{color.reset}"
        )

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    device_module, device_type = utils.device_module, utils.device_type
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    device_module.set_device(device)

    # init distributed — torchtitan 0.2 takes the Comm sub-config, not the whole job.
    world_size = dist_utils.init_distributed(
        job_config.comm,
        enable_cpu_backend=job_config.training.enable_cpu_offload,
        base_folder=job_config.job.dump_folder,
    )

    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        cp=job_config.experimental.context_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        ep=1,
        etp=1,
        world_size=world_size,
    )

    # Build meshes up front so we can read dp info.
    parallel_dims.build_mesh()
    if parallel_dims.dp_enabled:
        batch_mesh = parallel_dims.get_mesh("batch")
        dp_degree, dp_rank = batch_mesh.size(), batch_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    # Seed & determinism.
    _set_determinism(
        parallel_dims,
        device,
        seed=job_config.training.seed,
        deterministic=job_config.training.deterministic,
    )

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        job_config.model.tokenizer_path,
        trust_remote_code=True,
        model_max_length=int(1e10),
    )
    logger.info(f"{tokenizer}")
    logger.info(
        f"Loading dataset {job_config.training.dataset}"
        f":{job_config.training.dataset_name}"
        if job_config.training.dataset_name is not None
        else ""
    )
    dataset = build_dataset(
        dataset=job_config.training.dataset,
        dataset_name=job_config.training.dataset_name,
        dataset_split=job_config.training.dataset_split,
        data_dir=job_config.training.data_dir,
        data_files=job_config.training.data_files,
        data_probs=job_config.training.data_probs,
        streaming=job_config.training.streaming,
        dp_degree=dp_degree,
        num_workers=job_config.training.num_workers,
        seed=job_config.training.seed,
    )

    logger.info("Building dataloader...")
    dataloader = build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        rank=dp_rank,
        world_size=dp_degree,
        batch_size=job_config.training.batch_size,
        seq_len=job_config.training.seq_len,
        context_len=job_config.training.context_len,
        varlen=job_config.training.varlen,
        num_workers=job_config.training.num_workers,
        pin_memory=job_config.training.pin_memory,
        persistent_workers=job_config.training.persistent_workers,
        snapshot_every_n_steps=job_config.checkpoint.interval,
    )

    logger.info(f"Loading model config from {job_config.model.config}")
    model_config = AutoConfig.from_pretrained(job_config.model.config)
    # set model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. disable fused norm if TP is enabled
    # 3. vocab size from tokenizer
    # 4. context_len based on inputs
    loss_parallel_enabled = (
        parallel_dims.tp_enabled and not job_config.training.disable_loss_parallel
    )
    if parallel_dims.tp_enabled:
        if getattr(model_config, "fuse_norm", False):
            logger.warning(
                f"{color.red}"
                f"Fused norm is not compatible with tensor parallelism. "
                f"Disabling it for now."
                f"{color.reset}"
            )
            model_config.fuse_norm = False
    if loss_parallel_enabled:
        if getattr(model_config, "fuse_linear_cross_entropy", False):
            logger.warning(
                f"{color.red}"
                f"Loss parallel enabled. Disabling fused cross entropy for now."
                f"{color.reset}"
            )
            model_config.fuse_linear_cross_entropy = False
    model_config.vocab_size = max(tokenizer.vocab_size, model_config.vocab_size)

    logger.info(
        f"Building model from the config\n{color.green}{model_config}{color.reset}"
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)
        if (
            getattr(model_config, "fuse_linear_cross_entropy", False)
            and FusedLinearCrossEntropyLoss is not None
        ):
            model.criterion = FusedLinearCrossEntropyLoss(
                num_chunks=8 // max(parallel_dims.tp, 1)
            )
        # defer weight initialization until after parallelisms are applied
        model.apply(lambda m: setattr(m, "_is_hf_initialized", False))
    logger.info(f"{color.blue}\n{model}{color.reset}\n")

    # Build the collection of model converters. No-op if `model.converters` empty
    model_converters = build_model_converters(job_config, parallel_dims)
    model_converters.convert(model)

    # calculate model size and flops per token
    model_param_count, num_flops_per_token = get_nparams_and_flops(
        model, model_config, job_config.training.context_len
    )

    # move sharded model to CPU/GPU and initialize weights via DTensor
    if job_config.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
    elif job_config.training.enable_cpu_offload:
        init_device = "cpu"
    else:
        init_device = device_type

    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "Pipeline parallelism is not supported in this version of flame."
        )
    else:
        parallelize_fla(model, parallel_dims, job_config)
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.post_init()
        model.train()

        model_parts = [model]

    # Build optimizer / scheduler — new torchtitan API takes sub-configs.
    optimizers = build_optimizers(
        model_parts,
        job_config.optimizer,
        parallel_dims,
    )
    lr_schedulers = build_lr_schedulers(
        optimizers,
        job_config.lr_scheduler,
        job_config.training.steps,
    )
    # Post optimizer step model converters hook.
    optimizers.register_step_post_hook(
        lambda *args, **kwargs: model_converters.post_optimizer_hook(model_parts)
    )

    train_state = TrainState()

    # CheckpointManager — new signature takes ``checkpoint_config`` + ``base_folder``.
    checkpoint = CheckpointManager(
        dataloader=dataloader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        checkpoint_config=job_config.checkpoint,
        sd_adapter=None,
        base_folder=job_config.job.dump_folder,
        ft_manager=None,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert world_size == 1, (
            "Must create seed checkpoint using a single device, to disable sharding"
        )
        assert job_config.checkpoint.enable, (
            "Must enable checkpointing when creating a seed checkpoint"
        )
        checkpoint.save(curr_step=0, last_step=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint.load(step=job_config.checkpoint.load_step)

    # Build metrics processor after parallelism/model are ready.
    metric_logger = build_metrics_processor(job_config, parallel_dims)
    metric_logger.num_flops_per_token = num_flops_per_token
    metric_logger.optimizers = optimizers
    metric_logger.lr_schedulers = lr_schedulers
    metric_logger.model_parts = model_parts
    gpu_peak_flops = metric_logger.gpu_peak_flops
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
    device_mem_stats = metric_logger.device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # plot losses loaded from checkpoint (if any) to TensorBoard
    if train_state.step > 0 and len(metric_logger.data_loading_times) > 0:
        for idx, step in enumerate(train_state.log_steps):
            metric_logger.log(
                step,
                global_avg_loss=train_state.global_avg_losses[idx],
                global_max_loss=train_state.global_max_losses[idx],
                grad_norm=0.0,
            )

    data_iterator = iter(dataloader)

    train_context = dist_utils.get_train_context(loss_parallel_enabled)
    maybe_enable_amp = dist_utils.maybe_enable_amp(
        parallel_dims,
        job_config.training.mixed_precision_param,
        device_type,
    )

    # variables used to keep info for metrics logging
    metric_logger.device_memory_monitor.reset_peak_stats()

    global_batch_size = (
        job_config.training.batch_size
        * dp_degree
        * job_config.training.gradient_accumulation_steps
    )
    num_tokens_per_step = global_batch_size * job_config.training.seq_len
    # train loop
    logger.info(f"{color.red}***** Running training *****{color.reset}")
    logger.info(f"{color.green}  Training starts at step {train_state.step + 1}")
    logger.info(
        f"{color.green}  Number of tokens per sequence = {job_config.training.seq_len:,}"
    )
    logger.info(
        f"{color.green}  Gradient Accumulation steps = {job_config.training.gradient_accumulation_steps}"
    )
    logger.info(
        f"{color.green}  Instantaneous batch size (per device) = {job_config.training.batch_size:,}"
    )
    logger.info(
        f"{color.green}  Global batch size (w. parallel, distributed & accumulation) = {global_batch_size:,}"
        f" ({num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Total optimization steps = {job_config.training.steps:,} "
        f"({job_config.training.steps * num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Warmup steps = {job_config.lr_scheduler.warmup_steps:,}"
        f" ({job_config.lr_scheduler.warmup_steps * num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Number of parameters = {model_param_count:,} {color.reset}"
    )

    with (
        maybe_enable_profiling(
            job_config.profiling,
            global_step=train_state.step,
            base_folder=job_config.job.dump_folder,
        ) as torch_profiler,
        maybe_enable_memory_snapshot(
            job_config.profiling,
            global_step=train_state.step,
            base_folder=job_config.job.dump_folder,
        ) as memory_profiler,
    ):
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            optimizers.zero_grad()

            losses = []
            # do gradient accumulation if enabled
            for _ in range(job_config.training.gradient_accumulation_steps):
                # get batch
                data_load_start = time.perf_counter()
                batch = next(data_iterator)
                input_ids, labels = batch["input_ids"], batch["labels"]

                # Update metrics processor state before forward/backward
                metric_logger.ntokens_since_last_log += labels.numel()
                metric_logger.data_loading_times.append(
                    time.perf_counter() - data_load_start
                )

                input_ids = input_ids.to(device_type)
                labels = labels.to(device_type)
                cu_seqlens = (
                    batch["cu_seqlens"].to(device_type)
                    if "cu_seqlens" in batch
                    else None
                )
                if cu_seqlens is not None:
                    position_ids = prepare_position_ids(cu_seqlens).to(torch.int32)
                else:
                    position_ids = (
                        torch.arange(0, input_ids.shape[1], device=device_type)
                        .repeat(input_ids.shape[0], 1)
                        .to(torch.int32)
                    )

                optional_context_parallel_ctx = (
                    dist_utils.create_context_parallel_ctx(
                        cp_mesh=parallel_dims.get_mesh("cp"),
                        cp_buffers=[input_ids, labels, position_ids],
                        cp_seq_dims=[1, 1, 1],
                        cp_no_restore_buffers={input_ids, labels, position_ids},
                        cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                    )
                    if parallel_dims.cp_enabled
                    else None
                )

                # Non-PP forward / backward
                with train_context():
                    if optional_context_parallel_ctx is not None:
                        with optional_context_parallel_ctx:
                            with maybe_enable_amp:
                                output = model(
                                    input_ids=input_ids,
                                    labels=labels,
                                    position_ids=position_ids,
                                    cu_seqlens=cu_seqlens,
                                )
                    else:
                        with maybe_enable_amp:
                            output = model(
                                input_ids=input_ids,
                                labels=labels,
                                position_ids=position_ids,
                                cu_seqlens=cu_seqlens,
                            )
                    loss = output.loss / job_config.training.gradient_accumulation_steps
                    loss.backward()

                losses.append(loss)
            loss = sum(losses)

            # clip gradients
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                job_config.training.max_norm,
                foreach=True,
                pp_mesh=parallel_dims.get_optional_mesh("pp"),
                ep_enabled=parallel_dims.ep_enabled,
            )

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            if job_config.training.skip_nan_inf and (
                grad_norm.isnan() or grad_norm.isinf()
            ):
                logger.warning(
                    f"Skipping optimizer step - detected invalid gradient norm: {grad_norm:.4f}"
                )
                optimizers.zero_grad()
                train_state.skipped_step += 1
            else:
                optimizers.step()
            lr_schedulers.step()

            # log metrics
            if metric_logger.should_log(train_state.step):
                if parallel_dims.dp_cp_enabled:
                    loss = loss.detach()
                    loss_mesh = parallel_dims.get_optional_mesh("loss")
                    global_avg_loss = dist_utils.dist_mean(loss, loss_mesh)
                    global_max_loss = dist_utils.dist_max(loss, loss_mesh)
                else:
                    global_avg_loss = global_max_loss = loss.item()

                # Update train state tokens and elapsed time
                time_now = time.perf_counter()
                time_delta = time_now - metric_logger.time_last_log
                train_state.token += (
                    metric_logger.ntokens_since_last_log
                    * parallel_dims.world_size
                    / parallel_dims.non_data_parallel_size
                )
                train_state.elapsed += timedelta(seconds=time_delta)
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                last_lr = lr_schedulers.schedulers[0].get_last_lr()[0]
                eta = (
                    train_state.elapsed
                    * (job_config.training.steps - train_state.step)
                    / train_state.step
                )
                metric_logger.log(
                    train_state.step,
                    global_avg_loss,
                    global_max_loss,
                    grad_norm.item(),
                    extra_metrics={
                        "optimizer/lr": last_lr,
                        "optimizer/skipped_step": train_state.skipped_step,
                    },
                )

                logger.info(
                    f"{color.blue}lr: {last_lr:.4e} gnorm: {grad_norm:5.2f} "
                    f"{color.magenta}[{str(train_state.elapsed).split('.')[0]:>8}<{str(eta).split('.')[0]:>8}]{color.reset}"
                )

            checkpoint.save(
                train_state.step,
                last_step=(train_state.step == job_config.training.steps),
            )

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                dist_utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    parallel_dims=parallel_dims,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


def _set_determinism(
    parallel_dims: ParallelDims,
    device: torch.device,
    seed: int,
    deterministic: bool,
) -> None:
    """Seed + optional deterministic mode.

    torchtitan 0.2's ``set_determinism`` takes a ``DebugConfig`` dataclass,
    but flame keeps its training/debug knobs together. So we reimplement the
    logic inline against raw values.
    """
    if deterministic:
        logger.info("Deterministic algorithms enabled (expect perf degradation).")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if parallel_dims.world_size == 1:
        if seed is not None:
            torch.manual_seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed % 2 ** 32)
        return

    # Start from a common seed, then offset by PP rank so SPMD groups share a
    # seed while PP stages differ.
    if seed is None:
        seed_tensor = torch.get_rng_state()[:8].to(device)
        torch.distributed.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.to("cpu").view(torch.uint64).item())

    pp_mesh = parallel_dims.get_optional_mesh("pp")
    if pp_mesh is not None:
        seed = (seed + pp_mesh.get_local_rank()) % (2 ** 64)

    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed % 2 ** 32)

    if parallel_dims.world_size > parallel_dims.pp:
        torch.distributed.tensor._random.manual_seed(seed, parallel_dims.world_mesh)


if __name__ == "__main__":
    init_logger()
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()
