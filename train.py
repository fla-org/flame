# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import time
from datetime import timedelta

import torch
from datasets import load_dataset
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # noqa
from flame import utils
from flame.checkpoint import CheckpointManager, TrainState
from flame.data import DataCollatorForLanguageModeling, build_dataloader
from flame.metrics import build_device_memory_monitor, build_metric_logger
from flame.optimizer import build_lr_schedulers, build_optimizers
from flame.parallelisms.parallelize_fla import parallelize_fla
from flame.parallelisms.pipeline_fla import pipeline_fla
from flame.utils import device_module, device_type
from torchtitan.config_manager import JobConfig
from torchtitan.float8 import Float8Handler
from torchtitan.logging import init_logger, logger
from torchtitan.parallelisms import ParallelDims
from torchtitan.profiling import (maybe_enable_memory_snapshot,
                                  maybe_enable_profiling)


# Enable debug tracing on failure: httgs://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    # used for colorful printing
    color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color

    if job_config.job.print_args:
        logger.info(f"{color.green}{json.dumps(job_config.to_dict(), indent=2, sort_keys=True)}{color.reset}")

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        cp=job_config.experimental.context_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not job_config.training.disable_loss_parallel,
    )
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    device_module.set_device(device)
    utils.init_distributed(job_config)
    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    utils.set_determinism(
        world_mesh,
        device,
        job_config.training.seed,
        job_config.training.deterministic
    )

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(job_config.model.tokenizer_path)
    logger.info(f"{tokenizer}")
    logger.info("Loading dataset...")
    dataset = load_dataset(
        path=job_config.training.dataset,
        name=getattr(job_config.training, "dataset_name", None),
        split=job_config.training.dataset_split,
        trust_remote_code=True,
        streaming=True
    )
    logger.info(f"{dataset}")
    logger.info("Building dataloader...")
    dataloader = build_dataloader(
        rank=dp_rank,
        world_size=dp_degree,
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=job_config.training.batch_size,
        seq_len=job_config.training.seq_len,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, varlen=job_config.training.varlen),
        num_workers=job_config.training.num_workers,
        pin_memory=job_config.training.pin_memory,
        persistent_workers=job_config.training.persistent_workers,
        snapshot_every_n_steps=job_config.checkpoint.interval
    )

    logger.info(f"Loading model config from {job_config.model.config}")
    model_config = AutoConfig.from_pretrained(job_config.model.config)
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. context_len base on inputs
    model_config.vocab_size = tokenizer.vocab_size

    logger.info(f"Building model from the config\n{color.green}{model_config}{color.reset}")
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(model_config)
        # defer weight initialization until after parallelisms are applied
        model.apply(lambda m: setattr(m, '_is_hf_initialized', False))

    # a no-op hander if float8 is not enabled
    float8_handler = Float8Handler(job_config, parallel_dims)
    # swap to Float8Linear based on float8 configs
    float8_handler.convert_to_float8_training(model)

    # log model size
    model_param_count = model.num_parameters()
    num_flop_per_token = utils.get_num_flop_per_token(
        utils.get_num_params(model, exclude_embedding=True),
        model_config,
        job_config.training.seq_len,
    )
    logger.info(
        f"{color.blue}Model\n{model}\n"
        f"{color.red}{model_param_count:,} parameters in total{color.reset}"
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
        # apply PT-D Pipeline Parallel
        pp_schedule, model_parts = pipeline_fla(
            model,
            pp_mesh,
            parallel_dims,
            job_config,
            device,
            model_config
        )

        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for m in model_parts:
            # apply SPMD-style PT-D techniques
            parallelize_fla(m, world_mesh, parallel_dims, job_config)
            m.to_empty(device=init_device)
            with torch.no_grad():
                m.post_init()
            m.train()
    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        parallelize_fla(model, world_mesh, parallel_dims, job_config)
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.post_init()
        model.train()
        model_parts = [model]

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    optimizers = build_optimizers(model_parts, job_config)
    lr_schedulers = build_lr_schedulers(optimizers.optimizers, job_config)

    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=dataloader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert (
            world_size == 1
        ), "Must create seed-checkpoint using one gpu, to disable sharding"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint.load(step=job_config.checkpoint.load_step)
    metric_logger = build_metric_logger(job_config, parallel_dims)

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {
                "optim/global_avg_loss": train_state.global_avg_losses[idx],
                "optim/global_max_loss": train_state.global_max_losses[idx],
            }
            metric_logger.log(metrics, step=step)

    data_iterator = iter(dataloader)

    train_context = utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )

    # variables used to keep info for metrics logging
    losses_since_last_log = []
    ntokens_since_last_log = 0
    data_loading_times = []
    time_last_log = time.perf_counter()
    device_memory_monitor.reset_peak_stats()

    checkpoint.reset()

    # train loop
    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {job_config.training.batch_size}, "
        f"global batch size {job_config.training.batch_size * dp_degree} "
        f"({job_config.training.batch_size * dp_degree * job_config.training.seq_len} tokens), "
        f"sequence length {job_config.training.seq_len}, "
        f"total steps {job_config.training.steps} "
        f"(warmup {job_config.training.warmup_steps})"
    )
    with maybe_enable_profiling(
        job_config, global_step=train_state.step
    ) as torch_profiler, maybe_enable_memory_snapshot(
        job_config, global_step=train_state.step
    ) as memory_profiler:
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
                input_ids, labels = batch['input_ids'], batch['labels']
                ntokens_since_last_log += labels.numel()
                data_loading_times.append(time.perf_counter() - data_load_start)

                input_ids = input_ids.to(device_type)
                labels = labels.to(device_type)
                # apply context parallelism if cp is enabled
                optional_context_parallel_ctx = (
                    utils.create_context_parallel_ctx(
                        cp_mesh=world_mesh["cp"],
                        cp_buffers=[input_ids, labels, model.freqs_cis],
                        cp_seq_dims=[1, 1, 0],
                        cp_no_restore_buffers={input_ids, labels},
                        cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                    )
                    if parallel_dims.cp_enabled
                    else None
                )

                if parallel_dims.pp_enabled:
                    # Pipeline Parallel forward / backward inside step() call
                    is_last_stage = pp_mesh.get_local_rank() == pp_mesh.size() - 1

                    with train_context(optional_context_parallel_ctx):
                        if pp_mesh.get_local_rank() == 0:
                            pp_schedule.step(input_ids)
                        elif is_last_stage:
                            losses = []
                            pp_schedule.step(target=labels, losses=losses)
                        else:
                            pp_schedule.step()

                    # accumulate losses across pipeline microbatches
                    loss = (
                        torch.mean(torch.stack(losses))
                        if is_last_stage
                        else torch.Tensor([-1.0])
                    )
                else:
                    # Non-PP forward / backward
                    with train_context(optional_context_parallel_ctx):
                        output = model(input_ids=input_ids, labels=labels)
                        loss = output.loss
                        loss.backward()
                losses.append(loss)
            loss = sum(losses) / len(losses)

            # clip gradients
            grad_norm = utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                job_config.training.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            # sync float8 amaxes and scales
            float8_handler.sync_float8_amax_and_scale_history(model_parts)

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            optimizers.step()
            lr_schedulers.step()

            # calculate float8 dynamic amax/scale for all-parameter for FSDP2
            # it issues a single all-reduce for all parameters at once for better performance
            float8_handler.precompute_float8_dynamic_scale_for_fsdp(model_parts)

            losses_since_last_log.append(loss)

            # log metrics
            if train_state.step == 1 or train_state.step % job_config.metrics.log_freq == 0:
                losses = [loss.item() for loss in losses_since_last_log]
                avg_loss, max_loss = sum(losses) / len(losses), max(losses)
                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    global_avg_loss, global_max_loss = (
                        utils.dist_mean(avg_loss, world_mesh["dp_cp"]),
                        utils.dist_max(max_loss, world_mesh["dp_cp"]),
                    )
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss

                time_delta = time.perf_counter() - time_last_log

                # update train state
                train_state.tokens += utils.dist_sum(ntokens_since_last_log, world_mesh["dp_cp"])
                train_state.elapsed += timedelta(seconds=time_delta)
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                last_lr = lr_schedulers.schedulers[0].get_last_lr()[0]
                # tokens per second per device, abbreviated as tgs
                tgs = ntokens_since_last_log / time_delta * parallel_dims.non_data_parallel_size
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # httgs://arxiv.org/abs/2204.02311
                mfu = num_flop_per_token * tgs / gpu_peak_flops

                time_end_to_end = time_delta / job_config.metrics.log_freq
                time_data_loading = sum(data_loading_times) / len(data_loading_times)
                time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

                eta = train_state.elapsed * (job_config.training.steps - train_state.step) / train_state.step

                device_mem_stats = device_memory_monitor.get_peak_stats()

                metrics = {
                    "optim/global_avg_loss": global_avg_loss,
                    "optim/global_max_loss": global_max_loss,
                    "optim/learning_rate": last_lr,
                    "optim/grad_norm": grad_norm,
                    "speed/throughput(tgs)": tgs,
                    "speed/mfu(%)": mfu,
                    "time/end_to_end(s)": time_end_to_end,
                    "time/data_loading(s)": time_data_loading,
                    "time/data_loading(%)": time_data_loading_pct,
                    "memory/max_active(GiB)": device_mem_stats.max_active_gib,
                    "memory/max_active(%)": device_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
                    "memory/num_ooms": device_mem_stats.num_ooms,
                }
                metric_logger.log(metrics, step=train_state.step)

                logger.info(
                    f"{color.cyan}step: {train_state.step:>8,} token: {train_state.tokens:>15,}  "
                    f"{color.green}loss: {global_avg_loss:7.4f}  "
                    f"{color.blue}lr: {last_lr:.4e} gnorm: {grad_norm:5.2f} "
                    f"{color.yellow}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB "
                    f"{color.red}tgs: {round(tgs):7,} mfu: {mfu:6.2%} "
                    f"{color.magenta}[{str(train_state.elapsed).split('.')[0]:>8}<{str(eta).split('.')[0]:>8}]{color.reset}"
                )

                losses_since_last_log.clear()
                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = time.perf_counter()
                device_memory_monitor.reset_peak_stats()

            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()
    torch.distributed.destroy_process_group()