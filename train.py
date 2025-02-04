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
from datasets import interleave_datasets, load_dataset
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # noqa
from flame import utils
from flame.checkpoint import CheckpointManager, TrainState
from flame.config_manager import JobConfig
from flame.data import build_dataloader, shuffle
from flame.metrics import build_device_memory_monitor, build_metric_logger
from flame.optimizer import build_lr_schedulers, build_optimizers
from flame.parallelisms.parallelize_fla import parallelize_fla
from flame.parallelisms.pipeline_fla import pipeline_fla
from flame.utils import device_module, device_type
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
    tokenizer = AutoTokenizer.from_pretrained(
        job_config.model.tokenizer_path,
        trust_remote_code=True,
        model_max_length=int(1e10)
    )
    logger.info(f"{tokenizer}")
    logger.info(
        f"Loading dataset {job_config.training.dataset}"
        f":{job_config.training.dataset_name}" if job_config.training.dataset_name is not None else ""
    )

    min_num_shards = dp_degree * job_config.training.num_workers
    if len(job_config.training.dataset.split(',')) == 1:
        dataset = load_dataset(
            path=job_config.training.dataset,
            name=getattr(job_config.training, 'dataset_name', None),
            data_dir=getattr(job_config.training, 'data_dir', None),
            data_files=getattr(job_config.training, 'data_files', None),
            split=job_config.training.dataset_split or "train",
            trust_remote_code=True,
            streaming=job_config.training.streaming,
            num_proc=job_config.training.num_workers if not job_config.training.streaming else None
        )
        logger.info(f"{dataset}")

        logger.info(f"Shuffling the dataset with seed {job_config.training.seed}")
        if not job_config.training.streaming:
            # the states of map-style dataset is recoverable after shuffling
            dataset = dataset.shuffle(
                seed=job_config.training.seed
            ).to_iterable_dataset(num_shards=min_num_shards)
        else:
            if dataset.num_shards < min_num_shards:
                logger.warning(
                    f"{color.red}"
                    f"Dataset {job_config.training.dataset} has insufficient shards ({dataset.num_shards}). "
                    f"Need {min_num_shards} shards minimum for {dp_degree} data parallel workers × "
                    f"{job_config.training.num_workers} dataloader workers. "
                    f"Resharding dataset to {min_num_shards} shards and disabling streaming mode."
                    f"{color.reset}"
                )
                dataset = (
                    load_dataset(
                        path=job_config.training.dataset,
                        name=getattr(job_config.training, "dataset_name", None),
                        data_dir=getattr(job_config.training, "data_dir", None),
                        data_files=getattr(job_config.training, "data_files", None),
                        split=job_config.training.dataset_split or "train",
                        trust_remote_code=True,
                        streaming=False,
                        num_proc=job_config.training.num_workers,
                    )
                    .shuffle(seed=job_config.training.seed)
                    .to_iterable_dataset(num_shards=min_num_shards)
                )
            else:
                dataset = shuffle(dataset, seed=job_config.training.seed)
    else:
        datasets = job_config.training.dataset.split(',')
        if job_config.training.dataset_name is not None:
            dataset_names = [name or None for name in job_config.training.dataset_name.split(',')]
            assert len(dataset_names) == len(datasets), "The number of dataset names must match the number of datasets"
        else:
            dataset_names = [None] * len(datasets)
        if job_config.training.dataset_split is not None:
            dataset_splits = [split or 'train' for split in job_config.training.dataset_split.split(',')]
            assert len(dataset_splits) == len(datasets), "The number of dataset splits must match the number of datasets"
        else:
            dataset_splits = ['train'] * len(datasets)
        if job_config.training.data_dir is not None:
            data_dirs = [data_dir or None for data_dir in job_config.training.data_dir.split(',')]
            assert len(data_dirs) == len(datasets), "The number of data dirs must match the number of datasets"
        else:
            data_dirs = [None] * len(datasets)
        if job_config.training.data_files is not None:
            data_files = job_config.training.data_files.split(',')
            assert len(data_files) == len(datasets), "The number of data files must match the number of datasets"
        else:
            data_files = [None] * len(datasets)
        if job_config.training.data_probs is not None:
            data_probs = [float(p) for p in job_config.training.data_probs.split(',')]
            assert len(data_probs) == len(datasets), "The number of data probabilities must match the number of datasets"
        else:
            raise ValueError("Data sampling probabilities are required if using multiple datasets")

        subsets = []
        for i, prob in enumerate(data_probs):
            subset = load_dataset(
                path=datasets[i],
                name=dataset_names[i],
                data_dir=data_dirs[i],
                data_files=data_files[i],
                split=dataset_splits[i],
                trust_remote_code=True,
                streaming=job_config.training.streaming,
                num_proc=job_config.training.num_workers if not job_config.training.streaming else None
            )
            logger.info(
                f"Subset {color.cyan}{datasets[i]}" + (f":{dataset_names[i]} " if dataset_names[i] else " ") +
                f"(p = {prob:.3f}){color.reset}:\n" +
                f"{subset}"
            )

            logger.info(f"Shuffling the dataset with seed {job_config.training.seed}")
            if not job_config.training.streaming:
                # the states of map-style dataset is recoverable after shuffling
                subset = subset.shuffle(seed=job_config.training.seed).to_iterable_dataset(num_shards=min_num_shards)
            else:
                if subset.num_shards < min_num_shards:
                    logger.warning(
                        f"{color.red}"
                        f"Dataset {datasets[i]} has insufficient shards ({subset.num_shards}). "
                        f"Need {min_num_shards} shards minimum for {dp_degree} data parallel workers × "
                        f"{job_config.training.num_workers} dataloader workers. "
                        f"Resharding dataset to {min_num_shards} shards and disabling streaming mode."
                        f"{color.reset}"
                    )
                    # again, it's ok to directly shuffle the map-style dataset
                    # we expect an error raised if the map-style dataset still has not enough data shards
                    subset = (
                        load_dataset(
                            path=datasets[i],
                            name=dataset_names[i],
                            data_dir=data_dirs[i],
                            data_files=data_files[i],
                            split=dataset_splits[i],
                            trust_remote_code=True,
                            streaming=False,
                            num_proc=job_config.training.num_workers,
                        )
                        .shuffle(seed=job_config.training.seed)
                        .to_iterable_dataset(min_num_shards)
                    )
                else:
                    # we set relatively small buffer size here as interleaving could provide some randomness
                    subset = shuffle(
                        subset,
                        seed=job_config.training.seed,
                        buffer_size=max(128, 1024 // len(datasets)),
                    )

            if 'text' in subset.column_names:
                subset = subset.select_columns('text')
            elif 'content' in subset.column_names:
                subset = subset.select_columns('content')
            else:
                raise ValueError(f"Subset {datasets[i]} has no 'text' or 'content' column")
            subsets.append(subset)

        logger.info(f"Interleaving {len(subsets)} datasets with probabilities {data_probs}")
        dataset = interleave_datasets(
            datasets=subsets,
            probabilities=data_probs,
            stopping_strategy='all_exhausted',
            seed=job_config.training.seed
        )
        logger.info(f"{dataset}")

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
    logger.info(f"{color.blue}\n{model}{color.reset}\n")

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
    ntokens_since_last_log = 0
    data_loading_times = []
    time_last_log = time.perf_counter()
    device_memory_monitor.reset_peak_stats()

    checkpoint.reset()

    global_batch_size = job_config.training.batch_size * dp_degree * job_config.training.gradient_accumulation_steps
    num_tokens_per_step = global_batch_size * job_config.training.seq_len
    # train loop
    logger.info(f"{color.red}***** Running training *****{color.reset}")
    logger.info(f"{color.green}  Training starts at step {train_state.step + 1}")
    logger.info(f"{color.green}  Number of tokens per sequence = {job_config.training.seq_len:,}")
    logger.info(f"{color.green}  Gradient Accumulation steps = {job_config.training.gradient_accumulation_steps}")
    logger.info(f"{color.green}  Instantaneous batch size (per device) = {job_config.training.batch_size:,}")
    logger.info(f"{color.green}  Global batch size (w. parallel, distributed & accumulation) = {global_batch_size:,}"
                f" ({num_tokens_per_step:,} tokens)")
    logger.info(f"{color.green}  Total optimization steps = {job_config.training.steps:,} "
                f"({job_config.training.steps * num_tokens_per_step:,} tokens)")
    logger.info(f"{color.green}  Warmup steps = {job_config.training.warmup_steps:,}"
                f" ({job_config.training.warmup_steps * num_tokens_per_step:,} tokens)")
    logger.info(f"{color.green}  Number of parameters = {model_param_count:,} {color.reset}")

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

                """
                TODO[flame]: We need to carefully handle the position_ids for TP/CP
                Depending on the Models'PE, the position_ids might be different.

                e.g. for TP
                    For RoPE, all ranks have the same position_ids. [FOR HF model]
                    For sinusoidal, each rank has the coresponding chunked  position_ids. [FOR HF model]

                e.g. for CP, [optional_context_parallel_ctx shoudl automatically distbute the position_ids]
                    Each rank has the coresponding chunked position_ids. [FOR All model]

                """
                position_ids = torch.arange(
                    0, input_ids.shape[1], device=device_type
                ).repeat(input_ids.shape[0], 1)

                labels = labels.to(device_type)
                cu_seqlens = (
                    batch["cu_seqlens"].to(device_type)
                    if "cu_seqlens" in batch
                    else None
                )
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
                # #! TODO[flame], we should distribute the position_ids as well with CP

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
                    # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                    loss = (
                        torch.mean(torch.stack(losses)).to(device)
                        if is_last_stage
                        else torch.tensor([-1.0], device=device)
                    )
                else:
                    # Non-PP forward / backward
                    with train_context(optional_context_parallel_ctx):
                        output = model(
                            input_ids=input_ids,
                            labels=labels,
                            position_ids=position_ids,
                            cu_seqlens=cu_seqlens,
                        )
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
            if job_config.training.skip_nan_inf and (grad_norm.isnan() or grad_norm.isinf()):
                logger.warning(f"Skipping optimizer step - detected invalid gradient norm: {grad_norm:.4f}")
                optimizers.zero_grad()
                train_state.skipped_step += 1
            else:
                optimizers.step()
            lr_schedulers.step()

            # calculate float8 dynamic amax/scale for all-parameter for FSDP2
            # it issues a single all-reduce for all parameters at once for better performance
            float8_handler.precompute_float8_dynamic_scale_for_fsdp(model_parts)

            # log metrics
            if (
                train_state.step == 1
                or train_state.step % job_config.metrics.log_freq == 0
            ):
                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    loss = loss.detach()
                    global_avg_loss, global_max_loss = (
                        utils.dist_mean(loss, world_mesh["dp_cp"]),
                        utils.dist_max(loss, world_mesh["dp_cp"]),
                    )
                else:
                    global_avg_loss = global_max_loss = loss.item()

                time_delta = time.perf_counter() - time_last_log

                """
                TODO[flame]: check this after fixing TP/PP/CP, if we assume all ranks have the same number of tokens
                we can just multiple it by DP's dims, this avoiding to deal with the case taht dp does nto exists
                """
                # update train state
                # train_state.token += (
                #     utils.dist_reduce(
                #         torch.tensor(ntokens_since_last_log, device=device),
                #         "sum",
                #         world_mesh["dp_cp"],
                #     )
                #     / parallel_dims.non_data_parallel_size
                # )

                # update train state
                train_state.token += (
                    ntokens_since_last_log
                    * parallel_dims.world_size
                    / parallel_dims.non_data_parallel_size
                )

                train_state.elapsed += timedelta(seconds=time_delta)
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                last_lr = lr_schedulers.schedulers[0].get_last_lr()[0]
                # tokens per second per device, abbreviated as tgs
                tgs = ntokens_since_last_log / (time_delta * parallel_dims.non_data_parallel_size)
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
                    "optim/skipped": train_state.skipped_step,
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
                    f"{color.cyan}step: {train_state.step:>8,} token: {train_state.token:>15,}  "
                    f"{color.green}loss: {global_avg_loss:7.4f}  "
                    f"{color.blue}lr: {last_lr:.4e} gnorm: {grad_norm:5.2f} "
                    f"{color.yellow}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB "
                    f"{color.red}tgs: {round(tgs):7,} mfu: {mfu:6.2%} "
                    f"{color.magenta}[{str(train_state.elapsed).split('.')[0]:>8}<{str(eta).split('.')[0]:>8}]{color.reset}"
                )

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
