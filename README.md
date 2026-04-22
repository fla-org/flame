<div align="center">

# 🔥 Flame: Flash Language Modeling Made Easy

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/fla-org/flame)

</div>

Welcome to 🔥 `flame`, a minimal and efficient framework built on `torchtitan` for language models with blazing efficiency.

**Feature Highlights:**

- 🚀 Minimal, easy-to-use, extensible training framework
- 🤗 Seamless integration with `fla` and `transformers`
- 🔄 Zero-cost data preprocessing: online tokenization, dataset shuffling, and multiple datasets support
- 🔮 4D parallelism (coming soon)

## Setup

To get started, clone the `flame` repository and install the required dependencies:

```bash
git clone https://github.com/fla-org/flame.git
cd flame
pip install .
```

Install the latest version of fla
```
pip uninstall flash-linear-attention && pip install -U --no-use-pep517 git+https://github.com/fla-org/flash-linear-attention
```

Install torchtitan (>= 0.2):
```
pip install torchtitan
```
`flame` tracks torchtitan's 0.2 API (new dataclass-based `JobConfig`, `parallelism.*`
section, renamed checkpoint/lr_scheduler keys). Older torchtitan releases are
not supported.


## Dataset Preparation
To download the dataset to your local disk, create a new Python file with the following content and execute it:

```py
from datasets import load_dataset

# load fineweb-edu with parallel processing
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="default", num_proc=64, cache_dir="/your/cache/path")

# or load a subset with roughly 100B tokens, suitable for small- or medium-sized experiments
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", num_proc=64, cache_dir="/your/cache/path")
```

## Training Recipes

Here's an example of training a 340M FLA Transformer model with a LLaMA-like architecture from scratch on a 100BT subset of the Fineweb-edu corpus ~~in streaming mode~~. (Do not use streaming mode if you are concerned about resuming training.)
 
> [!WARNING]
> If the dataset is not downloaded beforehand, the streaming mode will attempt to fetch it from a remote server and download it on-the-fly, which can be highly unstable during training due to network issues.  
> For stable training, ensure the dataset is downloaded locally (see [**Dataset Preparation**](#dataset-preparation)). Otherwise, we assume you are only testing the new corpus.

```sh
bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/transformer-340M-4K-10B/batch1.seqlen65536.context4096.warmup1024.update1.steps20480.lr1e-3.cosine \
  --model.config configs/transformer_340M.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 1e-3 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.min_lr_factor 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 1 \
  --training.seq_len 65536 \
  --training.context_len 4096 \
  --training.varlen \
  --training.gradient_accumulation_steps 1 \
  --training.steps 20480 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-100BT \
  --training.dataset_split train \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --checkpoint.enable \
  --checkpoint.interval 2048 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 1
```

You can specify the number of GPUs by setting the environment variable `NGPU`, which defaults to 8.  
**For single-GPU debugging, set `NGPU=1`.**

We provide several [config files](https://github.com/fla-org/flame/tree/main/configs) for different models.
By default, the learning rate is set to 1e-3 with a cosine scheduler. Other schedulers, such as WSD (wsd), are also supported.

**Key parameters:**
- `--lr_scheduler.decay_ratio`: The proportion of the steps allocated to the decay phase. The learning rate will remain stable after the warmup period and only start decaying during the last `decay_ratio` portion of the total training steps, which is known as the Warmup-Stable-Decay (WSD) schedule.
- `--lr_scheduler.warmup_steps`: The number of steps for the learning rate warmup phase.
- `--training.steps`: Total number of training steps.
- `--training.batch_size`: Batch size per device, must be 1 if `--training.varlen` is set.
- `--training.seq_len`: The length of each sequence in the batch, which is concatenated from multiple samples.
- `--training.context_len`: The max allowed length of a sample. For non-varlen mode, this is equivalent to `seq_len`.
- `--training.varlen`: Whether to conduct variable-length sequence training.
- `--training.gradient_accumulation_steps`: Number of gradient accumulation steps.

> [!WARNING]
> The total number of tokens processed per batch, referred to as `global_batch_size`, is calculated as batch_size × gradient_accumulation_steps × num_gpus.
> Each step processes `global_batch_size * seq_len` tokens.  
> Monitor the value of `global_batch_size`, `warmup_steps`, and `steps` carefully when modifying any of the hyperparameters!

For the full set of flags, run:

```sh
bash train.sh -h
```

Notable flag groups (torchtitan >= 0.2 naming):

- `--job.*`: run metadata and dump folder.
- `--model.config` / `--model.tokenizer_path` / `--model.converters`: HF model config,
  tokenizer, and optional converters (e.g. `float8`).
- `--training.*`: `batch_size`, `seq_len`, `context_len`, `varlen`, `steps`,
  `gradient_accumulation_steps`, `tensor_parallel_degree`,
  `data_parallel_{replicate,shard}_degree`, `compile`, `mixed_precision_param`,
  `fsdp_reshard_after_forward`, `dataset` / `dataset_name` / `data_probs` etc.
- `--experimental.*`: `pipeline_parallel_degree` (not yet implemented in flame),
  `context_parallel_degree`, `enable_async_tensor_parallel`,
  `enable_compiled_autograd`, `custom_import`.
- `--lr_scheduler.*`: `warmup_steps`, `decay_type`, `decay_ratio`,
  `min_lr_factor`, `total_steps`.
- `--checkpoint.*`: `enable`, `interval`, `folder`, `async_mode`,
  `last_save_model_only`, `last_save_in_hf`, `initial_load_path`,
  `initial_load_model_only`, `initial_load_in_hf`,
  `enable_first_step_checkpoint`, `keep_latest_k`, `load_step`,
  `exclude_from_loading`.
- `--activation_checkpoint.{mode,selective_ac_option}`.
- `--comm.{init_timeout_seconds,train_timeout_seconds,trace_buf_size,mode}`.
- `--metrics.{log_freq,enable_tensorboard,enable_wandb,save_for_all_ranks}`.

<details>
<summary>Legacy (pre-0.2) flag notes</summary>

Some flag names changed when tracking torchtitan 0.2. BC aliases were removed,
so update any scripts that still use the old names:

| Old                                                 | New                                    |
| --------------------------------------------------- | -------------------------------------- |
| `--checkpoint.enable_checkpoint`                    | `--checkpoint.enable`                  |
| `--checkpoint.last_save_model_weights_only`         | `--checkpoint.last_save_model_only`    |
| `--checkpoint.initial_load_model_weights_only`      | `--checkpoint.initial_load_model_only` |
| `--checkpoint.interval_type` / `model_weights_only` | removed (no longer defined)            |
| `--lr_scheduler.lr_min`                             | `--lr_scheduler.min_lr_factor`         |
| `--experimental.custom_model_path`                  | `--experimental.custom_import`         |

</details>

### Training with variable-length inputs
When you set the `--training.varlen` flag, you're enabling a more efficient training method that packs multiple documents together into a single long sequence, eliminating the need for padding. 
This is particularly useful when your dataset contains documents of varying lengths. 
Let's break down how `--training.seq_len` and `--training.context_len` work in this mode.

* `--training.seq_len` (Packed Sequence Length): This is the total length of the final sequence fed to the model on one device. Instead of processing one document at a time, the dataloader takes multiple documents (each split to sequences no longer than `context_len`), concatenates them end-to-end, and creates a single long sequence of length `seq_len`.
* `--training.context_len` (Sample Length): This parameter defines the maximum number of tokens for a single document or sample. If a document from the dataset is longer than `context_len`, it will be truncated. For example, if `--training.context_len` is set to 4,096, a document with 5,000 tokens will be cut down to its first 4,096 tokens, leaving the left tokens as another independent sequence, while a document with 3000 tokens remains unchanged.

### Training with `torch.compile`

Starting from `torch 2.0`, `torch.compile` has been introduced as a new feature to seamlessly accelerate training processes.
In `flame`, one can simply enable `torch.compile` by adding `--training.compile` flag to your training script.

However, `fla` has integrated numerous fused kernels for acceleration, which may potentially conflict with `torch.compile`.
We are actively working on resolving these issues to make compilation transparent to users.
In the meantime, please ensure you are using the latest dependencies.

Specifically, **we recommend using `torch>=2.6` and `triton>=3.0`**.

### Training with multiple datasets

If you wish to train a model with all-round capabilities (e.g., code, math, and multilingual ability), it's necessary to train on multiple datasets.
`flame` allows training with multiple datasets easily.
For example, you can specify the following arguments to train on 6 datasets with different proportions:

```sh
  --training.dataset HuggingFaceFW/fineweb-edu,opencsg/Fineweb-Edu-Chinese-V2.1,OpenCoder-LLM/opc-fineweb-code-corpus,math-ai/AutoMathText,EleutherAI/proof-pile-2,OpenCoder-LLM/opc-fineweb-math-corpus   \
  --training.data_probs 0.6,0.15,0.15,0.014,0.058,0.028     \
```

### ~Finalizing training~

> [!NOTE]  
> We have done this conversion automatically in the training script since our latest updates.

Once training is complete, you may want to convert the distributed checkpoints (DCPs) into the 🤗 format for broader use.
To facilitate this, we provide a straightforward conversion script:

```sh
python -m flame.utils.convert_dcp_to_hf --path <path_to_model> --step <step> --config <path_to_config> --tokenizer <path_to_tokenizer>
```
After this, your model will be in the 🤗 format, ready to be shared or deployed.
You can then easily publish your model using the `huggingface_hub` for wider accessibility.

### Continual training

If you wish to build upon a strong pre-trained model (in 🤗 format) and continue training, we also offer a script to convert the 🤗 format model back into DCP format.
This allows you to seamlessly resume training with `flame`.
```sh
python -m flame.utils.convert_hf_to_dcp --model <path_to_hf> --checkpoint <path_to_dcp/checkpoint/step-0>
```
Here, `<path_to_dcp>` is the directory where your distributed checkpoints will be stored.
The checkpoint is intentionally saved at `<step-0>` within the checkpoint folder to ensure it is loadable by `flame` during the initial training step, similar to how a seed checkpoint is handled.

Once the conversion is complete, you can proceed with training using `flame` as usual, continuing from where the pretrained model left off.

## Multi-node training

If you have access to multi-node GPUs, consider leveraging them for optimal performance.
This process is straightforward and well-documented in the PyTorch [docs](https://pytorch.org/docs/stable/elastic/run.html).

To set up multi-node training:
* Set the environment variables `MASTER_ADDR=<ip>` and `MASTER_PORT=<port>` before running the training script across all nodes.
* If you're using a job scheduler like Slurm, it will handle these variables for you.

`torchtitan` provides a [Slurm script](https://github.com/pytorch/torchtitan/blob/main/multinode_trainer.slurm) for multi-node training, which you can use as a reference or starting point.

## Custom models

`flame` supports custom model architectures through seamless integration with the Hugging Face `transformers` library. To add your own model:

1. Create a new model directory under `custom_models/` (see `custom_models/sba` for a complete example)
2. Implement your model classes and configuration:
   - Define a config class inheriting from `PretrainedConfig` (see `custom_models/sba/config_sba.py` for an example)
   - Create model classes inheriting from `PreTrainedModel` (see `custom_models/sba/modeling_sba.py` for an example)
3. Register your models in `__init__.py`:
   - Import your model classes and config classes
   - Register your models with the `AutoModelForCausalLM`, `AutoModel` and `AutoConfig` classes (see `custom_models/sba/__init__.py` for an example)
4. Create a config file for your custom model, just need to specify the `model_type` to the one you just named for your custom model (example: `configs/sba_340m.json`).
5. Training is extremely simple, you can just use the `flame.train.py` script to train your custom model.







## Citation

If you find `flame` helpful for your work, please consider citing it.

```bib
@software{yang2025flame,
  title  = {Flame: Flash Language Modeling Made Easy},
  author = {Zhang, Yu and Yang, Songlin},
  url    = {https://github.com/fla-org/flame},
  month  = jan,
  year   = {2025}
}
```
