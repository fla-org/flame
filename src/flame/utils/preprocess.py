# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
from typing import Any, Dict, List

from transformers import AutoTokenizer, PreTrainedTokenizer

from flame.data import build_dataset
from torchtitan.tools.logging import init_logger, logger


def tokenize(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizer,
) -> Dict:
    if 'text' in examples:
        samples = examples['text']
    elif 'content' in examples:
        samples = examples['content']
    else:
        raise ValueError(f'No "text" or "content" field found in examples:\n{examples}')
    input_ids = tokenizer(samples)['input_ids']
    bits_per_token = [len(sample.encode(encoding='utf-8')) * 8 / len(input_ids[i]) for i, sample in enumerate(samples)]
    return {'input_ids': input_ids, 'bits_per_token': bits_per_token}


if __name__ == '__main__':
    init_logger()
    parser = argparse.ArgumentParser(description='Preprocess the dataset.')
    parser.add_argument(
        '--dataset',
        default='HuggingFaceFW/fineweb-edu',
        help='Dataset to use, with comma separated values',
    )
    parser.add_argument(
        '--dataset_name',
        default='sample-100BT',
        help='The name of the dataset config, with comma separated values if provided',
    )
    parser.add_argument(
        '--dataset_split',
        default='train',
        help='Dataset split to use, with comma separated values if provided',
    )
    parser.add_argument(
        '--data_dir',
        default=None,
        help='Data dirs to use, with comma separated values if provided',
    )
    parser.add_argument(
        '--data_files',
        default=None,
        help='Data files to use, with comma separated values if provided',
    )
    parser.add_argument(
        '--data_probs',
        default=None,
        help='Data sampling probabilities, with comma separated values if provided',
    )
    parser.add_argument(
        '--streaming',
        action='store_true',
        help='Whether to use streaming mode',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=64,
        help='Number of workers to use for preprocessing',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for preprocessing',
    )
    parser.add_argument(
        '--path',
        default='data',
        help='Path to save the preprocessed dataset',
    )
    parser.add_argument(
        '--tokenizer',
        default='fla-hub/transformer-1.3B-100B',
        help='Tokenizer to use',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size for processing"
    )
    args = parser.parse_args()

    logger.info(f'Loading tokenizer {args.tokenizer}')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    logger.info(f'{tokenizer}')
    logger.info(f'Loading dataset {args.dataset} {args.dataset_name} {args.dataset_split}')
    dataset = build_dataset(
        dataset=args.dataset,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        data_dir=args.data_dir,
        data_files=args.data_files,
        data_probs=args.data_probs,
        streaming=args.streaming,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    logger.info(f'Tokenizing and processing the dataset with batch size {args.batch_size}')
    dataset = dataset.map(
        lambda examples: tokenize(examples, tokenizer),
        batched=True,
        batch_size=args.batch_size,
        remove_columns=list(next(iter(dataset)).keys()),
        num_proc=args.num_workers,
        desc="Running tokenizer on dataset"
    )
    logger.info(f'{dataset}')
    logger.info(f'Saving tokenized dataset to {args.path}')
    dataset.save_to_disk(args.path)
