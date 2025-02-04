# -*- coding: utf-8 -*-

import argparse

from datasets import load_dataset


def reshard(
    data: str,
    split: str,
    output: str,
    num_shards: int = 1024,
):
    print(f"Loading dataset {data}...")
    dataset = load_dataset(data, split=split)
    print(f"{dataset}")
    print(f"Saving the dataset with {num_shards} shards to {output}...")
    dataset.save_to_disk(output, num_shards=num_shards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reshard dataset to desired number of shards.")
    parser.add_argument("--data", default="HuggingFaceFW/fineweb-edu", help="Dataset name need to be resharded.")
    parser.add_argument("--split", default="train", help="Split name need to be resharded.")
    parser.add_argument("--output", default="data/fineweb-edu", help="Target directory to store reshared files.")
    parser.add_argument("--num_shards", default=1024, help="Desired number of data shards.")

    args = parser.parse_args()
    reshard(args.data, args.split, args.output, args.num_shards)
