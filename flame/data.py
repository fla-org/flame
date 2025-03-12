# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import datasets
import numpy as np
import torch
from datasets import Dataset, IterableDataset
from datasets.iterable_dataset import ShufflingConfig
from torch.distributed.checkpoint.stateful import Stateful
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer

from torchtitan.tools.logging import logger


class BufferShuffledIterableDataset(IterableDataset):

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        seq_len: int = 2048,
        rank: int = 0,
        world_size: int = 1,
        buffer_size: int = 1024
    ) -> BufferShuffledIterableDataset:

        self.dataset = dataset
        self.tokenizer = tokenizer

        self.data = dataset.shard(world_size, rank)
        self.seq_len = seq_len

        self.rank = rank
        self.world_size = world_size
        self.buffer_size = buffer_size

        if tokenizer.vocab_size < torch.iinfo(torch.int16).max:
            self.dtype = torch.int16
        elif tokenizer.vocab_size < torch.iinfo(torch.int32).max:
            self.dtype = torch.int32
        else:
            self.dtype = torch.int64
        self.states = None
        self.buffer = torch.tensor([], dtype=self.dtype)
        self.tokens = []
        self.rand_id = 0
        self.token_id = 0
        self.rng_state = None
        self._epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._epoch + self.rank)
        if self.rng_state is not None:
            g.set_state(self.rng_state)

        rand_it = self.randint(0, self.buffer_size, g=g)
        if self.states is not None:
            self.data.load_state_dict(self.states)

        # max number of tokens allowed in the chunk buffer
        n_tokens = self.buffer_size * self.seq_len

        while True:
            for sample in self.tokenize(self.data):
                # keep appending the samples to the token buffer
                self.tokens += sample
                # if the token buffer is full, start sampling
                # NOTE: we first convert the token ids to a tensor of shape [n_chunks, seq_len] for efficiency
                if len(self.buffer) == 0 and len(self.tokens) >= n_tokens:
                    self.buffer = torch.tensor(self.tokens[:n_tokens], dtype=self.dtype).view(self.buffer_size, -1)
                    self.tokens = self.tokens[n_tokens:]
                if len(self.buffer) == self.buffer_size:
                    yield from self.sample(rand_it)

            n_chunks = len(self.tokens) // self.seq_len
            # handle the left tokens in the buffer
            if n_chunks > 0:
                n_tokens = n_chunks * self.seq_len
                indices = torch.randperm(n_chunks, generator=g).tolist()
                self.buffer = torch.tensor(self.tokens[:n_tokens], dtype=torch.long).view(n_chunks, -1)
                self.tokens = self.tokens[n_tokens:]
                for i in indices:
                    yield {'input_ids': self.buffer[i]}

    def tokenize(self, data, batch_size: int = 64):
        texts, states = [], []
        for sample in data:
            texts.append(sample['text'])
            states.append(self.data.state_dict())
            if len(texts) == batch_size:
                for s, tokenized in zip(states, self.tokenizer(texts, return_attention_mask=False)['input_ids']):
                    self.states = s
                    yield tokenized
                texts, states = [], []
        if len(texts) > 0:
            for s, tokenized in zip(states, self.tokenizer(texts, return_attention_mask=False)['input_ids']):
                self.states = s
                yield tokenized

    def sample(self, indices):
        n_tokens = (len(self.tokens) // self.seq_len) * self.seq_len
        while self.token_id < n_tokens:
            i = next(indices)
            start, end = self.token_id, self.token_id + self.seq_len
            self.token_id += self.seq_len
            yield {'input_ids': self.buffer[i].to(torch.long)}
            self.buffer[i] = torch.tensor(self.tokens[start:end], dtype=self.dtype)
        self.token_id = 0
        self.tokens = self.tokens[n_tokens:]

    def randint(
        self,
        low: int,
        high: int,
        buffer_size: int = 1024,
        g: torch.Generator = torch.Generator()
    ) -> Iterable[int]:
        indices = torch.empty(buffer_size, dtype=torch.long)
        while True:
            # record the generator states before sampling
            self.rng_state = g.get_state()
            indices = torch.randint(low, high, (buffer_size,), out=indices, generator=g)
            for i in indices[self.rand_id:].tolist():
                self.rand_id += 1
                yield i
            self.rand_id = 0

    def set_epoch(self, epoch):
        self._epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def state_dict(self):
        return {
            'states': self.states,
            'buffer': self.buffer.clone(),
            'tokens': deepcopy(self.tokens),
            'rand_id': self.rand_id,
            'token_id': self.token_id,
            'rng_state': self.rng_state,
            'epoch': self._epoch
        }

    def load_state_dict(self, state_dict):
        self.states = state_dict['states']
        self.buffer = state_dict['buffer'].clone()
        self.tokens = deepcopy(state_dict['tokens'])
        self.rand_id = state_dict['rand_id']
        self.token_id = state_dict['token_id']
        self.rng_state = state_dict['rng_state'].clone() if state_dict['rng_state'] is not None else None
        self._epoch = state_dict['epoch']


class OnlineTokenizedIterableDataset(IterableDataset):

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        seq_len: int = 2048,
        rank: int = 0,
        world_size: int = 1
    ) -> OnlineTokenizedIterableDataset:

        self.dataset = dataset
        self.tokenizer = tokenizer

        self.data = dataset.shard(world_size, rank)
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

        self.states = None
        self.tokens = []

    def __iter__(self):
        if self.states is not None:
            self.data.load_state_dict(self.states)

        while True:
            for sample in self.tokenize(self.data):
                # keep appending the samples to the token buffer
                self.tokens += sample

                while len(self.tokens) >= self.seq_len:
                    input_ids = torch.tensor(self.tokens[:self.seq_len], dtype=torch.long)
                    self.tokens = self.tokens[self.seq_len:]
                    yield {'input_ids': input_ids}

    def tokenize(self, data, buffer_size: int = 64):
        buffer, states = [], []
        for sample in data:
            if sample.get('text', None) is not None:
                buffer.append(sample['text'])
            elif sample.get('content', None) is not None:
                buffer.append(sample['content'])
            else:
                raise ValueError(f"No 'text' or 'content' field found in sample:\n{sample}")
            states.append(self.data.state_dict())
            if len(buffer) == buffer_size:
                for s, tokenized in zip(states, self.tokenizer(buffer, return_attention_mask=False)['input_ids']):
                    self.states = s
                    yield tokenized
                buffer, states = [], []
        if len(buffer) > 0:
            for s, tokenized in zip(states, self.tokenizer(buffer, return_attention_mask=False)['input_ids']):
                self.states = s
                yield tokenized

    def state_dict(self):
        return {
            'states': self.states,
            'tokens': deepcopy(self.tokens)
        }

    def load_state_dict(self, state_dict):
        self.states = state_dict['states']
        self.tokens = deepcopy(state_dict['tokens'])


class BufferShuffledExamplesIterable(datasets.iterable_dataset.BufferShuffledExamplesIterable):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_state_dict(self) -> dict:
        self._state_dict = self.ex_iterable._init_state_dict()
        self._state_dict['mem_buffer'] = ([],)
        self._state_dict['bit_generator_state'] = self.generator.bit_generator.state
        self._state_dict['bit_generator_index_offset'] = 0
        self._state_dict['bit_generator_index_offset_shuffle'] = 0
        return self._state_dict

    def __iter__(self):
        buffer_size = self.buffer_size
        rng = deepcopy(self.generator)
        # this is the shuffle buffer that we keep in memory
        mem_buffer = self._state_dict['mem_buffer'][0]
        # this is an infinite iterator that randomly samples the index of the source to pick examples from
        index_offset = self._state_dict["bit_generator_index_offset"] if self._state_dict else 0
        if self._state_dict:
            rng.bit_generator.state = self._state_dict["bit_generator_state"]
        indices_iterator = self._iter_random_indices(rng, buffer_size, random_batch_size=buffer_size)
        # skip already consumed ones
        for _ in range(index_offset):
            i = next(indices_iterator)

        for x in self.ex_iterable:
            if len(mem_buffer) < buffer_size:  # if the buffer is not full, keep filling the buffer
                mem_buffer.append(x)
            else:  # otherwise, pick an example from it
                i = next(indices_iterator)
                index_offset = (index_offset + 1) % buffer_size
                if self._state_dict:
                    self._state_dict["bit_generator_index_offset"] = index_offset
                    if index_offset == 0:
                        self._state_dict["bit_generator_state"] = rng.bit_generator.state
                selected = mem_buffer[i]
                mem_buffer[i] = x  # replace the picked example by a new one
                yield selected

        index_offset = self._state_dict["bit_generator_index_offset_shuffle"] if self._state_dict else 0
        if self._state_dict:
            rng.bit_generator.state = self._state_dict["bit_generator_state"]

        # when we run out of examples, we shuffle the remaining examples in the buffer and yield them
        for i in rng.permutation(len(mem_buffer))[index_offset:].tolist():
            index_offset = index_offset + 1
            if self._state_dict:
                self._state_dict["bit_generator_index_offset_shuffle"] = index_offset
            yield mem_buffer[i]

    def shuffle_data_sources(self, generator: np.random.Generator) -> BufferShuffledExamplesIterable:
        """Shuffle the wrapped examples iterable as well as the shuffling buffer."""
        return BufferShuffledExamplesIterable(
            self.ex_iterable.shuffle_data_sources(generator), buffer_size=self.buffer_size, generator=generator
        )

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> BufferShuffledExamplesIterable:
        """Keep only the requested shard."""
        return BufferShuffledExamplesIterable(
            self.ex_iterable.shard_data_sources(num_shards, index, contiguous=contiguous),
            buffer_size=self.buffer_size,
            generator=self.generator,
        )

    def load_state_dict(self, state_dict: dict) -> dict:
        def _inner_load_state_dict(state, new_state):
            if new_state is not None and isinstance(state, dict):
                for key in new_state:
                    state[key] = _inner_load_state_dict(state[key], new_state[key])
                return state
            elif new_state is not None and isinstance(state, list):
                for i in range(len(state)):
                    state[i] = _inner_load_state_dict(state[i], new_state[i])
                return state
            return new_state

        return _inner_load_state_dict(self._state_dict, state_dict)


def shuffle(
    dataset: IterableDataset,
    seed: int = 42,
    generator: np.random.Generator = None,
    buffer_size: int = 1024,
):
    generator = np.random.default_rng(seed) if generator is None else deepcopy(generator)
    return IterableDataset(
        ex_iterable=BufferShuffledExamplesIterable(
            dataset._ex_iterable, buffer_size=buffer_size, generator=generator
        ),
        info=dataset._info.copy(),
        split=dataset._split,
        formatting=dataset._formatting,
        shuffling=ShufflingConfig(generator=generator, _original_seed=seed),
        distributed=copy.deepcopy(dataset._distributed),
        token_per_repo_id=dataset._token_per_repo_id
    )


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        context_len (`int`):
            The maximum allowed length for each sequence.
            When `varlen=True`, longer sequences will be split into multiple chunks of at most this length.
        varlen (`bool`):
            Whether to return sequences with variable lengths.
            If `True`, the `cu_seqlens` indicating the start and end of each sequence will be returned.
            For example, if the sequence lengths are `[4, 8, 12]`,
            the returned `input_ids` will be a flattened tensor of shape `[1, 24]`, with `cu_seqlens` being `[0, 4, 12, 24]`.

    Returns:
        A dictionary with the following keys:
        - `input_ids`: A tensor of shape `[batch_size, seq_len]` containing the input ids.
        - `labels`: A tensor of shape `[batch_size, seq_len]` containing the labels.
        - `cu_seqlens`: A tensor containing the start and end positions of each sequence.

    NOTE: When in variable length mode, the `batch_size` must be 1.
    """

    tokenizer: PreTrainedTokenizer
    context_len: Optional[int] = None
    varlen: bool = False

    def __call__(
        self,
        examples: List[Union[List[int], Dict[str, Any]]]
    ) -> Dict[str, Any]:
        if not isinstance(examples[0], Dict):
            examples = [{'input_ids': example} for example in examples]

        def tensorize(example: Dict[str, Any]) -> Dict[str, Any]:
            tensorized = {}
            for key in ['input_ids', 'cu_seqlens']:
                if key not in example:
                    continue
                if isinstance(example[key], List):
                    tensorized[key] = torch.tensor(example[key], dtype=torch.long)
                elif isinstance(example[key], np.ndarray):
                    tensorized[key] = torch.from_numpy(example[key])
                else:
                    tensorized[key] = example[key]
            return tensorized

        examples = list(map(tensorize, examples))

        if not self.varlen:
            length_of_first = examples[0]['input_ids'].size(0)
            # Check if padding is necessary.
            if all(example['input_ids'].size(0) == length_of_first for example in examples):
                batch = {
                    'input_ids': torch.stack([example['input_ids'] for example in examples], dim=0),
                }
            else:
                # If yes, check if we have a `pad_token`.
                if self.tokenizer._pad_token is None:
                    raise ValueError(
                        f"You are attempting to pad samples but the tokenizer you are using "
                        f"({self.tokenizer.__class__.__name__}) does not have a pad token."
                    )
                batch = self.tokenizer.pad(examples, return_tensors='pt', return_attention_mask=False)
        else:
            if len(examples) > 1:
                raise ValueError("The batch size must be 1 for inputs with variable lengths.")
            batch = {
                'input_ids': torch.cat([example['input_ids'] for example in examples], dim=0).unsqueeze(0)
            }
            if 'cu_seqlens' in examples[0]:
                batch['cu_seqlens'] = torch.cat([example['cu_seqlens'] for example in examples], dim=0).unsqueeze(0)
            else:
                # determine boundaries by bos/eos positions
                if self.tokenizer.add_bos_token:
                    cu_seqlens = []
                    if batch['input_ids'][0, 0] != self.tokenizer.bos_token_id:
                        cu_seqlens.append(torch.tensor([0]))
                    cu_seqlens.append(torch.where(batch['input_ids'].eq(self.tokenizer.bos_token_id))[1])
                    cu_seqlens.append(torch.tensor([len(batch['input_ids'][0])]))
                    batch['cu_seqlens'] = torch.cat(cu_seqlens, dim=0).to(dtype=torch.int32)
                elif self.tokenizer.add_eos_token:
                    cu_seqlens = [torch.tensor([0])]
                    cu_seqlens.append(torch.where(batch['input_ids'].eq(self.tokenizer.eos_token_id))[1] + 1)
                    if batch['input_ids'][0, -1] != self.tokenizer.eos_token_id:
                        cu_seqlens.append(torch.tensor([len(batch['input_ids'][0])]))
                    batch['cu_seqlens'] = torch.cat(cu_seqlens, dim=0).to(dtype=torch.int32)
                else:
                    raise ValueError("You must allow the tokenizer to add either a bos or eos token as separators.")
            if self.context_len is not None:
                bos = batch['cu_seqlens'][:-1].tolist()
                eos = batch['cu_seqlens'][1:].tolist()
                batch['cu_seqlens'] = torch.cat(
                    [torch.arange(i, j, self.context_len) for i, j in zip(bos, eos)] +
                    [torch.tensor([len(batch['input_ids'][0])])]
                ).to(dtype=torch.int32)

        labels = batch['input_ids'].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


class ParallelAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """

    def __init__(
        self,
        rank: int,
        dataset: IterableDataset,
        batch_size: int,
        collate_fn: Callable,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        snapshot_every_n_steps: Optional[int] = 1
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            snapshot_every_n_steps=snapshot_every_n_steps
        )
        self.rank = rank

    def state_dict(self) -> Dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions
        return {f"rank_{self.rank}": pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # State being empty is valid
        if not state_dict:
            return

        if f"rank_{self.rank}" not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self.rank}, expected key "
                f"rank_{self.rank}"
            )
            return
        super().load_state_dict(pickle.loads(state_dict[f"rank_{self.rank}"]))


def build_dataloader(
    dataset: IterableDataset,
    tokenizer: PreTrainedTokenizer,
    rank: int,
    world_size: int,
    batch_size: int,
    seq_len: int,
    context_len: Optional[int] = None,
    varlen: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    snapshot_every_n_steps: Optional[int] = 1
):
    dataset = OnlineTokenizedIterableDataset(
        dataset=dataset,
        tokenizer=tokenizer,
        seq_len=seq_len,
        rank=rank,
        world_size=world_size
    )
    return ParallelAwareDataLoader(
        rank=rank,
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, context_len=context_len, varlen=varlen),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        snapshot_every_n_steps=snapshot_every_n_steps
    )
