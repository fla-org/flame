# -*- coding: utf-8 -*-
"""
Tests for flame.data — specifically, mid-iteration resume of
`OnlineTokenizedIterableDataset` and `ParallelAwareDataLoader`.

The shared invariant under test:
    collect(uninterrupted, N)
      ==
    collect(first half, K) + collect(resume-from-state, N - K)

Tokens yielded on either side of a checkpoint must be identical.
"""
from __future__ import annotations

import pickle
from typing import List

import pytest
import torch
from datasets import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from flame.data import (
    OnlineTokenizedIterableDataset,
    ParallelAwareDataLoader,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class DummyTokenizer:
    """Deterministic stand-in for a HF tokenizer.

    Encodes each character to `ord(c) % vocab_size`. No padding, no BOS/EOS.
    Matches the `tokenizer(list_of_texts, return_attention_mask=False)['input_ids']`
    contract that `OnlineTokenizedIterableDataset.tokenize` relies on.
    """
    vocab_size = 128
    pad_token_id = 0
    bos_token_id = None
    eos_token_id = None

    def __call__(self, texts, return_attention_mask=False, **kwargs):
        return {"input_ids": [[ord(c) % self.vocab_size for c in t] for t in texts]}


def make_dataset(num_samples: int = 2000, num_shards: int = 8, field: str = "text") -> Dataset:
    texts = [
        f"s{i:06d}_" + "abcdefg"[i % 7] * ((i % 17) + 3)
        for i in range(num_samples)
    ]
    ds = Dataset.from_dict({field: texts}).shuffle(seed=123)
    return ds.to_iterable_dataset(num_shards=num_shards)


def collect(dataloader, n: int) -> List[torch.Tensor]:
    it = iter(dataloader)
    return [next(it)["input_ids"].clone() for _ in range(n)]


def _build_dl(ds, rank: int, world_size: int, num_workers: int, seq_len: int = 32):
    dataset = OnlineTokenizedIterableDataset(
        dataset=ds,
        tokenizer=DummyTokenizer(),
        seq_len=seq_len,
        rank=rank,
        world_size=world_size,
    )
    return StatefulDataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=num_workers,
        snapshot_every_n_steps=1,
    )


def _assert_equal_sequences(a: List[torch.Tensor], b: List[torch.Tensor]) -> None:
    assert len(a) == len(b), f"length mismatch: {len(a)} vs {len(b)}"
    mismatches = [i for i, (x, y) in enumerate(zip(a, b)) if not torch.equal(x, y)]
    assert not mismatches, f"{len(mismatches)} mismatches; first at step {mismatches[0]}"


# --------------------------------------------------------------------------- #
# 1. Round-trip resume for OnlineTokenizedIterableDataset via StatefulDataLoader
# --------------------------------------------------------------------------- #

RESUME_AT = 12
TOTAL_STEPS = 28


@pytest.mark.parametrize("num_workers", [0, 2, 4])
@pytest.mark.parametrize(
    "world_size,rank",
    [(1, 0), (2, 0), (2, 1), (4, 3), (8, 5)],
    ids=["w1r0", "w2r0", "w2r1", "w4r3", "w8r5"],
)
def test_resume_matches_uninterrupted(num_workers, world_size, rank):
    """Save state at step K, load into a fresh dataloader, stream K..N.
    Every yielded `input_ids` must byte-equal what an uninterrupted run yields.
    """
    ds = make_dataset(num_shards=max(8, world_size * max(num_workers, 1)))

    dl_ref = _build_dl(ds, rank=rank, world_size=world_size, num_workers=num_workers)
    ref = collect(dl_ref, TOTAL_STEPS)
    del dl_ref

    dl_head = _build_dl(ds, rank=rank, world_size=world_size, num_workers=num_workers)
    it = iter(dl_head)
    head = [next(it)["input_ids"].clone() for _ in range(RESUME_AT)]
    state = dl_head.state_dict()
    del it, dl_head

    dl_tail = _build_dl(ds, rank=rank, world_size=world_size, num_workers=num_workers)
    dl_tail.load_state_dict(state)
    tail = collect(dl_tail, TOTAL_STEPS - RESUME_AT)

    _assert_equal_sequences(ref, head + tail)


# --------------------------------------------------------------------------- #
# 2. State is picklable (DCP serializes non-tensor state as bytes)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("num_workers", [0, 2])
def test_state_dict_is_picklable(num_workers):
    ds = make_dataset(num_shards=max(8, 2 * max(num_workers, 1)))
    dl = _build_dl(ds, rank=0, world_size=2, num_workers=num_workers)
    it = iter(dl)
    for _ in range(5):
        next(it)
    state = dl.state_dict()
    blob = pickle.dumps(state)
    reloaded = pickle.loads(blob)
    assert isinstance(reloaded, dict)


# --------------------------------------------------------------------------- #
# 3. Content-field fallback ('content' instead of 'text')
# --------------------------------------------------------------------------- #

def test_content_field_is_accepted():
    ds = make_dataset(num_shards=4, field="content")
    dl = _build_dl(ds, rank=0, world_size=1, num_workers=0)
    batches = collect(dl, 3)
    assert all(b.dtype == torch.long for b in batches)
    assert all(b.shape == (1, 32) for b in batches)


def test_missing_text_and_content_raises():
    ds = (
        Dataset.from_dict({"payload": ["abcdef" * 10 for _ in range(50)]})
        .to_iterable_dataset(num_shards=2)
    )
    dl = _build_dl(ds, rank=0, world_size=1, num_workers=0)
    with pytest.raises(ValueError, match="No 'text' or 'content' field"):
        next(iter(dl))


# --------------------------------------------------------------------------- #
# 4. ParallelAwareDataLoader: per-rank key isolation
# --------------------------------------------------------------------------- #

def _build_parallel_dl(ds, rank, world_size, num_workers, seq_len=32):
    dataset = OnlineTokenizedIterableDataset(
        dataset=ds,
        tokenizer=DummyTokenizer(),
        seq_len=seq_len,
        rank=rank,
        world_size=world_size,
    )

    def _collate(batch):
        return {"input_ids": torch.stack([b["input_ids"] for b in batch], dim=0)}

    kwargs = dict(
        rank=rank,
        dataset=dataset,
        batch_size=1,
        collate_fn=_collate,
        num_workers=num_workers,
        snapshot_every_n_steps=1,
    )
    # prefetch_factor must be None when num_workers == 0 (torchdata enforces this).
    # ParallelAwareDataLoader's default (2) fails that check, so override explicitly.
    if num_workers == 0:
        kwargs["prefetch_factor"] = None
    return ParallelAwareDataLoader(**kwargs)


def test_parallel_aware_state_dict_keys_by_rank():
    ds = make_dataset(num_shards=8)
    dl = _build_parallel_dl(ds, rank=3, world_size=4, num_workers=0)
    it = iter(dl)
    for _ in range(3):
        next(it)
    state = dl.state_dict()
    assert list(state.keys()) == ["rank_3"], state.keys()


def test_parallel_aware_missing_key_is_noop():
    """If a rank's key is absent, load_state_dict returns silently — this is the
    code's current behavior and the very thing that can mask a desync bug in
    production. Lock it down with a test so future changes that raise instead
    (recommended) fail loudly here and get reviewed."""
    ds = make_dataset(num_shards=8)
    dl = _build_parallel_dl(ds, rank=1, world_size=4, num_workers=0)
    before = collect(dl, 3)
    del dl

    dl2 = _build_parallel_dl(ds, rank=1, world_size=4, num_workers=0)
    # state dict contains someone else's rank — should NOT crash
    dl2.load_state_dict({"rank_0": pickle.dumps({})})
    after = collect(dl2, 3)
    # With no valid state loaded, dl2 starts fresh → matches `before`
    _assert_equal_sequences(before, after)


def test_parallel_aware_resume_per_rank(num_workers=0):
    """Simulate DCP's merge: all ranks save, their state_dicts are union-merged,
    then each rank loads from the merged dict. Every rank must resume its own
    stream correctly — and not cross-contaminate."""
    world_size = 4
    ds = make_dataset(num_shards=16)

    ref_per_rank, merged_state = [], {}
    # Step 1: each rank runs head and snapshots
    head_per_rank = []
    for r in range(world_size):
        dl_ref = _build_parallel_dl(ds, rank=r, world_size=world_size, num_workers=num_workers)
        ref_per_rank.append(collect(dl_ref, 20))
        del dl_ref

        dl = _build_parallel_dl(ds, rank=r, world_size=world_size, num_workers=num_workers)
        it = iter(dl)
        head = [next(it)["input_ids"].clone() for _ in range(10)]
        head_per_rank.append(head)
        merged_state.update(dl.state_dict())
        del it, dl

    assert set(merged_state.keys()) == {f"rank_{i}" for i in range(world_size)}

    # Step 2: each rank loads from merged state and reads the tail
    for r in range(world_size):
        dl = _build_parallel_dl(ds, rank=r, world_size=world_size, num_workers=num_workers)
        dl.load_state_dict(merged_state)
        tail = collect(dl, 10)
        _assert_equal_sequences(ref_per_rank[r], head_per_rank[r] + tail)


# --------------------------------------------------------------------------- #
# 5. Seq lengths are preserved on resume
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seq_len", [16, 32, 128])
def test_resume_preserves_seq_len(seq_len):
    ds = make_dataset(num_shards=4)
    dl = _build_dl(ds, rank=0, world_size=1, num_workers=0, seq_len=seq_len)
    it = iter(dl)
    for _ in range(5):
        batch = next(it)
        assert batch["input_ids"].shape == (1, seq_len)
    state = dl.state_dict()
    del it, dl

    dl2 = _build_dl(ds, rank=0, world_size=1, num_workers=0, seq_len=seq_len)
    dl2.load_state_dict(state)
    for _ in range(5):
        batch = next(iter(dl2))
        assert batch["input_ids"].shape == (1, seq_len)
