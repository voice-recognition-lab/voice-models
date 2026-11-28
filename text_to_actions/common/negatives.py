"""Mine hard negatives from train requests and sample distinct random candidates."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Sequence

import torch

from .types import ActionCatalog, RankingEncoder, Sample

if TYPE_CHECKING:
    from ..data import Dataset


@torch.inference_mode()
def mine_hard_negatives(encoder: RankingEncoder, dataset: Dataset, samples: Sequence[Sample], count: int) -> dict[str, list[str]]:
    """Select the highest-scoring incorrect actions without tracking gradients."""
    encoder.model.eval()
    encoder.index_actions(dataset.actions)
    predictions = encoder.search(samples, top_k=count + 1)
    result = {}

    for sample, candidates in zip(samples, predictions):
        negative_ids = []

        for candidate in candidates:
            if candidate["action_id"] != sample["action_id"]:
                negative_ids.append(candidate["action_id"])

        result[sample["sample_id"]] = negative_ids[:count]

    return result


def candidate_groups(
    samples: Sequence[Sample], actions: ActionCatalog, hard_negatives: dict[str, list[str]],
    random_count: int, rng: random.Random,
) -> list[list[str]]:
    """Combine the gold action, hard negatives, and distinct random negatives."""
    groups = []
    for sample in samples:
        positive = sample["action_id"]
        if positive is None:
            raise ValueError("Candidate groups require labeled samples")
        hard = hard_negatives[sample["sample_id"]]
        selected = {positive, *hard}
        available = sample.get("available_action_ids", actions)
        pool = [action_id for action_id in available if action_id not in selected]
        random_ids = rng.sample(pool, min(random_count, len(pool)))
        groups.append([positive, *hard, *random_ids])
    return groups
