"""Reproducible sample selection for training and evaluation."""

import random
from typing import Optional, TypeVar


T = TypeVar("T")


def select_subset(samples: list[T], limit: Optional[int], seed: int) -> list[T]:
    """Select a reproducible random subset when a sample limit is configured."""
    if limit is None:
        return samples
    return random.Random(seed).sample(samples, min(limit, len(samples)))


