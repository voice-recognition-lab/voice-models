"""Retrieval accuracy and recall at configurable candidate counts."""
from typing import Any, Sequence

from .types import Predictions, Sample


def retrieval_metrics(samples: Sequence[Sample], predictions: Predictions, top_k: Sequence[int]) -> dict[str, Any]:
    """Calculate top-1 accuracy and recall over examples with gold action IDs."""
    if len(samples) != len(predictions):
        raise ValueError("Each sample must have a prediction")
    labeled = [
        (sample, candidates) for sample, candidates in zip(samples, predictions)
        if sample["action_id"] is not None
    ]
    recall = {
        str(k): sum(
            sample["action_id"] in [candidate["action_id"] for candidate in candidates[:k]]
            for sample, candidates in labeled
        ) / len(labeled) if labeled else None
        for k in sorted(set([1, *top_k]))
    }
    return {
        "samples": len(samples), "labeled_samples": len(labeled),
        "unlabeled_samples": len(samples) - len(labeled),
        "accuracy": recall["1"], "recall_at_k": recall,
    }


