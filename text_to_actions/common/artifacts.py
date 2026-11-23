"""Run metadata and prediction files shared by training and evaluation."""

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from .types import Predictions, Sample

import torch
import transformers


def runtime_metadata() -> dict[str, Any]:
    """Record the repository revision and installed model library versions."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, check=False,
    )
    return {
        "git_commit": commit.stdout.strip() or None,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def write_predictions(
    path: Path, samples: Sequence[Sample], predictions: Predictions, retrieved: Predictions | None = None,
) -> None:
    """Write ranked candidates, optionally including the initial retrieval shortlist."""
    with path.open("w", encoding="utf-8") as output:
        for index, (sample, candidates) in enumerate(zip(samples, predictions)):
            record = {
                "sample_id": sample["sample_id"], "gold_action_id": sample["action_id"],
                "candidates": candidates,
            }
            if retrieved is not None:
                record["retrieved_candidates"] = retrieved[index]
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
