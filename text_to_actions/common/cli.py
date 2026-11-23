"""CLI parsing and explicit options for training and evaluation entry points."""

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingOptions:
    config: Path | None
    profile: str | None
    resume: Path | None
    actions: Path | None


@dataclass
class EvaluationOptions:
    config: Path
    profile: str | None
    actions: Path | None
    checkpoint: Path | None
    limit: int | None
    output: Path
    split: list[str]
    mode: str


def parse_training_args() -> TrainingOptions:
    """Parse CLI options without loading configs or constructing a trainer."""
    parser = argparse.ArgumentParser(description="Train or resume an action selection model.")
    parser.add_argument("--config", type=Path, help="Model config with named encoder/training profiles")
    parser.add_argument("--profile", help="Profile in --config (default: cpu_full for a new run)")
    parser.add_argument("--resume", type=Path, help="Run directory, last.json, or latest epoch checkpoint")
    parser.add_argument("--actions", type=Path, help="Override actions_path; resume requires unchanged catalog contents")
    return TrainingOptions(**vars(parser.parse_args()))


def parse_evaluation_args(
    default_config: Path, default_output: str, allowed_splits: tuple[str, ...] = ("validation",),
    default_split: str = "validation", supports_hybrid: bool = False,
) -> EvaluationOptions:
    """Parse evaluation options and reject invalid sample limits."""
    parser = argparse.ArgumentParser(description="Evaluate action ranking over a catalog or shortlist.")
    parser.add_argument("--config", type=Path, default=default_config, help="Model config with profiles, or a saved evaluation_config.json")
    parser.add_argument("--profile", help="Profile in --config (default: cpu_full); omit for saved checkpoint configs")
    parser.add_argument("--actions", type=Path, help="Override actions_path with an alternative action catalog")
    parser.add_argument("--split", choices=allowed_splits, nargs="+", default=[default_split])
    parser.add_argument("--checkpoint", type=Path, help="Saved model and tokenizer directory")
    parser.add_argument("--limit", type=int, help="Evaluate the first N records per split for a smoke check")
    parser.add_argument("--output", type=Path, default=Path(default_output))
    parser.set_defaults(mode="all")
    if supports_hybrid:
        parser.add_argument("--mode", choices=("all", "hybrid"), help="all: score the full available catalog; hybrid: rerank bi-encoder top-k")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return EvaluationOptions(**vars(args))
