"""Evaluate bi-encoder candidate retrieval."""

from pathlib import Path

from ..model import BiEncoder
from ...common.cli import parse_evaluation_args
from ...common.evaluation import run_evaluation


def main(allowed_splits: tuple[str, ...] = ("validation",), default_split: str = "validation") -> None:
    args = parse_evaluation_args(
        Path(__file__).resolve().parents[1] / "config.json",
        "text_to_actions/runs/bi_encoder", allowed_splits, default_split,
    )
    run_evaluation(args, BiEncoder.from_config)


if __name__ == "__main__":
    main()
