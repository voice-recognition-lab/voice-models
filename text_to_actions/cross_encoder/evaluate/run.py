"""Evaluate cross-encoder ranking over the full catalog or a bi-encoder shortlist."""

from pathlib import Path

from ..model import CrossEncoder
from ..retriever import create_retriever
from ...common.cli import parse_evaluation_args
from ...common.evaluation import run_evaluation


def main(allowed_splits: tuple[str, ...] = ("validation",), default_split: str = "validation") -> None:
    args = parse_evaluation_args(
        Path(__file__).resolve().parents[1] / "config.json",
        "text_to_actions/runs/cross_encoder", allowed_splits, default_split,
        supports_hybrid=True,
    )
    run_evaluation(args, CrossEncoder.from_config, retriever_factory=create_retriever)


if __name__ == "__main__":
    main()
