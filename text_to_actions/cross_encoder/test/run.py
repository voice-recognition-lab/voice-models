"""Final evaluation of the selected cross-encoder on the test split."""

from pathlib import Path

from ..model import CrossEncoder
from ..retriever import create_retriever
from ...common.cli import parse_evaluation_args
from ...common.evaluation import run_evaluation


def main() -> None:
    args = parse_evaluation_args(
        Path(__file__).resolve().parents[1] / "config.json",
        "text_to_actions/runs/cross_encoder", allowed_splits=("test",), default_split="test",
        supports_hybrid=True,
    )
    run_evaluation(args, CrossEncoder.from_config, retriever_factory=create_retriever)


if __name__ == "__main__":
    main()
