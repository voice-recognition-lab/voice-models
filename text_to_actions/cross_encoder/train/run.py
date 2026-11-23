"""Train a cross-encoder using negatives selected by a fixed bi-encoder."""

from pathlib import Path

import torch
from torch.nn import functional as F

from ..model import CrossEncoder
from ..retriever import create_retriever
from ...bi_encoder.model import BiEncoder
from ...common.text import action_text, query_text
from ...common.trainer import Trainer, run_training
from ...common.cli import parse_training_args
from ...common.types import EncoderConfig, Sample, TrainingState


class CrossEncoderTrainer(Trainer):
    """Learn candidate scores and select checkpoints by validation accuracy."""

    model_type: str = "cross_encoder"
    selection_metric: str = "accuracy"
    best_metric_key: str = "best_accuracy"
    encoder: CrossEncoder
    negative_encoder: BiEncoder

    def create_encoder(self, config: EncoderConfig) -> CrossEncoder:
        """Build the joint command/action scorer."""
        return CrossEncoder.from_config(config)

    def create_negative_encoder(self, state: TrainingState | None) -> BiEncoder:
        """Load a fixed bi-encoder and retain its resolved settings for resume."""
        saved = state.get("negative_encoder_config") if state is not None else None
        retriever, self.negative_encoder_config = create_retriever(self.encoder_config, saved)
        return retriever

    def compute_loss(self, samples: list[Sample], groups: list[list[str]]) -> torch.Tensor:
        """Apply listwise cross-entropy to each command's candidate scores."""
        queries = []
        actions = []
        for sample, group in zip(samples, groups):
            queries.extend([query_text(sample)] * len(group))
            actions.extend(action_text(self.dataset.actions[key]) for key in group)
        scores = self.encoder.score_pairs(queries, actions)
        rows = scores.split([len(group) for group in groups])
        logits = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=-torch.inf)
        targets = torch.zeros(len(samples), device=self.encoder.device, dtype=torch.long)
        return F.cross_entropy(logits / self.config["temperature"], targets)


def main() -> None:
    args = parse_training_args()
    run_training(args, CrossEncoderTrainer, Path(__file__).resolve().parents[1] / "config.json")


if __name__ == "__main__":
    main()
