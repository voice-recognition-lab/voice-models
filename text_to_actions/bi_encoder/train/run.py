"""Fine-tune a shared bi-encoder with explicit hard and random negatives."""

from pathlib import Path

import torch
from torch.nn import functional as F

from ..model import BiEncoder
from ...common.text import action_text, query_text
from ...common.trainer import Trainer, run_training
from ...common.cli import parse_training_args
from ...common.types import ActionCatalog, EncoderConfig, Sample


def contrastive_loss(
    encoder: BiEncoder, samples: list[Sample], actions: ActionCatalog,
    groups: list[list[str]], temperature: float,
) -> torch.Tensor:
    """Compute candidate cross-entropy with gradients through both encoder branches."""
    action_ids = list(dict.fromkeys(action_id for group in groups for action_id in group))
    positions = {action_id: index for index, action_id in enumerate(action_ids)}
    queries = encoder.embed([query_text(sample) for sample in samples])
    candidates = encoder.embed([action_text(actions[action_id]) for action_id in action_ids])
    width = max(len(group) for group in groups)
    indices = torch.tensor([
        [positions[action_id] for action_id in group] + [0] * (width - len(group))
        for group in groups
    ], device=encoder.device, dtype=torch.long)
    lengths = torch.tensor([len(group) for group in groups], device=encoder.device)
    mask = torch.arange(width, device=encoder.device)[None, :] < lengths[:, None]
    logits = (queries @ candidates.T).gather(1, indices) / temperature
    logits = logits.masked_fill(~mask, -torch.inf)
    targets = torch.zeros(len(samples), device=encoder.device, dtype=torch.long)
    return F.cross_entropy(logits, targets)


class BiEncoderTrainer(Trainer):
    """Fine-tune independent command and action embeddings."""

    model_type: str = "bi_encoder"
    selection_metric: str = "recall"
    best_metric_key: str = "best_recall"
    encoder: BiEncoder

    def create_encoder(self, config: EncoderConfig) -> BiEncoder:
        """Build the bi-encoder from its configuration."""
        return BiEncoder.from_config(config)

    def compute_loss(self, samples: list[Sample], groups: list[list[str]]) -> torch.Tensor:
        """Optimize normalized command/action similarity against candidate labels."""
        return contrastive_loss(self.encoder, samples, self.dataset.actions, groups, self.config["temperature"])


def main() -> None:
    args = parse_training_args()
    run_training(args, BiEncoderTrainer, Path(__file__).resolve().parents[1] / "config.json")


if __name__ == "__main__":
    main()
