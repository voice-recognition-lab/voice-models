"""Joint encoding and scoring of command/action pairs."""

from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ..common.text import action_text, query_text
from ..common.config import validate_encoder_config
from ..common.types import ActionCatalog, EncoderConfig, PathLike, Predictions, Sample


class CrossEncoder:
    """Score each command together with its candidate action schema."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    revision: Optional[str]
    device: Union[str, torch.device]
    max_length: int
    batch_size: int
    actions: ActionCatalog
    action_texts: dict[str, str]
    score_type: str = "logit"
    pooling: Optional[str] = None

    def __init__(self, checkpoint: PathLike, revision: str | None = "main", max_length: int = 512, batch_size: int = 16,
                 device: str | torch.device = "cpu", model_cache: PathLike = ".venv/huggingface") -> None:
        if max_length < 1 or batch_size < 1:
            raise ValueError("max_length and batch_size must be positive")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, revision=revision, cache_dir=model_cache, num_labels=1,
        ).to(device).eval()
        self.revision = None if Path(checkpoint).is_dir() else getattr(self.model.config, "_commit_hash", None)
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, revision=self.revision or revision, cache_dir=model_cache,
        )
        limit = min(self.model.config.max_position_embeddings, self.tokenizer.model_max_length)
        if max_length > limit:
            raise ValueError(f"max_length exceeds model/tokenizer limit {limit}")
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.actions = {}
        self.action_texts = {}

    @classmethod
    def from_config(cls, config: EncoderConfig) -> "CrossEncoder":
        """Construct the scorer from its JSON configuration."""
        validate_encoder_config(config, model_type="cross_encoder")
        return cls(
            checkpoint=config["checkpoint"],
            revision=config["revision"],
            max_length=config["max_length"],
            batch_size=config["batch_size"],
            device=config["device"],
            model_cache=config["model_cache"],
        )

    def score_pairs(self, queries: Sequence[str], actions: Sequence[str]) -> torch.Tensor:
        """Return one logit per pair while retaining gradients for training."""
        if len(queries) != len(actions):
            raise ValueError("Each command must have a candidate action text")
        outputs = []
        for start in range(0, len(queries), self.batch_size):
            inputs = self.tokenizer(
                queries[start:start + self.batch_size], actions[start:start + self.batch_size],
                padding=True, truncation=False, return_tensors="pt",
            )
            for index, length in enumerate(inputs["attention_mask"].sum(dim=1).tolist(), start):
                if length > self.max_length:
                    raise ValueError(f"Pair {index} has {length} tokens, exceeding max_length={self.max_length}")
            outputs.append(self.model(**inputs.to(self.device)).logits[:, 0])
        return torch.cat(outputs) if outputs else torch.empty(0, device=self.device)

    def index_actions(self, actions: ActionCatalog) -> None:
        """Store schema texts for scoring the current catalog."""
        if not actions:
            raise ValueError("Cannot index an empty catalog")
        self.actions = dict(actions)
        self.action_texts = {key: action_text(action) for key, action in actions.items()}

    def invalidate_index(self) -> None:
        """Clear the catalog so the next evaluation explicitly rebuilds it."""
        self.actions = {}
        self.action_texts = {}

    def search(self, samples: Sequence[Sample], top_k: int) -> Predictions:
        """Rank available candidates, processing command/action pairs in batches."""
        if not self.actions:
            raise ValueError("Call index_actions before search")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        results: Predictions = []
        for sample in samples:
            ids = list(sample.get("available_action_ids", self.actions))
            if len(ids) != len(set(ids)) or set(ids) - self.actions.keys():
                raise ValueError("Candidate IDs must be distinct and present in the catalog")
            if not ids:
                results.append([])
                continue
            with torch.inference_mode():
                scores = self.score_pairs([query_text(sample)] * len(ids), [self.action_texts[key] for key in ids])
            order = torch.argsort(scores, descending=True, stable=True)[:top_k]
            results.append([{"action_id": ids[i], "score": float(scores[i])} for i in order.tolist()])
        return results
