"""Independent encoding of commands and action schemas for candidate retrieval."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, Sequence, Union

import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from ..common.hashing import fingerprint
from ..common.text import action_text, query_text
from ..common.config import validate_encoder_config
from ..common.types import ActionCatalog, EncoderConfig, PathLike, Predictions, Sample


class BiEncoder:
    """Shared text encoder and action catalog index."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    checkpoint: Union[str, Path]
    revision: Optional[str]

    max_length: int
    batch_size: int
    device: Union[str, torch.device]

    embedding_cache: Path
    cache_embeddings: bool
    action_ids: list[str]
    action_positions: dict[str, int]
    action_embeddings: Optional[torch.Tensor]
    score_type: str = "cosine"
    pooling: Optional[str] = "normalized_cls"

    def __init__(
        self, checkpoint: PathLike, revision: str | None = "main", max_length: int = 512, batch_size: int = 32,
        device: str | torch.device = "cpu", model_cache: PathLike = ".venv/huggingface",
        embedding_cache: PathLike = ".venv/text_to_actions/embeddings",
    ) -> None:
        if max_length < 1 or batch_size < 1:
            raise ValueError("max_length and batch_size must be positive")
        self.model = AutoModel.from_pretrained(
            checkpoint, revision=revision, cache_dir=model_cache,
        ).to(device).eval()
        self.revision = None if Path(checkpoint).is_dir() else getattr(self.model.config, "_commit_hash", None)
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, revision=self.revision or revision, cache_dir=model_cache,
        )
        limit = min(self.model.config.max_position_embeddings, self.tokenizer.model_max_length)
        if max_length > limit:
            raise ValueError(f"max_length={max_length} exceeds model/tokenizer limit {limit}")
        self.checkpoint = checkpoint
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.embedding_cache = Path(embedding_cache)
        self.action_ids = []
        self.action_positions = {}
        self.action_embeddings = None
        self.cache_embeddings = self.revision is not None

    @classmethod
    def from_config(cls, config: EncoderConfig) -> "BiEncoder":
        """Construct the encoder from the shared JSON configuration."""
        validate_encoder_config(config, model_type="bi_encoder")
        return cls(
            checkpoint=config["checkpoint"],
            revision=config["revision"],
            max_length=config["max_length"],
            batch_size=config["batch_size"],
            device=config["device"],
            model_cache=config["model_cache"],
            embedding_cache=config["embedding_cache"],
        )

    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        """Encode one batch with gradients available for both training branches."""
        inputs = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        for index, length in enumerate(inputs["attention_mask"].sum(dim=1).tolist()):
            if length > self.max_length:
                raise ValueError(
                    f"Input {index} has {length} tokens, exceeding max_length={self.max_length}"
                )
        output = self.model(**inputs.to(self.device)).last_hidden_state[:, 0, :]
        return F.normalize(output, p=2, dim=1)

    def invalidate_index(self) -> None:
        """Call after weight updates so inference cannot reuse pretrained catalog vectors."""
        self.action_embeddings = None
        self.cache_embeddings = False

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Return L2-normalized CLS embeddings on the configured device."""
        batches = []
        for start in range(0, len(texts), self.batch_size):
            with torch.inference_mode():
                batches.append(self.embed(texts[start:start + self.batch_size]))
        if not batches:
            return torch.empty((0, self.model.config.hidden_size), device=self.device)
        return torch.cat(batches)

    def index_actions(self, actions: ActionCatalog) -> None:
        """Cache catalog embeddings by contents, resolved model revision and encoding settings."""
        action_ids = list(actions)
        texts = [action_text(actions[action_id]) for action_id in action_ids]
        if not texts:
            raise ValueError("Cannot index an empty catalog")
        signature = {
            "checkpoint": self.checkpoint, "revision": self.revision,
            "pooling": "normalized_cls", "max_length": self.max_length,
            "serialization_version": 1, "texts": texts,
        }
        digest = fingerprint(signature, ensure_ascii=True)
        cache_path = self.embedding_cache / f"{digest}.pt"
        # Local checkpoints without a resolved revision are encoded afresh.
        if self.cache_embeddings and cache_path.exists():
            embeddings = torch.load(cache_path, map_location=self.device, weights_only=True)
            expected_shape = (len(texts), self.model.config.hidden_size)
            if tuple(embeddings.shape) != expected_shape or not torch.isfinite(embeddings).all():
                raise ValueError(f"Invalid embedding cache {cache_path}; remove it and retry")
        else:
            embeddings = self.encode(texts)
            if self.cache_embeddings:
                self.embedding_cache.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(dir=self.embedding_cache, delete=False) as output:
                    temporary = Path(output.name)
                try:
                    torch.save(embeddings, temporary)
                    temporary.replace(cache_path)
                finally:
                    temporary.unlink(missing_ok=True)
        self.action_ids = action_ids
        self.action_positions = {action_id: index for index, action_id in enumerate(action_ids)}
        self.action_embeddings = embeddings

    def search(self, samples: Sequence[Sample], top_k: int) -> Predictions:
        """Return scored candidates for each request, respecting its available_action_ids."""
        if self.action_embeddings is None:
            raise ValueError("Call index_actions before search")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        results: Predictions = []
        for start in range(0, len(samples), self.batch_size):
            batch = samples[start:start + self.batch_size]
            embeddings = self.encode([query_text(sample) for sample in batch])
            scores = embeddings @ self.action_embeddings.T
            for sample, row in zip(batch, scores):
                ids = list(sample.get("available_action_ids", self.action_ids))
                if len(ids) != len(set(ids)):
                    raise ValueError("Duplicate candidate IDs")
                unknown = set(ids) - self.action_positions.keys()
                if unknown:
                    raise ValueError(f"Unknown candidate IDs: {sorted(unknown)}")
                positions = [self.action_positions[action_id] for action_id in ids]
                candidate_scores = row[positions]
                order = torch.argsort(candidate_scores, descending=True, stable=True)[:top_k]
                results.append([
                    {"action_id": ids[index], "score": float(candidate_scores[index])}
                    for index in order.tolist()
                ])
        return results
