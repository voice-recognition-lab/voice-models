"""Dataset contracts, configuration types, and the shared ranking interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, NotRequired, Protocol, Sequence, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class ContextObject(TypedDict):
    object_id: str
    object_type: str
    name: str
    attributes: NotRequired[dict[str, Any]]


class Context(TypedDict):
    objects: list[ContextObject]
    selected_object_ids: list[str]


class Sample(TypedDict):
    sample_id: str
    text: str
    context: Context
    action_id: str | None
    args: dict[str, Any]
    group_id: str
    split: Literal["train", "validation", "test"]
    available_action_ids: NotRequired[list[str]]


class Action(TypedDict):
    action_id: str
    description: str
    parameters: dict[str, Any]


class Candidate(TypedDict):
    action_id: str
    score: float


class EncoderConfig(TypedDict):
    checkpoint: str
    revision: str | None
    dataset: str
    model_cache: str
    max_length: int
    batch_size: int
    device: str
    num_threads: int
    seed: int
    top_k: list[int]
    embedding_cache: NotRequired[str]
    actions_path: NotRequired[str]
    retriever_config: NotRequired[str]
    retriever_profile: NotRequired[str]
    retrieval_top_k: NotRequired[int]
    retriever: NotRequired["EncoderConfig"]


class TrainingConfig(TypedDict):
    output_dir: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    temperature: float
    warmup_ratio: float
    max_grad_norm: float
    hard_negatives: int
    random_negatives: int
    refresh_hard_negatives: bool
    selection_k: int
    log_every: int
    train_limit: NotRequired[int | None]
    validation_limit: NotRequired[int | None]
    encoder_config: NotRequired[str]


ActionCatalog = dict[str, Action]
Predictions = list[list[Candidate]]
TrainingState = dict[str, Any]
PathLike = str | Path


class RankingEncoder(Protocol):
    """Operations used by the shared training and evaluation lifecycle."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    revision: str | None
    device: str | torch.device
    pooling: str | None
    score_type: str

    def index_actions(self, actions: ActionCatalog) -> None: ...

    def search(self, samples: Sequence[Sample], top_k: int) -> Predictions: ...

    def invalidate_index(self) -> None: ...
