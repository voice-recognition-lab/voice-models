"""Shared epoch training, validation, logging, and checkpoint lifecycle."""

from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Optional, TextIO

import torch
from filelock import FileLock
from transformers import get_linear_schedule_with_warmup

from .checkpoint import (
    capture_rng_state, load_training_state, resolve_checkpoint, restore_rng_state,
    save_epoch_checkpoint, trim_training_log, validate_resume_config,
)
from .config import load_training_config, validate_config
from .cli import TrainingOptions
from .artifacts import runtime_metadata
from .types import EncoderConfig, RankingEncoder, Sample, TrainingConfig, TrainingState
from .sampling import select_subset
from .io import write_json
from .hashing import file_sha256
from ..data import Dataset, load_dataset, resolve_dataset_paths
from .metrics import retrieval_metrics
from .negatives import candidate_groups, mine_hard_negatives


class Trainer(ABC):
    """Own the model, optimization state, and training run artifacts."""

    config: TrainingConfig
    encoder_config: EncoderConfig

    dataset: Dataset
    training: list[Sample]
    validation: list[Sample]
    skipped_train_samples: int

    model_type: str
    selection_metric: str
    best_metric_key: str

    encoder: RankingEncoder
    negative_encoder: RankingEncoder
    negative_encoder_config: Optional[EncoderConfig]
    optimizer: torch.optim.AdamW
    scheduler: torch.optim.lr_scheduler.LambdaLR

    run_dir: Path
    history: dict[str, Any]
    hard_negatives: Optional[dict[str, list[str]]]
    total_steps: int
    global_step: int
    completed_epoch: int
    dataset_hashes: dict[str, str]
    dataset_paths: dict[str, Path]
    source_revision: Optional[str]

    def __init__(
        self, config: TrainingConfig | None, encoder_config: EncoderConfig | None,
        resume_checkpoint: Path | None = None,
    ) -> None:
        state = load_training_state(resume_checkpoint) if resume_checkpoint is not None else None
        if state is not None:
            config = state["training_config"] if config is None else config
            encoder_config = state["encoder_config"] if encoder_config is None else encoder_config
        if state is not None and state.get("model_type", "bi_encoder") != self.model_type:
            raise ValueError("Checkpoint belongs to a different model type")
        if config is None or encoder_config is None:
            raise ValueError("Training requires configs or a complete resume checkpoint")
        self.config = config
        self.encoder_config = encoder_config
        validate_config(config, encoder_config, self.model_type)
        random.seed(encoder_config["seed"])
        torch.manual_seed(encoder_config["seed"])
        torch.set_num_threads(encoder_config["num_threads"])

        self.load_data()
        if state is not None:
            validate_resume_config(state, config, encoder_config, self.dataset_hashes)
        self.setup_models(resume_checkpoint, state)
        self.setup_optimization()
        self.setup_run(resume_checkpoint)
        if state is not None:
            self.restore_state(state)

    def load_data(self) -> None:
        """Load train/validation subsets and record the selected data hashes."""
        self.dataset_paths = resolve_dataset_paths(self.encoder_config["dataset"], self.encoder_config.get("actions_path"))
        self.dataset = load_dataset(self.encoder_config["dataset"], self.encoder_config.get("actions_path"))
        train_samples = self.dataset.samples_for_split("train")
        self.training = [
            sample for sample in train_samples
            if sample["action_id"] is not None
            and len(sample.get("available_action_ids", self.dataset.actions)) > 1
        ]
        self.skipped_train_samples = len(train_samples) - len(self.training)
        self.training = select_subset(self.training, self.config.get("train_limit"), self.encoder_config["seed"])
        self.validation = select_subset(
            self.dataset.samples_for_split("validation"),
            self.config.get("validation_limit"), self.encoder_config["seed"],
        )
        if not self.training or not any(sample["action_id"] is not None for sample in self.validation):
            raise ValueError("Training requires train examples with negatives and labeled validation examples")

        self.dataset_hashes = {
            name: file_sha256(path) for name, path in self.dataset_paths.items()
        }

    def setup_models(self, resume_checkpoint: Path | None, state: TrainingState | None) -> None:
        """Construct the trainable encoder and the negative mining encoder."""
        model_config = self.encoder_config.copy()
        if resume_checkpoint is not None:
            model_config.update(checkpoint=str(resume_checkpoint), revision=None)
        self.encoder = self.create_encoder(model_config)
        self.negative_encoder_config = None
        self.negative_encoder = self.create_negative_encoder(state)

    def setup_optimization(self) -> None:
        """Initialize the optimizer and the complete learning rate schedule."""
        self.optimizer = torch.optim.AdamW(
            self.encoder.model.parameters(),
            lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"],
        )
        steps_per_epoch = math.ceil(len(self.training) / self.config["batch_size"])
        self.total_steps = steps_per_epoch * self.config["epochs"]
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.total_steps * self.config["warmup_ratio"]),
            num_training_steps=self.total_steps,
        )

    def setup_run(self, resume_checkpoint: Path | None) -> None:
        """Initialize run paths, history, and epoch counters."""
        if resume_checkpoint is None:
            self.run_dir = Path(self.config["output_dir"]) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            self.run_dir.mkdir(parents=True)
        else:
            self.run_dir = Path(resume_checkpoint).parent
        self.history = {
            "epochs": [], "selection_k": self.config["selection_k"],
            "best_checkpoint": None, self.best_metric_key: None,
        }
        self.hard_negatives = None
        self.global_step = 0
        self.completed_epoch = 0
        self.source_revision = self.encoder.revision

    @abstractmethod
    def create_encoder(self, config: EncoderConfig) -> RankingEncoder:
        """Construct the model used for training and validation."""

    @abstractmethod
    def compute_loss(self, samples: list[Sample], groups: list[list[str]]) -> torch.Tensor:
        """Calculate the model-specific candidate loss."""

    def create_negative_encoder(self, state: TrainingState | None) -> RankingEncoder:
        """Use the trainable encoder to mine negatives by default."""
        return self.encoder

    def save_metadata(self) -> None:
        """Record configurations, dataset hashes, and software versions for this run."""
        hashes = {
            str(self.dataset_paths[name]): digest for name, digest in self.dataset_hashes.items()
        }
        metadata = {
            "model_type": self.model_type, "training_config": self.config,
            "encoder_config": self.encoder_config,
            "resolved_revision": self.source_revision,
            "negative_encoder_config": self.negative_encoder_config,
            "dataset_sha256": hashes,
            **runtime_metadata(),
            "train_samples": len(self.training),
            "skipped_train_samples": self.skipped_train_samples,
            "validation_samples": len(self.validation),
            "catalog_actions": len(self.dataset.actions),
            "total_steps": self.total_steps,
        }
        write_json(self.run_dir / "config.json", metadata)

    def refresh_negatives(self, epoch: int) -> None:
        """Mine train candidates initially and whenever epoch refresh is enabled."""
        if self.hard_negatives is None or self.config["refresh_hard_negatives"]:
            print(f"Epoch {epoch}: mining train hard negatives", flush=True)
            self.hard_negatives = mine_hard_negatives(
                self.negative_encoder, self.dataset, self.training, self.config["hard_negatives"],
            )

    def train_step(self, samples: list[Sample], rng: random.Random) -> float:
        """Process one batch and update the encoder weights and learning rate."""
        groups = candidate_groups(
            samples, self.dataset.actions, self.hard_negatives,
            self.config["random_negatives"], rng,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.compute_loss(samples, groups)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at step {self.global_step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.encoder.model.parameters(), self.config["max_grad_norm"], error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.scheduler.step()
        self.encoder.invalidate_index()
        self.global_step += 1
        return loss.item()

    def log_step(self, log: TextIO, epoch: int, loss: float) -> None:
        """Write batch loss and learning rate to the run log and console."""
        record = {
            "epoch": epoch, "step": self.global_step, "loss": loss,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    def train_epoch(self, epoch: int, log: TextIO) -> float:
        """Shuffle train commands, process their batches, and return mean loss."""
        self.refresh_negatives(epoch)
        rng = random.Random(self.encoder_config["seed"] + epoch)
        shuffled = list(self.training)
        rng.shuffle(shuffled)
        self.encoder.model.train()
        total_loss = 0.0
        batch_size = self.config["batch_size"]
        for start in range(0, len(shuffled), batch_size):
            samples = shuffled[start:start + batch_size]
            loss = self.train_step(samples, rng)
            total_loss += loss * len(samples)
            last_batch = start + len(samples) == len(shuffled)
            if self.global_step % self.config["log_every"] == 0 or last_batch:
                self.log_step(log, epoch, loss)
        return total_loss / len(self.training)

    def evaluate(self) -> dict[str, Any]:
        """Rebuild the action index and measure retrieval on validation."""
        self.encoder.model.eval()
        self.encoder.index_actions(self.dataset.actions)
        predictions = self.encoder.search(self.validation, max(self.encoder_config["top_k"]))
        return retrieval_metrics(self.validation, predictions, self.encoder_config["top_k"])

    def restore_state(self, state: TrainingState) -> None:
        """Restore learning state and rebase checkpoint references to this run directory."""
        if self.total_steps != state["total_steps"]:
            raise ValueError("Training schedule length differs from the checkpoint")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self.completed_epoch = state["completed_epoch"]
        self.hard_negatives = state["hard_negatives"]
        self.source_revision = state["source_revision"]
        self.history = deepcopy(state["history"])
        for key in ("best_checkpoint", "best_evaluation_config", "last_checkpoint"):
            if self.history.get(key):
                self.history[key] = str(self.run_dir / self.history[key])
        if not (Path(self.history["best_checkpoint"]) / "model.safetensors").is_file():
            raise ValueError("Best checkpoint is missing; copy the full run directory when relocating")
        self.encoder.invalidate_index()
        restore_rng_state(state["rng"], self.encoder.device)

    def save_checkpoint(self, epoch: int, history: dict[str, Any]) -> None:
        """Save the complete epoch state with portable checkpoint references."""
        checkpoint = self.run_dir / f"epoch-{epoch:03d}"
        portable_history = deepcopy(history)
        for key in ("best_checkpoint", "best_evaluation_config", "last_checkpoint"):
            if portable_history.get(key):
                portable_history[key] = str(Path(portable_history[key]).relative_to(self.run_dir))
        state = {
            "format_version": 1, "completed_epoch": epoch,
            "global_step": self.global_step, "total_steps": self.total_steps,
            "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
            "rng": capture_rng_state(self.encoder.device), "hard_negatives": self.hard_negatives,
            "history": portable_history, "model_type": self.model_type, "training_config": self.config,
            "encoder_config": self.encoder_config, "dataset_sha256": self.dataset_hashes,
            "source_revision": self.source_revision,
            "negative_encoder_config": self.negative_encoder_config,
        }
        save_epoch_checkpoint(
            checkpoint, self.encoder.model, self.encoder.tokenizer, state,
            self.evaluation_config(checkpoint),
        )

    def evaluation_config(self, checkpoint: Path) -> EncoderConfig:
        """Preserve the selected model and resolved retriever settings for evaluation."""
        config: EncoderConfig = {**self.encoder_config, "checkpoint": str(checkpoint), "revision": None}
        if self.negative_encoder_config is not None:
            config["retriever"] = deepcopy(self.negative_encoder_config)
        return config

    def publish_history(self) -> None:
        """Update run metrics and last/best pointers from a committed checkpoint."""
        write_json(self.run_dir / "metrics.json", self.history)
        checkpoints = set()
        for name in ("last", "best"):
            checkpoint = self.history.get(f"{name}_checkpoint")
            if checkpoint:
                write_json(self.run_dir / f"{name}.json", {"checkpoint": Path(checkpoint).name})
                checkpoints.add(checkpoint)
        for checkpoint in checkpoints:
            write_json(Path(checkpoint) / "evaluation_config.json", self.evaluation_config(checkpoint))

    def record_epoch(self, epoch: int, train_loss: float, metrics: dict[str, Any], started: float) -> None:
        """Commit every epoch and update best selection after the snapshot is published."""
        history = deepcopy(self.history)
        checkpoint = self.run_dir / f"epoch-{epoch:03d}"
        if self.selection_metric == "accuracy":
            score = metrics["accuracy"]
        else:
            score = metrics["recall_at_k"][str(self.config["selection_k"])]
        if history[self.best_metric_key] is None or score > history[self.best_metric_key]:
            history[self.best_metric_key] = score
            history["best_checkpoint"] = str(checkpoint)
            history["best_evaluation_config"] = str(checkpoint / "evaluation_config.json")
        history["last_checkpoint"] = str(checkpoint)
        record = {
            "epoch": epoch, "step": self.global_step, "train_loss": train_loss,
            "validation": metrics, "seconds": perf_counter() - started,
        }
        history["epochs"].append(record)
        self.save_checkpoint(epoch, history)
        self.history = history
        self.completed_epoch = epoch
        self.publish_history()
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def fit(self) -> None:
        """Train remaining epochs, appending logs after the restored checkpoint boundary."""
        # A resumed run shares its log and pointers with the original process.
        with FileLock(self.run_dir / ".training.lock", timeout=0):
            if self.completed_epoch:
                latest = resolve_checkpoint(self.run_dir)
                if latest.name != f"epoch-{self.completed_epoch:03d}":
                    raise ValueError("Run advanced while loading its state; retry resume")
                self.publish_history()
                trim_training_log(self.run_dir / "training.jsonl", self.global_step)
            self.save_metadata()
            with (self.run_dir / "training.jsonl").open("a", encoding="utf-8") as log:
                for epoch in range(self.completed_epoch + 1, self.config["epochs"] + 1):
                    started = perf_counter()
                    train_loss = self.train_epoch(epoch, log)
                    metrics = self.evaluate()
                    self.record_epoch(epoch, train_loss, metrics, started)
        print(f"Best checkpoint: {self.history['best_checkpoint']}", flush=True)


def run_training(args: TrainingOptions, trainer_class: type[Trainer], default_config: Path) -> None:
    """Start or resume training using explicit options from an entry point."""
    checkpoint = resolve_checkpoint(args.resume) if args.resume is not None else None
    config_path = args.config
    if config_path is None and (checkpoint is None or args.profile is not None):
        config_path = Path(default_config)
    config: TrainingConfig | None = None
    encoder_config: EncoderConfig | None = None
    if config_path is not None:
        config, encoder_config = load_training_config(config_path, args.profile)
    if args.actions is not None:
        if encoder_config is None:
            encoder_config = load_training_state(checkpoint)["encoder_config"].copy()
        encoder_config["actions_path"] = str(args.actions)
    trainer = trainer_class(config, encoder_config, checkpoint)
    trainer.fit()

