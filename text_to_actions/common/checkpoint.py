"""Epoch checkpoint publication, training state loading, and RNG restoration."""

import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .io import write_atomic, write_json
from .types import EncoderConfig, PathLike, TrainingConfig, TrainingState


def capture_rng_state(device: str | torch.device) -> dict[str, Any]:
    """Capture Python, CPU, and the selected training GPU random generators."""
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.device(device).type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: dict[str, Any], device: str | torch.device) -> None:
    """Restore available generators after model and optimizer initialization."""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.device(device).type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def save_epoch_checkpoint(
    path: PathLike, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
    state: TrainingState, evaluation_config: EncoderConfig,
) -> None:
    """Publish weights and the matching training state as one complete directory."""
    path = Path(path)
    with TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as directory:
        temporary = Path(directory)
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        torch.save(state, temporary / "training_state.pt")
        write_json(temporary / "evaluation_config.json", evaluation_config)
        write_json(temporary / "checkpoint.json", {
            "format_version": 1, "completed_epoch": state["completed_epoch"],
            "global_step": state["global_step"],
        })
        temporary.rename(path)


def resolve_checkpoint(path: PathLike) -> Path:
    """Find the latest complete epoch, including a checkpoint published before a crash."""
    path = Path(path).resolve()
    explicit = None
    if path.is_file():
        pointer = json.loads(path.read_text(encoding="utf-8"))
        explicit = path.parent / pointer["checkpoint"]
        run_dir = path.parent
    elif (path / "training_state.pt").is_file():
        explicit = path
        run_dir = path.parent
    elif (path / "model.safetensors").is_file():
        raise ValueError(f"{path}: weights-only checkpoint has no state for resuming training")
    else:
        run_dir = path
    completed = []
    for candidate in run_dir.glob("epoch-*"):
        if all((candidate / name).is_file() for name in (
            "checkpoint.json", "training_state.pt", "model.safetensors", "evaluation_config.json",
        )):
            metadata = json.loads((candidate / "checkpoint.json").read_text(encoding="utf-8"))
            completed.append((metadata["completed_epoch"], candidate))
    if not completed:
        raise ValueError(f"{path}: no complete training checkpoint found")
    latest = max(completed, key=lambda item: item[0])[1]
    if explicit is not None and explicit.resolve() != latest.resolve():
        raise ValueError(f"Resume from the latest completed epoch: {latest}")
    return latest


def load_training_state(path: PathLike) -> TrainingState:
    """Load tensor and primitive training state on CPU for portable restoration."""
    state = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported training checkpoint format")
    return state


def validate_resume_config(
    state: TrainingState, config: TrainingConfig, encoder_config: EncoderConfig, dataset_hashes: dict[str, str],
) -> None:
    """Check data and learning settings while allowing new paths and devices."""
    if dataset_hashes != state["dataset_sha256"]:
        raise ValueError("Dataset contents differ from the checkpoint")
    allowed_training = {"encoder_config", "output_dir", "log_every"}
    allowed_encoder = {"dataset", "actions_path", "device", "model_cache", "embedding_cache", "num_threads"}
    for label, current, saved, allowed in (
        ("training", config, state["training_config"], allowed_training),
        ("encoder", encoder_config, state["encoder_config"], allowed_encoder),
    ):
        changed = [key for key in current.keys() | saved.keys()
                   if key not in allowed and current.get(key) != saved.get(key)]
        if changed:
            raise ValueError(f"Incompatible {label} resume settings: {sorted(changed)}")


def trim_training_log(path: PathLike, global_step: int) -> None:
    """Discard log records beyond the last committed training step."""
    path = Path(path)
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        records = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break
                raise ValueError(f"{path}:{index + 1}: invalid committed log record")
            if record["step"] > global_step:
                break
            records.append(record)
        write_atomic(path, "".join(json.dumps(record) + "\n" for record in records))
