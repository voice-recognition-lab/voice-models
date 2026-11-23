"""Profile loading and validation of encoder and training settings."""

import math
import json
from pathlib import Path
from typing import Any, Mapping, cast

import torch

from .types import EncoderConfig, PathLike, TrainingConfig


def load_config_profile(path: PathLike, profile: str | None = None) -> dict[str, Any]:
    """Select a named profile or read a standalone saved configuration."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path}: configuration must be a JSON object")
    if "checkpoint" in config or "encoder_config" in config:
        if profile is not None:
            raise ValueError("--profile requires a file with named profiles")
        return config
    name = profile or "cpu_full"
    if name not in config:
        raise ValueError(f"Unknown profile {name!r} in {path}; available: {', '.join(config)}")
    selected = config[name]
    validate_keys(selected, {"encoder", "training"}, set(), f"profile {name}")
    return selected


def load_encoder_config(path: PathLike, profile: str | None = None) -> EncoderConfig:
    """Read encoder settings from a profile or a saved evaluation config."""
    config = load_config_profile(path, profile)
    encoder = config["encoder"] if "encoder" in config else config
    validate_encoder_config(encoder)
    return cast(EncoderConfig, encoder)


def load_training_config(path: PathLike, profile: str | None = None) -> tuple[TrainingConfig, EncoderConfig]:
    """Read training and encoder settings, retaining legacy config support."""
    config = load_config_profile(path, profile)
    if "training" in config and "encoder" in config:
        validate_config(config["training"], config["encoder"])
        return cast(TrainingConfig, config["training"]), cast(EncoderConfig, config["encoder"])
    encoder = load_encoder_config(config["encoder_config"])
    validate_config(config, encoder)
    return cast(TrainingConfig, config), encoder


def validate_encoder_config(config: Mapping[str, Any], model_type: str | None = None) -> None:
    """Reject missing, unknown, and invalid encoder settings before loading weights."""
    required = {
        "checkpoint", "revision", "dataset", "model_cache", "max_length", "batch_size",
        "device", "num_threads", "seed", "top_k",
    }
    optional = {"actions_path", "embedding_cache", "retriever_config", "retriever_profile", "retrieval_top_k", "retriever"}
    validate_keys(config, required, optional, "encoder")
    if model_type == "bi_encoder":
        validate_keys(config, required | {"embedding_cache"}, {"actions_path"}, "bi-encoder")
    elif model_type == "cross_encoder":
        validate_keys(config, required, optional - {"embedding_cache"}, "cross-encoder")
    for name in ("max_length", "batch_size", "num_threads"):
        validate_integer(config[name], name, minimum=1)
    validate_integer(config["seed"], "seed", minimum=0)
    for name in ("checkpoint", "dataset", "model_cache", "embedding_cache", "actions_path", "retriever_config"):
        if name in config and (not isinstance(config[name], (str, Path)) or not str(config[name]).strip()):
            raise ValueError(f"{name} must be a nonempty path or model identifier")
    revision = config["revision"]
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("revision must be a nonempty string or null")
    device = config["device"]
    if not isinstance(device, str):
        raise ValueError("device must be a string such as cpu or cuda:0")
    try:
        torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device: {device!r}") from exc
    top_k = config["top_k"]
    if not isinstance(top_k, list) or not top_k or any(type(k) is not int or k < 1 for k in top_k):
        raise ValueError("top_k must be a nonempty list of positive integers")
    if "retrieval_top_k" in config:
        validate_integer(config["retrieval_top_k"], "retrieval_top_k", minimum=1)
    if "retriever_profile" in config:
        if not isinstance(config["retriever_profile"], str) or not config["retriever_profile"].strip():
            raise ValueError("retriever_profile must be a nonempty string")
        if "retriever_config" not in config and "retriever" not in config:
            raise ValueError("retriever_profile requires retriever_config or saved retriever settings")
    if "retriever" in config:
        validate_encoder_config(config["retriever"], model_type="bi_encoder")


def validate_keys(config: Mapping[str, Any], required: set[str], optional: set[str], label: str) -> None:
    """Report malformed config sections, missing keys, and likely key typos."""
    if not isinstance(config, dict):
        raise ValueError(f"{label} must be a JSON object")
    missing = required - config.keys()
    unknown = config.keys() - required - optional
    if missing or unknown:
        problems = []
        if missing:
            problems.append(f"missing keys: {', '.join(sorted(missing))}")
        if unknown:
            problems.append(f"unknown keys: {', '.join(sorted(unknown))}")
        raise ValueError(f"Invalid {label} config: {'; '.join(problems)}")


def validate_integer(value: Any, name: str, minimum: int) -> None:
    """Check an integer bound without accepting booleans as integers."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_config(config: Mapping[str, Any], encoder_config: Mapping[str, Any], model_type: str | None = None) -> None:
    """Validate training and encoder settings without changing checkpoint selection."""
    required = {
        "epochs", "batch_size", "selection_k", "log_every", "hard_negatives", "random_negatives",
        "learning_rate", "temperature", "max_grad_norm", "warmup_ratio", "weight_decay",
        "refresh_hard_negatives", "output_dir",
    }
    validate_keys(config, required, {"train_limit", "validation_limit", "encoder_config"}, "training")
    validate_encoder_config(encoder_config, model_type)
    if model_type == "cross_encoder" and "retriever" not in encoder_config and "retriever_config" not in encoder_config:
        raise ValueError("Cross-encoder training requires retriever_config or saved retriever settings")
    for name in ("epochs", "batch_size", "selection_k", "log_every"):
        validate_integer(config[name], name, minimum=1)
    for name in ("hard_negatives", "random_negatives"):
        validate_integer(config[name], name, minimum=0)
    for name in ("train_limit", "validation_limit"):
        value = config.get(name)
        if value is not None:
            validate_integer(value, name, minimum=1)
    if config["hard_negatives"] + config["random_negatives"] == 0:
        raise ValueError("Specify at least one negative candidate")
    for name in ("learning_rate", "temperature", "max_grad_norm"):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
    for name in ("warmup_ratio", "weight_decay"):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if config["warmup_ratio"] >= 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if type(config["refresh_hard_negatives"]) is not bool:
        raise ValueError("refresh_hard_negatives must be boolean")
    if not isinstance(config["output_dir"], (str, Path)) or not str(config["output_dir"]).strip():
        raise ValueError("output_dir must be a nonempty path")
    if config["selection_k"] not in encoder_config["top_k"]:
        raise ValueError("selection_k must be included in top_k")
