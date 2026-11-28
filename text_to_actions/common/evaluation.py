"""Evaluate candidate ranking over a catalog or a retrieved shortlist."""

from datetime import datetime, timezone
import json
from pathlib import Path
import random
from time import perf_counter
from typing import Callable

import torch

from .io import write_json
from .hashing import file_sha256
from ..data import load_dataset, resolve_dataset_paths
from .metrics import retrieval_metrics
from .config import load_encoder_config, validate_encoder_config
from .cli import EvaluationOptions
from .artifacts import runtime_metadata, write_predictions
from .types import EncoderConfig, RankingEncoder, Sample


def run_evaluation(
    args: EvaluationOptions, model_factory: Callable[[EncoderConfig], RankingEncoder],
    retriever_factory: Callable[[EncoderConfig], tuple[RankingEncoder, EncoderConfig]] | None = None,
) -> None:
    """Evaluate a model using parsed options, optionally reranking candidates."""
    config = load_encoder_config(args.config, args.profile)
    if args.actions is not None:
        config["actions_path"] = str(args.actions)
    if args.checkpoint is not None:
        config["checkpoint"] = str(args.checkpoint)
        config["revision"] = None
    validate_encoder_config(config)
    if args.mode == "hybrid":
        if retriever_factory is None:
            raise ValueError("Hybrid evaluation requires a retriever factory")
        if "retrieval_top_k" not in config:
            raise ValueError("Hybrid evaluation requires retrieval_top_k")
        if "retriever" not in config and "retriever_config" not in config:
            raise ValueError("Hybrid evaluation requires retriever_config or saved retriever settings")
    top_k = config["top_k"]
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["num_threads"])

    dataset = load_dataset(config["dataset"], config.get("actions_path"))
    encoder = model_factory(config)
    retriever = None
    retriever_config = None
    if args.mode == "hybrid":
        retriever, retriever_config = retriever_factory(config)
        retriever.model.eval()
        retriever.index_actions(dataset.actions)
    start = perf_counter()
    encoder.index_actions(dataset.actions)
    index_seconds = perf_counter() - start

    run_dir = args.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True)
    hashes = {
        str(path): file_sha256(path)
        for path in resolve_dataset_paths(config["dataset"], config.get("actions_path")).values()
    }
    report = {
        "config": config, "resolved_revision": encoder.revision,
        **runtime_metadata(), "dataset_sha256": hashes,
        "catalog_actions": len(dataset.actions),
        "pooling": encoder.pooling, "score_type": encoder.score_type, "index_seconds": index_seconds,
        "limit": args.limit, "results": {}, "mode": args.mode,
        "retriever_config": retriever_config,
        "retriever_revision": retriever.revision if retriever is not None else None,
    }
    for split in dict.fromkeys(args.split):
        samples = dataset.samples_for_split(split)
        if args.limit is not None:
            samples = samples[:args.limit]
        start = perf_counter()
        requests: list[Sample] = samples
        retrieved = None
        if retriever is not None:
            retrieved = retriever.search(samples, config["retrieval_top_k"])
            requests = [{**sample, "available_action_ids": [item["action_id"] for item in candidates]}
                        for sample, candidates in zip(samples, retrieved)]
        predictions = encoder.search(requests, max(top_k))
        elapsed = perf_counter() - start
        metrics = retrieval_metrics(samples, predictions, top_k)
        counts = [len(sample.get("available_action_ids", dataset.actions)) for sample in requests]
        metrics.update({
            "retrieval_seconds": elapsed,
            "amortized_ms_per_sample": elapsed * 1000 / len(samples) if samples else None,
            "candidate_count_min": min(counts, default=0),
            "candidate_count_max": max(counts, default=0),
        })
        if retrieved is not None:
            metrics["bi_encoder"] = retrieval_metrics(samples, retrieved, top_k)
        report["results"][split] = metrics
        write_predictions(run_dir / f"{split}.predictions.jsonl", samples, predictions, retrieved)
        print(json.dumps({"split": split, **metrics}, ensure_ascii=False, indent=2), flush=True)
    write_json(run_dir / "metrics.json", report)
    print(f"Results saved to {run_dir}")
