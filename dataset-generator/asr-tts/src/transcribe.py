"""Whisper recognition on CPU or NVIDIA CUDA from an audio manifest."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

# Load PyAV/CTranslate2 before libsndfile: the opposite order stalled imports
# in this Windows Python 3.11 environment.
from faster_whisper import WhisperModel
from huggingface_hub import HfApi, snapshot_download
import soundfile as sf
from cuda_runtime import probe_cuda
from pipeline_config import DEFAULT_CONFIG, read_config
from recognition_config import asr_device, validate_asr


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cpu_name():
    if platform.system() == "Windows":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def transcribe(model, path, decoding):
    segments, _ = model.transcribe(str(path), **decoding)
    # Collect every segment and its host tokens from blocking CTranslate2 calls.
    return " ".join(segment.text.strip() for segment in segments).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", help="Profile in config.json (default: cpu_full)")
    args = parser.parse_args()
    config = validate_asr("whisper", read_config(args.config, args.profile)["ASR"]["whisper"])
    model_id = config["model"]
    requested_revision = config["revision"]
    threads = config["cpu_threads"]
    device, index = asr_device(config["device"])
    compute_type = config["compute_type"]
    gpu = None
    if device == "cuda":
        try:
            gpu = probe_cuda("whisper", index, compute_type)
        except ValueError as error:
            parser.error(str(error))
    decoding = config["decoding"]
    root = args.dataset.resolve()
    manifest_path = root / "audio_manifest.jsonl"
    rows = []
    ids = set()
    for line in manifest_path.read_text(encoding="utf-8-sig").splitlines():
        row = json.loads(line)
        for field in ("audio_id", "sample_id", "path"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Invalid {field}")
        if row["audio_id"] in ids:
            raise ValueError("Duplicate audio_id")
        path = (root / row["path"]).resolve()
        if Path(row["path"]).is_absolute() or not path.is_relative_to(root):
            raise ValueError("Audio path must be relative to dataset root")
        info = sf.info(path)
        if (info.format, info.subtype, info.channels, info.samplerate) != ("WAV", "PCM_16", 1, 16000):
            raise ValueError(f"Unexpected audio format: {path}")
        if info.frames <= 0:
            raise ValueError("Empty audio")
        rows.append((row, path, info.frames / info.samplerate))
        ids.add(row["audio_id"])
    if not rows:
        raise ValueError("Empty manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    revision = HfApi().model_info(model_id, revision=requested_revision).sha
    model_path = snapshot_download(
        model_id, revision=revision, cache_dir=str(args.cache_dir),
        allow_patterns=["config.json", "model.bin", "tokenizer.json", "vocabulary.*", "preprocessor_config.json"],
    )
    model = WhisperModel(model_path, device=device, device_index=index, compute_type=compute_type,
                         cpu_threads=threads, num_workers=1)
    actual_device = model.model.device
    if actual_device != device or (gpu is not None and model.model.device_index != [index]):
        raise RuntimeError(f"Whisper resolved to unexpected device {actual_device}:{model.model.device_index}")
    print("Model loaded; warming up on first WAV (not timed).", flush=True)
    transcribe(model, rows[0][1], decoding)
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(model.transcribe).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_id, "requested_revision": requested_revision, "resolved_revision": revision,
        "device": f"cuda:{index}" if gpu is not None else actual_device,
        "compute_type": model.model.compute_type, "gpu": gpu,
        "cpu_threads": threads, "num_workers": 1,
        "decoding": {**defaults, **decoding},
        "hardware": {"cpu": cpu_name(), "logical_cpus": os.cpu_count(),
                     "os": platform.platform(), "machine": platform.machine()},
        "python": platform.python_version(),
        "dependencies": sorted(
            [{"name": d.metadata["Name"], "version": d.version}
             for d in importlib.metadata.distributions()], key=lambda d: d["name"].lower()),
        "manifest_sha256": digest(manifest_path), "script_sha256": digest(Path(__file__)),
        "warmup": {"runs": 1, "audio_id": rows[0][0]["audio_id"], "fully_consumed": True},
        "timing": "perf_counter: audio loading, feature extraction, decoding and all segments; excludes model load, warmup and output I/O",
        "status": "running",
    }
    settings_path = args.output / "asr_settings.json"
    with settings_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
    with (args.output / "predictions.jsonl").open("x", encoding="utf-8") as predictions, \
            (args.output / "timings.jsonl").open("x", encoding="utf-8") as timings:
        for row, path, duration in rows:
            start = perf_counter()
            text = transcribe(model, path, decoding)
            elapsed = perf_counter() - start
            prediction = {"sample_id": row["sample_id"], "audio_id": row["audio_id"], "text": text}
            timing = {"sample_id": row["sample_id"], "audio_id": row["audio_id"],
                      "duration_sec": duration, "elapsed_sec": elapsed, "rtf": elapsed / duration,
                      "audio_sha256": digest(path)}
            predictions.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            timings.write(json.dumps(timing) + "\n")
            predictions.flush()
            timings.flush()
            print(f"{row['audio_id']}: {elapsed:.3f}s, RTF={elapsed / duration:.3f}", flush=True)
    metadata["status"] = "completed"
    # Only this run's own metadata is updated; existing output roots are rejected.
    settings_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
