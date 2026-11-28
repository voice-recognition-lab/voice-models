"""Read profile configurations and immutable legacy run snapshots."""
import json
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def read_config(path=DEFAULT_CONFIG, profile=None):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Configuration must be an object")
    if "VOICES" in value:
        if profile is not None:
            raise ValueError("Legacy snapshots do not accept --profile")
        return value
    name = profile or "cpu_full"
    if name not in value:
        raise ValueError(f"Unknown profile: {name}; available: {sorted(value)}")
    selected = value[name]
    if not isinstance(selected, dict) or set(selected) != {"pipeline", "tts", "asr"}:
        raise ValueError("Profile requires pipeline, tts and asr sections")
    pipeline, tts = selected["pipeline"], selected["tts"]
    if set(pipeline) != {"dataset", "output_dir", "interpreters", "sample_limit"}:
        raise ValueError("Invalid pipeline fields")
    tts_fields = {"voices", "noise_snrs_db", "seed", "model", "model_sha256", "cpu_threads", "put_accent", "put_yo"}
    if not isinstance(tts, dict) or set(tts) - {"device"} != tts_fields:
        raise ValueError("Invalid tts fields")
    for field in ("dataset", "output_dir"):
        if not isinstance(pipeline[field], str) or not pipeline[field].strip():
            raise ValueError(f"pipeline.{field} must be a nonempty path")
    limit = pipeline["sample_limit"]
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("sample_limit must be null or a positive integer")
    return {
        "samples": pipeline["dataset"].rstrip("/") + "/dataset/samples.jsonl",
        "output": pipeline["output_dir"], "interpreters": pipeline["interpreters"],
        "sample_limit": limit, "VOICES": tts["voices"],
        "NOISE_SNRS_DB": tts["noise_snrs_db"], "SEED": tts["seed"],
        "TTS": {k: v for k, v in tts.items() if k not in ("voices", "noise_snrs_db", "seed")},
        "ASR": selected["asr"],
    }
