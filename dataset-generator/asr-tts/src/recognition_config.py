"""Supported recognition settings, shared by preflight and ASR entrypoints."""

import re

from pipeline_config import read_config


def asr_device(value):
    if value == "cpu":
        return "cpu", 0
    if isinstance(value, str) and re.fullmatch(r"cuda(?::(?:0|[1-9][0-9]*))?", value):
        _, _, index = value.partition(":")
        return "cuda", int(index or 0)
    raise ValueError(f"Expected cpu, cuda or cuda:N, got {value!r}")


def validate_asr(name, value):
    required = {"model", "revision", "device", "compute_type", "cpu_threads", "decoding"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"ASR.{name} requires exactly {sorted(required)}")
    expected_model = read_config()["ASR"][name]["model"]
    device, _ = asr_device(value["device"])
    supported = ("int8", "float32") if name == "whisper" else ("float32",)
    if value["model"] != expected_model or value["compute_type"] not in supported:
        raise ValueError(f"ASR.{name} on {device} requires {expected_model} and compute_type in {supported}")
    if not isinstance(value["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", value["revision"]):
        raise ValueError(f"ASR.{name}.revision must be an immutable 40-character commit")
    threads = value["cpu_threads"]
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 64:
        raise ValueError(f"ASR.{name}.cpu_threads must be in [1, 64]")
    decoding = value["decoding"]
    if not isinstance(decoding, dict):
        raise ValueError(f"ASR.{name}.decoding must be an object")
    if name == "gigaam":
        if decoding != {"method": "greedy_ctc"}:
            raise ValueError("GigaAM e2e_ctc supports its official greedy_ctc decoder only")
    else:
        required = {"language", "task", "beam_size", "temperature", "condition_on_previous_text",
                    "vad_filter", "word_timestamps"}
        optional = {"initial_prompt", "hotwords"}
        if not required <= decoding.keys() or decoding.keys() - required - optional:
            raise ValueError("Invalid Whisper decoding keys")
        if (decoding["language"] != "ru" or decoding["task"] != "transcribe"
                or any(decoding.get(key) is not None for key in optional)):
            raise ValueError("Whisper requires ru/transcribe and no prompt/hotwords")
        if isinstance(decoding["beam_size"], bool) or not isinstance(decoding["beam_size"], int) or not 1 <= decoding["beam_size"] <= 20:
            raise ValueError("Invalid Whisper beam_size")
        if isinstance(decoding["temperature"], bool) or not isinstance(decoding["temperature"], (int, float)) or not 0 <= decoding["temperature"] <= 1:
            raise ValueError("Invalid Whisper temperature")
        for key in ("condition_on_previous_text", "vad_filter", "word_timestamps"):
            if not isinstance(decoding[key], bool):
                raise ValueError(f"Whisper {key} must be boolean")
    return value


def validate_tts(value):
    required = {"model", "model_sha256", "cpu_threads", "put_accent", "put_yo"}
    if not isinstance(value, dict) or set(value) - {"device"} != required:
        raise ValueError(f"TTS requires {sorted(required)} with optional device")
    if value.get("device", "cpu") != "cpu":
        raise ValueError("TTS.device must be cpu")
    if value["model"] != "v5_ru" or not isinstance(value["model_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["model_sha256"]):
        raise ValueError("TTS requires v5_ru and its SHA256 revision")
    threads = value["cpu_threads"]
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 64:
        raise ValueError("Invalid TTS cpu_threads")
    if any(not isinstance(value[key], bool) for key in ("put_accent", "put_yo")):
        raise ValueError("TTS accent/yo settings must be boolean")
    return value
