"""Sequential pipeline orchestrator; all processing is delegated to existing scripts."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from input_validation import load_config, load_samples
from pipeline_config import DEFAULT_CONFIG
from recognition_config import asr_device, validate_asr, validate_tts

REPO = Path(__file__).resolve().parents[3]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_status(root, status):
    temporary = root / "status.json.tmp"
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(root / "status.json")


def probe_environment(python, packages):
    code = (
        "import sys,json,importlib.metadata as m; "
        "assert sys.version_info >= (3,10), 'Python >= 3.10 required'; "
        "print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,"
        "'base_prefix':sys.base_prefix,'python':sys.version,"
        "'packages':{name:m.version(name) for name in sys.argv[1:]}}))"
    )
    try:
        result = subprocess.run([str(python), "-c", code, *packages], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=60, check=True)
        environment = json.loads(result.stdout)
        if environment["prefix"] == environment["base_prefix"]:
            raise ValueError(f"Interpreter is not in a virtual environment: {python}")
        return environment
    except (subprocess.SubprocessError, json.JSONDecodeError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise ValueError(f"Environment check failed for {python}: {detail[-2000:]}") from error


def probe_accelerator(python, backend, settings):
    command = [str(python), "-B", str(Path(__file__).with_name("cuda_runtime.py")),
               backend, settings["device"], settings["compute_type"]]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=60, check=True)
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise ValueError(f"CUDA preflight failed for {backend} with {python}: {detail.strip()}") from error


def preflight(config_path, repo=REPO, probe=probe_environment, profile=None):
    config = load_config(config_path, profile)
    required = {"samples", "output", "interpreters", "VOICES", "NOISE_SNRS_DB", "SEED", "TTS", "ASR"}
    if not required <= config.keys() or config.keys() - required - {"sample_limit"}:
        raise ValueError(f"Pipeline configuration requires {sorted(required)}")
    for name in ("samples", "output"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} must be a nonempty path")
    samples_path = (repo / config["samples"]).resolve()
    output = (repo / config["output"]).resolve()
    if not output.is_relative_to(repo.resolve()) or output == repo.resolve():
        raise ValueError("Output must be a new directory inside the repository")
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}")
    if not samples_path.is_file():
        raise ValueError(f"Samples file not found: {samples_path}")
    samples = load_samples(samples_path)
    limit = config.get("sample_limit")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("sample_limit must be null or positive")
        samples = samples[:limit]
    validate_tts(config["TTS"])
    if not isinstance(config["ASR"], dict) or set(config["ASR"]) != {"whisper", "gigaam"}:
        raise ValueError("ASR must contain whisper and gigaam configurations")
    for name in ("whisper", "gigaam"):
        validate_asr(name, config["ASR"][name])
    interpreters = config["interpreters"]
    if not isinstance(interpreters, dict) or set(interpreters) != {"main", "gigaam"}:
        raise ValueError("interpreters must contain main and gigaam")
    resolved = {**config, "samples": str(samples_path), "output": str(output), "interpreters": {}}
    for name, value in interpreters.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid interpreter path: {name}")
        path = (repo / value).resolve()
        if not path.is_file():
            raise ValueError(f"Interpreter not found: {name}={path}; see the ASR/TTS README for environment setup")
        resolved["interpreters"][name] = str(path)
    environments = {
        "main": probe(Path(resolved["interpreters"]["main"]),
                      ["torch", "numpy", "scipy", "soundfile", "faster-whisper", "ctranslate2",
                       "av", "huggingface-hub", "jiwer"]),
        "gigaam": probe(Path(resolved["interpreters"]["gigaam"]),
                        ["torch", "torchaudio", "transformers", "huggingface-hub", "hydra-core",
                         "sentencepiece", "soundfile", "imageio-ffmpeg", "pyannote.core"]),
    }
    for backend, interpreter in (("whisper", "main"), ("gigaam", "gigaam")):
        settings = config["ASR"][backend]
        if asr_device(settings["device"])[0] == "cuda":
            environments[interpreter]["gpu"] = probe_accelerator(
                resolved["interpreters"][interpreter], backend, settings)
    return config, resolved, samples, environments


def build_stages(config, root, repo=REPO):
    primary = config["interpreters"]["main"]
    gigaam = config["interpreters"]["gigaam"]
    dataset = str(root / "dataset")
    # Recognition receives only model settings, never annotation paths/content.
    run_config = str(root / "config.models.json")
    def command(python, script, *arguments):
        folder = repo / "evaluation" if script in {"evaluate.py", "compare_conditions.py"} else repo / "dataset-generator" / "asr-tts" / "src"
        return [python, "-u", str(folder / script), *map(str, arguments)]
    stages = [
        ("tts_checks_and_noise", command(primary, "synthesize.py", "--samples", root / "input_samples.jsonl", "--config", run_config, "--output", dataset)),
        ("audio_audit", command(primary, "check_dataset.py", "--dataset", dataset, "--output", root / "dataset/dataset_checks.json")),
        ("whisper", command(primary, "transcribe.py", "--dataset", dataset, "--config", run_config, "--output", root / "whisper")),
        ("gigaam", command(gigaam, "transcribe_gigaam.py", "--dataset", dataset, "--config", run_config, "--output", root / "gigaam")),
        ("evaluate_whisper", command(primary, "evaluate.py", "--dataset", dataset, "--results", root / "whisper", "--output", root / "evaluation-whisper")),
        ("evaluate_gigaam", command(primary, "evaluate.py", "--dataset", dataset, "--results", root / "gigaam", "--output", root / "evaluation-gigaam")),
    ]
    stages.append(("comparison", command(primary, "compare_conditions.py", "--dataset", dataset,
                   "--whisper", root / "evaluation-whisper", "--gigaam", root / "evaluation-gigaam",
                   "--output", root / "comparison")))
    return stages


def run_command(command, log, cwd):
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with log.open("x", encoding="utf-8") as stream:
        with subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, env=environment) as process:
            try:
                return process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise


def verify_stage_outputs(root, name, expected_wav):
    expected = {
        "tts_checks_and_noise": ["dataset/audio_manifest.jsonl", "dataset/samples.jsonl", "dataset/tts_settings.json", "dataset/audio_checks.json"],
        "audio_audit": ["dataset/dataset_checks.json"],
        "whisper": ["whisper/predictions.jsonl", "whisper/timings.jsonl", "whisper/asr_settings.json"],
        "gigaam": ["gigaam/predictions.jsonl", "gigaam/timings.jsonl", "gigaam/asr_settings.json"],
        "evaluate_whisper": ["evaluation-whisper/summary.json", "evaluation-whisper/record_metrics.jsonl"],
        "evaluate_gigaam": ["evaluation-gigaam/summary.json", "evaluation-gigaam/record_metrics.jsonl"],
        "comparison": ["comparison/report.md", "comparison/summary.json"],
    }
    for relative in expected.get(name, []):
        if not (root / relative).is_file():
            raise RuntimeError(f"Stage {name} did not create required artifact: {relative}")
    if name in ("whisper", "gigaam"):
        settings = json.loads((root / name / "asr_settings.json").read_text(encoding="utf-8"))
        if settings.get("status") != "completed":
            raise RuntimeError(f"Stage {name} returned partial ASR results")
    if name == "audio_audit":
        qc = json.loads((root / "dataset/dataset_checks.json").read_text(encoding="utf-8"))
        if not qc.get("passed") or qc["files"] != expected_wav:
            raise RuntimeError("Audio audit failed or WAV count differs from configuration")


def execute_stages(root, stages, status, runner=run_command, repo=REPO):
    status["status"] = "running"
    status["stages"] = [{"name": name, "status": "pending", "command": command,
                         "log": f"logs/{name}.log"} for name, command in stages]
    write_status(root, status)
    current = None
    try:
        for current in status["stages"]:
            current.update(status="running", started_at=timestamp())
            write_status(root, status)
            print(f"Starting {current['name']} (log: {root / current['log']})", flush=True)
            code = runner(current["command"], root / current["log"], repo)
            current["exit_code"] = code
            if code != 0:
                log = root / current["log"]
                tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]) if log.is_file() else ""
                raise RuntimeError(f"Stage {current['name']} failed with exit code {code}; see {log}\n{tail}")
            verify_stage_outputs(root, current["name"], status.get("expected_wav"))
            current.update(status="completed", finished_at=timestamp(), exit_code=0)
            write_status(root, status)
        status.update(status="completed", finished_at=timestamp())
        write_status(root, status)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if current is not None:
            current.update(status="failed", finished_at=timestamp(), error=str(error) or "Interrupted")
        for stage in status["stages"]:
            if stage["status"] == "pending":
                stage["status"] = "skipped"
        status.update(status="failed", finished_at=timestamp(), error=str(error) or "Interrupted")
        write_status(root, status)
        print(f"Pipeline FAILED: {status['error']}", file=sys.stderr, flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", help="Configuration profile (default: cpu_full)")
    parser.add_argument("--check", action="store_true", help="Validate data and environments without loading models")
    args = parser.parse_args()
    try:
        config, resolved, samples, environments = preflight(args.config, profile=args.profile)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Preflight failed before loading models: {error}\n")
    if args.check:
        print(f"Preflight passed: {len(samples)} samples; no models loaded")
        return
    root = Path(resolved["output"])
    try:
        root.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.exit(2, f"Cannot reserve new output directory: {error}\n")
    (root / "logs").mkdir()
    for filename, value in (("config.requested.json", config), ("config.resolved.json", resolved),
                            ("environments.json", environments),
                            ("config.models.json", {key: config[key] for key in ("VOICES", "NOISE_SNRS_DB", "SEED", "TTS", "ASR")})):
        with (root / filename).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
    with (root / "input_samples.jsonl").open("x", encoding="utf-8") as stream:
        for row in samples:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    expected = len(samples) * len(config["VOICES"]) * (1 + len(config["NOISE_SNRS_DB"]))
    status = {"started_at": timestamp(), "preflight": "passed", "unique_samples": len(samples),
              "unique_group_ids": len({s["group_id"] for s in samples}), "expected_wav": expected}
    code = execute_stages(root, build_stages(resolved, root), status)
    if code == 0:
        print(f"Pipeline completed. Report: {root / 'comparison/report.md'}", flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
