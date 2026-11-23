# Bi-encoder

Retrieve actions using independent command and action embeddings.

## Installation

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r text_to_actions/requirements.txt
```

GPU: CUDA-enabled PyTorch and an NVIDIA driver.

## [config.json](config.json)

### Profiles

- `cpu_full` — full dataset, CPU (default).
- `cpu_trial` — 256 train / 128 validation, 1 epoch.
- `gpu_full` — full dataset, one GPU (`cuda:0`).

Each profile contains two sections:

### `encoder` — model and evaluation

- `checkpoint`, `dataset`, `device` — model, data directory, compute device.
- `batch_size` — texts per encoding batch.
- `top_k` — Recall@k metric cutoffs.
- `actions_path` — optional catalog override; default: `<dataset>/inputs/actions.jsonl`.
- `embedding_cache` — action embedding cache directory.

### `training` — fine-tuning

- `epochs`, `batch_size`, `learning_rate` — epochs, commands per step, learning rate.
- `train_limit`, `validation_limit` — sample limits; omit for full splits.
- `hard_negatives`, `random_negatives` — incorrect candidates per command.
- `output_dir` — checkpoint/log destination.

## Training

### CPU trial

```bash
python -m text_to_actions.bi_encoder.train.run --profile cpu_trial
```

### Full GPU run

```bash
python -m text_to_actions.bi_encoder.train.run --profile gpu_full
```

### Resume

```bash
python -m text_to_actions.bi_encoder.train.run --resume RUN
```

`RUN`: existing run directory, `last.json`, or latest completed `epoch-XXX`.
Restores saved settings, weights, optimizer, scheduler, step, and random state.
Copy the full run when moving machines. Learning settings and catalog contents must match.

## Validation and test

### Pretrained baseline

```bash
python -m text_to_actions.bi_encoder.evaluate.run --profile cpu_full
```

### Trained checkpoint

`CHECKPOINT`: a saved `epoch-XXX` directory. Its config needs no profile.

```bash
python -m text_to_actions.bi_encoder.evaluate.run \
  --config CHECKPOINT/evaluation_config.json
python -m text_to_actions.bi_encoder.test.run \
  --config CHECKPOINT/evaluation_config.json
```

### Flags

- `--config PATH` — another profile file, or a saved evaluation config for validation/test.
- `--profile NAME` — select a profile; default: `cpu_full` for new runs. Resume without it uses saved settings.
- `--checkpoint DIR` — validation/test only: override model weights/tokenizer.
- `--limit N` — validation/test only: first N samples; omit for the full split.
- `--output DIR` — validation/test results destination.

`training.train_limit` and `training.validation_limit` affect training only, not standalone evaluation.

## Russian descriptions

Add to training or evaluation commands:

```bash
--actions text_to_actions/translate_experiment/artifacts/opus_mt/actions.ru.jsonl
```

Overrides `encoder.actions_path`. Switching catalog contents requires a new training run.
