# Cross-encoder

Score command/action pairs and rerank candidates.

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
- `cpu_trial` — 512 train / 128 validation, 2 epochs (~10–15 min on CPU).
- `gpu_full` — full dataset, one GPU (`cuda:0`).

Each profile contains two sections:

### `encoder` — model and evaluation

- `checkpoint`, `dataset`, `device` — model, data directory, compute device.
- `batch_size` — pairs per encoding batch.
- `top_k` — Recall@k metric cutoffs.
- `actions_path` — optional catalog override; default: `<dataset>/inputs/actions.jsonl`.
- `retriever_config`, `retriever_profile` — bi-encoder file/profile for hybrid retrieval and hard negatives.
- `retrieval_top_k` — hybrid shortlist size (30).

### `training` — fine-tuning

- `epochs`, `batch_size`, `learning_rate` — epochs, commands per step, learning rate.
- `train_limit`, `validation_limit` — sample limits; omit for full splits.
- `hard_negatives`, `random_negatives` — incorrect candidates per command.
- `output_dir` — checkpoint/log destination.

## Training

### CPU trial

```bash
python -m text_to_actions.cross_encoder.train.run --profile cpu_trial
```

### Full GPU run

```bash
python -m text_to_actions.cross_encoder.train.run --profile gpu_full
```

### Resume

```bash
python -m text_to_actions.cross_encoder.train.run --resume RUN
```

`RUN`: existing run directory, `last.json`, or latest completed `epoch-XXX`.
Restores saved settings, weights, optimizer, scheduler, step, and random state.
Copy the full run when moving machines. Learning settings and catalog contents must match.

## Validation and test

### Pretrained baseline

```bash
python -m text_to_actions.cross_encoder.evaluate.run --profile cpu_full
```

The default model has a random scoring head until fine-tuned.

### Trained checkpoint

`CHECKPOINT`: a saved `epoch-XXX` directory. Its config needs no profile.

```bash
python -m text_to_actions.cross_encoder.evaluate.run \
  --config CHECKPOINT/evaluation_config.json --mode hybrid
python -m text_to_actions.cross_encoder.test.run \
  --config CHECKPOINT/evaluation_config.json --mode hybrid
```

### Cross-encoder modes

- `--mode all` — score every available action (default).
- `--mode hybrid` — bi-encoder retrieves top-k; cross-encoder reranks them.

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
