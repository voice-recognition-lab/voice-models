# Text to actions

Convert Russian commands into structured actions using a catalog of action descriptions
and parameter schemas. The current implementation selects the action type; argument
extraction is the next planned stage.

## Planned architecture

```text
Command + optional context       Action catalog
             │                       │
             └──── Bi-encoder ────────┘
                         │
                 Top-k action candidates
                         │
                   Cross-encoder
                         │
                   Selected action
                         │
            Argument extraction and normalization
                         │
                 JSON Schema validation
                         │
                   action_id + args
```

### 1. Candidate retrieval — implemented

- One shared BERT encodes commands and actions independently.
- Command input: text and nonempty context. Action input: description and parameter schema.
- Normalized CLS embeddings are compared using cosine similarity.
- Action embeddings are cached while weights and catalog contents remain unchanged.

### 2. Candidate reranking — implemented

- Cross-encoder jointly processes each command/action pair and produces one score.
- Hybrid mode reranks the bi-encoder shortlist; its default size is 30.
- Full-catalog mode scores all available actions for comparison.

The two models are trained separately. The planned full run first fine-tunes the
bi-encoder, then uses that checkpoint for cross-encoder hard negatives and hybrid retrieval.
Current default configs still reference the pretrained bi-encoder.

### 3. Arguments and output — planned

- Extract values for the selected action's parameters from command text or context.
- Keep value sources/spans separate from normalized argument values.
- Validate the final `args` against the selected action's JSON Schema.

Description-based scoring allows new actions to enter the catalog without adding
classifier labels. Quality on unseen actions still requires separate evaluation.

## Project status

### Implemented

- Dataset contract validation and existing train/validation/test splits.
- Bi-encoder and cross-encoder training with hard and random negatives.
- Validation/test entry points, Accuracy and Recall@k metrics.
- Epoch checkpoints with optimizer, scheduler, and random state for resume.
- English or translated Russian descriptions through `--actions`.

### Completed experiments

- Full validation baselines: 1,653 commands, 55 candidate actions.
- Bi-encoder CPU training probes.
- Cross-encoder CPU trial: 512 train / 128 validation samples, two epochs.
- OPUS-MT translation and English/Russian catalog comparison.

See [baseline reports](reports/). The untrained cross-encoder baseline uses a random
scoring head. Small training runs verify the pipeline; full-training quality remains unmeasured.

### Next steps

1. Fine-tune both models on the full training split using one 16 GB NVIDIA GPU.
2. Evaluate the hybrid pipeline and choose the retrieval shortlist size on validation.
3. Evaluate selected checkpoints on the held-out test split.
4. Prepare argument annotations and implement extraction and normalization.

Current defaults use `cointegrated/rubert-tiny2` and FP32; mixed precision is not implemented.
Dataset `v1.1` has empty contexts and no paraphrase clustering. Gold `args` are available,
but span/source labels and populated-context examples still need preparation.

## Folder contents

- [bi_encoder](bi_encoder/README.md) — retrieval model, training, evaluation, and test commands.
- [cross_encoder](cross_encoder/README.md) — pair scoring and hybrid reranking commands.
- [common](common/) — shared training lifecycle, CLI, types, metrics, and checkpoint handling.
- [data.py](data.py) — action catalog and sample loading/validation.
- [translate_experiment](translate_experiment/README.md) — OPUS-MT translation and comparisons.
- [reports](reports/) — baseline summaries; `runs/` stores metrics, predictions, and checkpoints.

## Installation and configuration

Run commands from the repository root:

```bash
source .venv/bin/activate
python -m pip install -r text_to_actions/requirements.txt
```

[Bi-encoder config](bi_encoder/config.json) and [cross-encoder config](cross_encoder/config.json)
each contain `cpu_full`, `cpu_trial`, and `gpu_full` profiles, with `encoder` and `training` sections.
GPU execution requires CUDA-enabled PyTorch and an NVIDIA driver.

## Quick start

### CPU training trials

```bash
python -m text_to_actions.bi_encoder.train.run --profile cpu_trial
python -m text_to_actions.cross_encoder.train.run --profile cpu_trial
```

### Full CPU training

Use all training and validation samples without subset limits:

```bash
python -m text_to_actions.bi_encoder.train.run --profile cpu_full
python -m text_to_actions.cross_encoder.train.run --profile cpu_full
```

### Trained hybrid evaluation

Point `encoder.retriever_config` to the trained bi-encoder's `evaluation_config.json`
before training the cross-encoder. Omit `retriever_profile` when using that saved config.

```bash
python -m text_to_actions.cross_encoder.evaluate.run \
  --config CHECKPOINT/evaluation_config.json --mode hybrid
```

`CHECKPOINT` is a trained cross-encoder's `epoch-XXX` directory.
Model READMEs describe all flags, resume, GPU runs, and Russian catalogs.
