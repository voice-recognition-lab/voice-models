# Voice Models

Research project for converting Russian voice commands into structured actions
using descriptions of available actions and application context. The planned
pipeline uses ASR for transcription and a BERT-based model for action selection
and argument extraction.

## Status

Use of the synthetic text generator is temporarily frozen because generated
commands have insufficient linguistic quality and can disagree with their
action and argument labels. The code and experimental dataset in `data/v1/`
are retained for reproducibility and error analysis.

Dataset version `data/v1.1/` uses Russian examples from MASSIVE and an action
catalog from MASSIVE-Agents.

## Text dataset adapter

Imports Russian examples from MASSIVE and action definitions from MASSIVE-Agents.
[Adapter README](dataset-generator/massive/README.md)

## Synthetic text generator

Generates synthetic Russian voice commands with action labels and arguments.
[Generator README](dataset-generator/text/README.md)

## TTS and ASR evaluation

The ASR/TTS module provides speech synthesis, audio validation, noise
augmentation, transcription with Whisper and GigaAM, and independent evaluation.

- [Module README](dataset-generator/asr-tts/README.md)
- [Data contracts](docs/contracts.md)
- [Synthetic commands experiment report](docs/reports/asr-synthetic-commands-v1.md)

The completed experiment covers 151 commands and 1,359 synthetic WAV files.
WER is 44.09% for Whisper-small and 35.90% for GigaAM-v3. These results are a
preliminary evaluation of the synthetic speech pipeline, not action selection
or execution accuracy.

Synthesized speech can differ from its text reference. Numbers and technical
names remain challenging, and manual review of a few recordings does not
validate the entire dataset.

ASR/TTS code and tests are in `dataset-generator/asr-tts/`, configurations
in `configs/`, and experiment data in `data/v1/`. Model weights, audio,
virtual environments, and runtime caches are excluded from Git.