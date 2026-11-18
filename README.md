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

- Текстовый генератор: `dataset-generator/text/` (отдельный PR).
- [TTS, аудиоаудит, шум, ASR и оценка](dataset-generator/asr-tts/README.md).
- [Контракты данных](docs/contracts.md).
- [Отчёт завершённого ASR-эксперимента](docs/reports/zhanna-text-only-v1.0.1.md).

В ASR-эксперименте обработаны 151 команда / 1359 синтетических WAV двумя моделями. WER: Whisper-small — 44,09%, GigaAM-v3 — 35,90%. Это предварительная оценка синтетического тракта, не точности выполнения действий. TTS иногда отличается от эталона; числа и технические имена остаются сложными случаями. Ручная проверка нескольких записей не подтверждает весь набор.

Код и тесты ASR/TTS находятся в `dataset-generator/asr-tts/`, настройки — в `configs/`, данные — в `data/v1/`. Веса, аудио, окружения и runtime-кеш в Git не включаются.
