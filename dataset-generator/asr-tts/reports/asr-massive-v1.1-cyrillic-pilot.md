# Whisper and GigaAM on a MASSIVE v1.1 synthetic speech pilot

## Scope

Diagnostic pilot: 15 unchanged texts, xenia, clean audio, no noise.
The initial 20-text selection included five texts with Latin characters;
those five were excluded from this pilot and recorded in excluded.json.
Silero failed on standalone olly and iphone. In a separately reviewed
mixed-language recording, the user heard the Russian phrase without olly.
No transliteration or correction of the source dataset was applied.

All 15 pilot WAV passed the technical audio audit. The user listened
to all 15 and reported no problems. The confirmation identifies each WAV
by audio_id and SHA256; it does not apply to other recordings.
Both ASR timing files contain the same verified hashes for these WAV.

## Results

| Model | WAV | WER | CER | RTF |
|---|---:|---:|---:|---:|
| Whisper-small | 15 | 3.66% | 1.26% | 0.5704 |
| GigaAM-v3 | 15 | 3.05% | 0.81% | 0.0887 |

Ordinary evaluation uses the existing normalization without changes.
Time written as digits contributes to WER/CER against a word-form reference.
No new numerical equivalence rules were introduced.

## Observed differences

- Whisper: dorozhno-transportnykh was rendered as two differently inflected
  words; rozovyy became rozovoe; pesnyu became pesni; otmeni became otmenim.
- Whisper wrote five and seven as digits.
- GigaAM wrote the times as 5:00 and 7:00 and rendered Chayf as Chaf.
- Full Russian references and hypotheses are preserved in the artifacts.

## Artifacts

[Pilot files](../experiments/massive-v1.1-cyrillic-pilot/)
include inputs, exclusions, model settings, predictions, timings,
per-record metrics, audio checks, and hash-bound listening confirmation.
WAV and model weights remain outside the published text bundle.

## Limitations

This is a deliberately selected diagnostic subset of 13,387 source texts,
not a representative or complete benchmark of data/v1.1.
It uses one synthetic voice without noise and excludes Latin-containing
texts. Timing describes this CPU run and excludes loading and warmup.
No models were trained or fine-tuned. Earlier experiment results remain unchanged.
