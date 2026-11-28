# ASR/TTS CPU Trial

The trial used the first 20 commands from `data/v1.1/dataset/samples.jsonl` in their original order, including the commands containing `olly`. Silero synthesized the commands with the `xenia` voice and no added noise. All 20 WAV files passed validation. They contained 31.41 seconds of audio in total, with no clipping.

Whisper small (INT8) and GigaAM v3 (e2e CTC, float32) transcribed all recordings on CPU. Both used the same `offline_asr.py` wrapper to load their pinned model revisions from the local cache.

| Model | WER | CER | RTF | Transcription time |
|---|---:|---:|---:|---:|
| Whisper small | 16.28% | 9.47% | 1.830 | 57.50 s |
| GigaAM v3 | 17.44% | 9.27% | 0.201 | 6.33 s |

WER and CER measure word and character errors. RTF is transcription time divided by audio duration, excluding model loading and warmup. On these 20 commands, Whisper had lower WER, while GigaAM had lower CER and took less time to transcribe the recordings.

Both models omitted `olly` from their transcripts for commands 16 and 19. Listen to the start of these recordings to check whether Silero pronounced it.

- [Command 16 — «olly приглуши светильники в холле»](../dataset/audio/audio_0017_xenia_clean.wav)
- [Command 19 — «olly выключи свет в спальне»](../dataset/audio/audio_0020_xenia_clean.wav)

Per-record metrics are available as JSONL. [Whisper](../whisper/record_metrics.jsonl) · [GigaAM](../gigaam/record_metrics.jsonl)
