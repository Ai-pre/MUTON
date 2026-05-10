# Evaluation

MUTON should be evaluated as both a model pipeline and a real-time mobile service.

The repository includes a runtime evaluation script:

```text
scripts/evaluate_runtime_pipeline.py
```

Detailed usage is documented in:

```text
docs/EVALUATION.md
```

The current Graduation Project 2 result summary is documented in:

```text
docs/EVALUATION_RESULTS.md
```

If only MELD Raw mp4 files are available, use:

```text
scripts/prepare_meld_eval_manifest.py
```

This script extracts representative frames, 16kHz WAV audio, translated reference text, and reference summaries into an evaluation manifest.

## STT Evaluation

STT quality is important because summary generation depends on the finalized transcript.

Recommended checks:

- Korean transcription accuracy
- repeated-token suppression
- hallucination filtering
- behavior in quiet versus noisy environments
- utterance segmentation timing
- latency from speech end to subtitle display

The current recommended STT path is `OpenAI whisper-1`. The local Korean Whisper backend can be used as a comparison path.

The evaluation script writes `stt_results.csv` with:

- CER
- WER
- STT confidence
- mean chunk latency
- total STT latency

## Multimodal Summary Evaluation

The summary should be checked for whether it reflects:

- transcript meaning
- facial expression
- audio tone or urgency
- conversational intent
- Korean fluency
- stability across repeated requests

Useful comparison:

- P-project fusion Transformer output
- Graduation Project 2 Qwen2.5-Omni output
- text-only summary baseline

The runtime evaluation endpoint supports:

- `text`
- `text_face`
- `text_audio`
- `full`

It also supports `base` vs `lora` comparison through the `use_adapter` flag.

## Service Evaluation

Because MUTON is a mobile real-time system, service-level evaluation is as important as model quality.

Recommended checks:

- Android-to-server request stability
- Cloudflare Tunnel availability
- response latency for `/process_audio_chunk`
- response latency for `/process_video_chunk`
- response latency for `/get_fusion_analysis`
- behavior when the backend URL changes
- behavior when there is no visual input
- behavior when STT confidence is low

The script writes `summary_results.csv`, `summary_human_eval_template.csv`, `results.json`, and `report.md` under the selected output directory.

## Current Result Summary

The current Qwen2.5-Omni evaluation compares the base model with the `ko_stage` LoRA adapter on 30 samples. ROUGE-L is used only as an auxiliary relative metric because the task is open-ended Korean emotion and situation summarization.

| Model | Input | ROUGE-L F1 | Latency |
|---|---:|---:|---:|
| Qwen2.5-Omni Base | Text + Face + Audio | 0.0265 | 3.3456s |
| Qwen2.5-Omni + LoRA | Text + Face + Audio | 0.1405 | 3.2301s |
| Qwen2.5-Omni + LoRA | Text + Face | 0.1616 | 3.2464s |

Human-style semantic scoring was also used to evaluate emotion reflection, intent reflection, fluency, and faithfulness.

| Model | Emotion | Intent | Fluency | Faithfulness | Total |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Omni Base | 2.77 | 2.90 | 2.17 | 2.90 | 10.73 / 20 |
| Qwen2.5-Omni + LoRA | 4.00 | 4.23 | 4.33 | 4.03 | 16.60 / 20 |

The LoRA-adapted model was selected as the better output in 29 of 30 paired comparisons. This suggests that adaptation mainly improved the output style: shorter, less chatbot-like, and more suitable for MUTON's captioning use case.

## User-Centered Evaluation

For the final project, user-centered evaluation can focus on whether the output helps users understand conversation context better than plain STT.

Possible survey criteria:

- subtitle readability
- summary usefulness
- emotional/contextual clarity
- perceived latency
- trust in the output
- preference compared with STT-only captions
