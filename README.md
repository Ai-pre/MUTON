# MUTON

MUTON is a multimodal dialogue assistance project for conversational understanding from face, audio, and transcript signals.

This branch's current recommended runtime path is:

- multimodal summary and reasoning: `Qwen2.5-Omni`
- speech-to-text: `OpenAI whisper-1` API
- local fallback STT: `ghost613/whisper-large-v3-turbo-korean`
- mobile backend discovery: `server` branch `backend_url.json`

Detailed runtime and troubleshooting notes live in [docs/QWEN_RUNTIME.md](docs/QWEN_RUNTIME.md).

GitHub wiki-ready pages live under `wiki/`, and a minimal API client example lives in `examples/python_api_client.py`.
;;;;

## Repository Layout

```text
MUTON/
  muton/                      shared config and import compatibility helpers
  scripts/
    run_qwen_server.py        FastAPI entrypoint for the current mobile demo
    run_server.py             legacy fusion server entrypoint
    update_backend_url.py     writes backend_url.json for the Android app
    export_qwen_omni_*.py     JSONL exporters for Qwen2.5-Omni LoRA training
    train_qwen_omni_lora*.py  Qwen2.5-Omni LoRA training wrappers
  src/
    server_qwen.py            current Qwen summary + STT server
    encoders.py               face/audio encoders and STT backends
    qwen_omni_dataset.py      JSONL dataset builders and media materialization
    server.py                 legacy fusion server
    fusion_seq2seq.py         legacy seq2seq experiments
  preprocessing/             asset preparation metadata
  backend_url.json           tracked backend URL file for Android remote config
```

## Quick Start

### 1. Install

Stable project environment:

```bash
pip install -r requirements.txt
```

Qwen2.5-Omni path:

```bash
pip install -r requirements-qwen-omni.txt
```

### 2. Run The Recommended Server

Recommended production-ish demo setup:

```bash
export OPENAI_API_KEY=YOUR_KEY
export MUTON_QWEN_ADAPTER=/home/jaesang02/MUTON_cpy/out/qwen_omni_lora/ko_stage
export MUTON_QWEN_STT_BACKEND=openai
CUDA_VISIBLE_DEVICES=1 python scripts/run_qwen_server.py
```

This gives:

- STT: `whisper-1` API with utterance-level filtering
- summary: `Qwen2.5-Omni + ko_stage LoRA`
- face-only top-bar emotion: 6-class mapped visual label

### 3. Expose The Server

```bash
cloudflared tunnel --url http://127.0.0.1:5000
```

### 4. Publish The Current Tunnel URL For Android

Use a separate `server` worktree/repo checkout and update:

```bash
python scripts/update_backend_url.py https://xxxxx.trycloudflare.com
git add backend_url.json
git commit -m "Update backend URL"
git push origin HEAD:server
```

Android should read:

```text
https://raw.githubusercontent.com/Ai-pre/MUTON/refs/heads/server/backend_url.json
```

## STT Backends

`src/server_qwen.py` supports three STT modes through `MUTON_QWEN_STT_BACKEND`.

- `openai`: recommended for the current app demo; restores the earlier Whisper API path with logprob/no-speech filtering
- `whisper`: local Korean Whisper fallback using `ghost613/whisper-large-v3-turbo-korean`
- `qwen`: Qwen audio transcription path for experiments only

Relevant environment variables:

- `OPENAI_API_KEY`
- `MUTON_QWEN_STT_BACKEND`
- `MUTON_STT_MODEL_NAME`
- `MUTON_STT_MIN_TRANSCRIPT_CONFIDENCE`
- `MUTON_STT_SUMMARY_MIN_CONFIDENCE`

## Qwen2.5-Omni Training Flow

### Export JSONL

```bash
python scripts/export_qwen_omni_meld_dataset.py \
  --input_pt out/meld_train_pseudo_mm.pt \
  --videos_root data/MELD/MELD.Raw/train_splits \
  --meld_csv data/MELD/MELD.Raw/train_sent_emo.csv \
  --out_jsonl out/qwen_omni_meld_train.jsonl \
  --media_root out/qwen_omni_meld_train_media

python scripts/export_qwen_omni_meld_dataset.py \
  --input_pt out/meld_dev_pseudo_mm.pt \
  --videos_root data/MELD/MELD.Raw/dev_splits \
  --meld_csv data/MELD/MELD.Raw/dev_sent_emo.csv \
  --out_jsonl out/qwen_omni_meld_dev.jsonl \
  --media_root out/qwen_omni_meld_dev_media

python scripts/export_qwen_omni_ko_dataset.py \
  --json_path preprocessing/multi_text.json \
  --face_root data/Korea/face_crops \
  --audio_root data/Korea/audio \
  --out_jsonl out/qwen_omni_ko.jsonl
```

### Two-Stage LoRA

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_qwen_omni_lora_two_stage.py \
  --model_name Qwen/Qwen2.5-Omni-7B \
  --stage_a_train_jsonl out/qwen_omni_meld_train.jsonl \
  --stage_a_val_jsonl out/qwen_omni_meld_dev.jsonl \
  --stage_b_train_jsonl out/qwen_omni_ko.jsonl \
  --load_in_4bit \
  --gradient_checkpointing
```

## Notes

- `out/` is treated as generated artifact space and should not be committed.
- The Android app path is now documented against `MUTON`, not `MUTON_cpy`.
- Legacy fusion and seq2seq experiment scripts are kept for comparison, but they are not the recommended runtime path for this branch.
