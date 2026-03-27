# MUTON

MUTON is a multimodal dialogue assistance project for conversational understanding from face, audio, and transcript signals.

This branch's current recommended runtime path is:

- multimodal summary and reasoning: `HyperCLOVAX-SEED-Omni-8B` through OmniServe
- speech-to-text: `OpenAI whisper-1` API
- local fallback STT: `ghost613/whisper-large-v3-turbo-korean`
- mobile backend discovery: `server` branch `backend_url.json`

Detailed runtime and troubleshooting notes live in [docs/HYPERCLOVAX_RUNTIME.md](docs/HYPERCLOVAX_RUNTIME.md).

GitHub wiki-ready pages live under `wiki/`, and a minimal API client example lives in `examples/python_api_client.py`.

## Repository Layout

```text
MUTON/
  muton/                      shared config and import compatibility helpers
  scripts/
    run_hyperclovax_server.py FastAPI entrypoint for the current HyperCLOVA demo
    run_qwen_server.py        FastAPI entrypoint for the Qwen path kept for comparison
    run_server.py             legacy fusion server entrypoint
    update_backend_url.py     writes backend_url.json for the Android app
    export_qwen_omni_*.py     JSONL exporters for Qwen2.5-Omni LoRA training
    train_qwen_omni_lora*.py  Qwen2.5-Omni LoRA training wrappers
  src/
    server_hyperclovax.py     current HyperCLOVA summary + STT server
    server_qwen.py            Qwen summary + STT server kept for comparison
    hyperclovax_client.py     OpenAI-compatible HyperCLOVA request helper
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

HyperCLOVA / multimodal runtime path:

```bash
pip install -r requirements-qwen-omni.txt
```

### 2. Run The Recommended Server

Recommended production-ish demo setup:

```bash
export OPENAI_API_KEY=YOUR_KEY
export MUTON_HC_BASE_URL=http://127.0.0.1:8000/b/v1
export MUTON_HC_MODEL_NAME=track_b_model
export MUTON_HC_STT_BACKEND=openai
python scripts/run_hyperclovax_server.py
```

This gives:

- STT: `whisper-1` API with utterance-level filtering
- summary: `HyperCLOVAX-SEED-Omni-8B` through an OpenAI-compatible OmniServe endpoint
- face-only top-bar emotion: 6-class mapped visual label
- utterance-level audio/text snapshot sync before summary generation

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

`src/server_hyperclovax.py` supports two STT modes through `MUTON_HC_STT_BACKEND`.

- `openai`: recommended for the current app demo; restores the earlier Whisper API path with logprob/no-speech filtering
- `whisper`: local Korean Whisper fallback using `ghost613/whisper-large-v3-turbo-korean`

Relevant environment variables:

- `OPENAI_API_KEY`
- `MUTON_HC_BASE_URL`
- `MUTON_HC_MODEL_NAME`
- `MUTON_HC_STT_BACKEND`
- `MUTON_STT_MODEL_NAME`
- `MUTON_STT_MIN_TRANSCRIPT_CONFIDENCE`
- `MUTON_STT_SUMMARY_MIN_CONFIDENCE`

## HyperCLOVA Runtime Notes

The HyperCLOVA path expects an external OmniServe endpoint. `src/server_hyperclovax.py` exposes staged face/audio files through `/media/*` and sends their public URLs to the HyperCLOVA-compatible endpoint.

```bash
export MUTON_HC_BASE_URL=http://127.0.0.1:8000/b/v1
export MUTON_HC_MODEL_NAME=track_b_model
python scripts/run_hyperclovax_server.py
```

If you want the older Qwen training and inference flow, it is still documented in [docs/QWEN_RUNTIME.md](docs/QWEN_RUNTIME.md).

## Notes

- `out/` is treated as generated artifact space and should not be committed.
- The Android app path is now documented against `MUTON`, not `MUTON_cpy`.
- Legacy fusion, seq2seq, and Qwen scripts are kept for comparison, but HyperCLOVA is the recommended runtime path for this branch.
