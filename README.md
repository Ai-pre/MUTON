# MUTON

MUTON is a real-time multimodal dialogue assistance project for hearing-impaired users, especially users who rely on oral communication rather than sign language. The project goes beyond plain speech-to-text by combining face, audio, and transcript signals to estimate conversational tone and generate short context-aware summaries.

## Why MUTON Exists

Many captioning tools can tell a user what was said, but not how it was said. In real conversations, emotion, tone, hesitation, tension, and attitude often change the meaning of the same sentence. MUTON was built to reduce that gap by interpreting multimodal cues together instead of relying on transcript text alone.

The project started in P-project as a directly designed multimodal pipeline with separate encoders and a custom fusion Transformer. In Graduation Project 2, the focus shifted from "can we build the whole pipeline ourselves?" to "which architecture works better in a real service setting?" That transition led to richer sequence experiments and finally to a Qwen2.5-Omni based summary path.

## Current Recommended Runtime

- multimodal summary and reasoning: `Qwen2.5-Omni + ko_stage LoRA`
- speech-to-text: `OpenAI whisper-1`
- local fallback STT: `ghost613/whisper-large-v3-turbo-korean`
- backend: `FastAPI`
- mobile endpoint discovery: `backend_url.json` on `server_main`

This split is intentional. The current system uses `whisper-1` for subtitle quality and Qwen2.5-Omni for multimodal reasoning, because that combination behaved most reliably in the mobile demo setting.

## Core Features

- real-time Android-to-server streaming for camera frames and PCM audio
- utterance-level buffering with VAD-based speech segmentation
- subtitle generation from streaming speech
- face-only visual emotion output for the mobile UI
- multimodal Korean summary generation from committed face, audio, and transcript snapshots
- confidence-aware suppression to avoid misleading summaries in noisy environments

## What Changed In Graduation Project 2
<img width="1237" height="395" alt="파이프라인(졸업작품2)" src="https://github.com/user-attachments/assets/9609c160-2fd2-4331-98dd-60bdd73efc45" />

- The summary engine moved from a directly designed fusion Transformer to a pretrained multimodal generator.
- The data pipeline evolved from feature-oriented samples to JSONL message-format samples for multimodal generation.
- STT and summary responsibilities were separated: `whisper-1` handles transcription, while Qwen handles reasoning.
- The runtime now commits one utterance snapshot at a time so transcript, audio, and face stay synchronized during summary generation.

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
pip install -r requirements-qwen-omni.txt
```

### 2. Run The Recommended Server

```bash
export OPENAI_API_KEY=YOUR_KEY
export MUTON_QWEN_ADAPTER=/home/jaesang02/MUTON_cpy/out/qwen_omni_lora/ko_stage
export MUTON_QWEN_STT_BACKEND=openai
CUDA_VISIBLE_DEVICES=1 python scripts/run_qwen_server.py
```

### 3. Expose The Server

```bash
cloudflared tunnel --url http://127.0.0.1:5000
```

### 4. Publish The Current Tunnel URL

```bash
python scripts/update_backend_url.py https://xxxxx.trycloudflare.com
git add backend_url.json
git commit -m "Update backend URL"
git push origin server_main
```

Android should read:

```text
https://raw.githubusercontent.com/Ai-pre/MUTON/server_main/backend_url.json
```

## Repository Guide

```text
MUTON/
  scripts/
    run_qwen_server.py        current FastAPI entrypoint
    run_server.py             legacy fusion server entrypoint
    update_backend_url.py     updates backend_url.json for the Android app
    export_qwen_omni_*.py     JSONL exporters for Qwen training
    train_qwen_omni_lora*.py  LoRA training wrappers
  src/
    server_qwen.py            current Qwen summary + STT server
    encoders.py               face/audio encoders and STT backends
    qwen_omni_dataset.py      dataset builders and media materialization
    server.py                 legacy fusion runtime
    fusion_seq2seq.py         legacy seq2seq experiments
  wiki/                       GitHub wiki-ready documentation
  backend_url.json            tracked Android backend discovery file
```

## Documentation

- wiki home: `wiki/Home.md`
- installation: `wiki/Installation.md`
- API reference: `wiki/API.md`
- architecture and model evolution: `wiki/Architecture.md`
- request examples: `wiki/Examples.md`
- Android client repository: [MUTON-Android](https://github.com/Ai-pre/MUTON-Android)

## Legacy Experiment Note

Legacy fusion and seq2seq experiment files are intentionally kept in the repository because they document the transition from P-project to Graduation Project 2. They are useful as baselines and comparison points, but they are not the recommended runtime path for the current mobile demo.
