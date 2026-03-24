# MUTON Wiki

## Project Overview

MUTON is a multimodal dialogue assistance system that combines:

- face-based affect estimation
- utterance-level speech transcription
- multimodal reasoning and summary generation

The current deployment path in this repository is:

- STT: `OpenAI whisper-1`
- multimodal summary: `Qwen2.5-Omni + ko_stage LoRA`
- backend: `FastAPI`
- mobile endpoint discovery: `backend_url.json` on the `server` branch

The system is designed for real-time conversational assistance and currently exposes simple HTTP endpoints that can be used by Android, Python, or any HTTP client.

## Main Features

- real-time face frame ingestion
- utterance-level audio buffering with VAD
- subtitle generation
- multimodal Korean summary generation
- confidence-aware suppression of unreliable summaries

## Important Entry Points

- `scripts/run_qwen_server.py`
- `src/server_qwen.py`
- `src/encoders.py`
- `scripts/update_backend_url.py`

## Recommended Reading Order

1. [Installation](Installation)
2. [API](API)
3. [Examples](Examples)
4. [Architecture](Architecture)
