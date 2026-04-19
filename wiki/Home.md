# MUTON Wiki

## Project Overview

MUTON is a real-time multimodal dialogue assistance system for hearing-impaired users, especially users who depend on oral communication and context reading in daily conversation. The project combines facial cues, speech, and transcript text to produce both subtitles and short context-aware summaries.

The current recommended deployment path in this repository is:

- STT: `OpenAI whisper-1`
- multimodal summary: `Qwen2.5-Omni + ko_stage LoRA`
- backend: `FastAPI`
- mobile endpoint discovery: `backend_url.json` on `server_main`

## From P-project To Graduation Project 2

P-project focused on proving that a full multimodal pipeline could be designed and implemented end to end: separate encoders, a custom fusion model, Android streaming, and server-side inference. Graduation Project 2 keeps the same service goal but shifts the modeling strategy toward stronger real-world quality by comparing richer sequence structures and a pretrained multimodal generation path based on Qwen2.5-Omni.

This repository therefore contains both:

- legacy experimental paths kept for comparison
- the current recommended runtime path used for the mobile demo

## Main Features

- real-time camera frame ingestion
- utterance-level speech buffering with VAD
- subtitle generation
- face-only visual emotion output for the app UI
- multimodal Korean summary generation
- confidence-aware suppression for unreliable summaries

## Where To Place Figures

- Add the P-project vs Graduation Project 2 pipeline comparison image in `Architecture.md`.
- Add mobile demo screenshots after the feature overview in this Home page or in project release notes.

## Important Entry Points

- `scripts/run_qwen_server.py`
- `src/server_qwen.py`
- `src/encoders.py`
- `scripts/update_backend_url.py`

## Recommended Reading Order

1. [Architecture](Architecture)
2. [Installation](Installation)
3. [API](API)
4. [Examples](Examples)
