# Installation

## Requirements

- Python `3.10+`
- an OpenAI-compatible HyperCLOVA OmniServe endpoint for summary inference
- OpenAI API key if using the recommended `whisper-1` STT backend

## Main Libraries

- `fastapi`
- `uvicorn`
- `torch`
- `torchaudio`
- `transformers`
- `accelerate`
- `openai`
- `opencv-python`
- `mediapipe`
- `librosa`
- `soundfile`
- `Pillow`

See:

- `requirements.txt`
- `requirements-qwen-omni.txt`

## Environment Setup

Stable environment:

```bash
pip install -r requirements.txt
```

HyperCLOVA runtime path:

```bash
pip install -r requirements-qwen-omni.txt
```

## Required Environment Variables

Recommended runtime:

```bash
export OPENAI_API_KEY=YOUR_KEY
export MUTON_HC_BASE_URL=http://127.0.0.1:8000/b/v1
export MUTON_HC_MODEL_NAME=track_b_model
export MUTON_HC_STT_BACKEND=openai
```

Optional STT tuning:

```bash
export MUTON_STT_MIN_TRANSCRIPT_CONFIDENCE=0.45
export MUTON_STT_SUMMARY_MIN_CONFIDENCE=0.55
```

## Start The Server

```bash
python scripts/run_hyperclovax_server.py
```

Health check:

```bash
curl http://127.0.0.1:5000/health
```

Expected response:

```json
{
  "status": "ok",
  "backend": "hyperclovax_omni"
}
```

## Optional Public Access With Cloudflare

```bash
cloudflared tunnel --url http://127.0.0.1:5000
```

If the tunnel URL changes, update:

```bash
python scripts/update_backend_url.py https://xxxxx.trycloudflare.com
git add backend_url.json
git commit -m "Update backend URL"
git push origin HEAD:server
```
