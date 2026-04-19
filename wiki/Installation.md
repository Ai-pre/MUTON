# Installation

## Requirements

- Python `3.10+`
- CUDA-capable GPU for Qwen2.5-Omni summary inference
- OpenAI API key if using the recommended `whisper-1` STT backend
- `cloudflared` if public mobile access is needed

## Main Libraries

- `fastapi`
- `uvicorn`
- `torch`
- `torchaudio`
- `transformers`
- `accelerate`
- `peft`
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

```bash
pip install -r requirements.txt
pip install -r requirements-qwen-omni.txt
```

## Recommended Runtime Variables

```bash
export OPENAI_API_KEY=YOUR_KEY
export MUTON_QWEN_ADAPTER=/home/jaesang02/MUTON_cpy/out/qwen_omni_lora/ko_stage
export MUTON_QWEN_STT_BACKEND=openai
```

Optional STT tuning:

```bash
export MUTON_STT_MIN_TRANSCRIPT_CONFIDENCE=0.45
export MUTON_STT_SUMMARY_MIN_CONFIDENCE=0.55
```

## Start The Server

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_qwen_server.py
```

Health check:

```bash
curl http://127.0.0.1:5000/health
```

Expected response:

```json
{
  "status": "ok",
  "backend": "qwen_omni"
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
git push origin server_main
```

Android should read:

```text
https://raw.githubusercontent.com/Ai-pre/MUTON/server_main/backend_url.json
```
