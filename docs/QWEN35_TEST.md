# Qwen3.5-Omni Test Branch

This branch keeps the current `server_main` Android and FastAPI flow, but changes the multimodal summary backend used by `/get_fusion_analysis`.

## Runtime Split

- STT: `OpenAI whisper-1`
- visual emotion UI: existing `FaceEncoder`
- realtime multimodal summary: `qwen3.5-omni-plus` through Alibaba Cloud Model Studio
- conversation record summary after camera stop: `gpt-4o`

## Required Environment Variables

```bash
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
export DASHSCOPE_API_KEY=YOUR_ALIBABA_MODEL_STUDIO_KEY
export MUTON_QWEN_STT_BACKEND=openai
export MUTON_QWEN_SUMMARY_BACKEND=qwen35_api
export MUTON_QWEN35_MODEL_NAME=qwen3.5-omni-plus
export MUTON_QWEN35_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
```

For the Beijing region, use:

```bash
export MUTON_QWEN35_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## Install

Qwen3.5-Omni requires the newer OpenAI-compatible SDK.

```bash
pip install -r requirements.txt
pip install -r requirements-qwen-omni.txt
```

If the existing environment still has an old OpenAI SDK:

```bash
pip install -U "openai>=1.52.0"
```

## Run

```bash
cd ~/MUTON_cpy
git checkout qwen3.5_test
git pull --rebase origin qwen3.5_test

source ~/miniconda3/etc/profile.d/conda.sh
conda activate muton

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
export DASHSCOPE_API_KEY=YOUR_ALIBABA_MODEL_STUDIO_KEY
export MUTON_QWEN_STT_BACKEND=openai
export MUTON_QWEN_SUMMARY_BACKEND=qwen35_api

CUDA_VISIBLE_DEVICES=1 python scripts/run_qwen_server.py
```

Expected startup difference from `server_main`:

```text
Skipping local Qwen2.5-Omni load because MUTON_QWEN_SUMMARY_BACKEND=qwen35_api
```

## Fallback Modes

Use image and audio by default:

```bash
export MUTON_QWEN35_INPUT_MODE=image_audio
```

If the API rejects multi-message image/audio input, test image-only:

```bash
export MUTON_QWEN35_INPUT_MODE=image
```

To return to the old local Qwen2.5 summary path inside this branch:

```bash
export MUTON_QWEN_SUMMARY_BACKEND=local
export MUTON_QWEN_ADAPTER=/home/jaesang02/MUTON_cpy/out/qwen_omni_lora/ko_stage
```
