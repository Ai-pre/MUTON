# MUTON_cpy

MUTON is a multimodal conversation-assistance research project for people with hearing loss.  
This repository combines face, audio, and text signals to estimate emotion, arousal, valence, and a short context summary.

## Repository Overview

- `src/server.py`: FastAPI inference server
- `src/encoders.py`: face, audio, and text encoder implementations
- `src/train_fusion_meld.py`: MELD pretraining for the fusion model
- `src/train_fusion_ko_final.py`: Korean fine-tuning on the fused dataset
- `src/train_fusion_seq2seq.py`: experimental fusion encoder + text decoder training
- `src/train_fusion_ko_kfold.py`: K-fold evaluation for the Korean dataset
- `src/eval_ko_loocv.py`: leave-one-out evaluation
- `src/Embedding.py`: build `fusion_dataset.pt` from prepared assets
- `preprocessing/`: media extraction scripts
- `scripts/`: stable helper entrypoints for local execution
- `muton/`: shared config and compatibility helpers

## Project Layout

```text
MUTON_cpy/
  muton/
    config.py
    encoders.py
  scripts/
    run_server.py
    build_fusion_dataset.py
  preprocessing/
    extract_audio.py
    extract_auido.py
    extract_face.py
  src/
    Embedding.py
    encoders.py
    server.py
    train_fusion_meld.py
    train_fusion_ko_final.py
    train_fusion_seq2seq.py
    train_fusion_ko_kfold.py
    eval_ko_loocv.py
```

## Environment Setup

Python 3.10+ is recommended.

```bash
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you use the OpenAI-backed transcription or summary flow:

```powershell
$env:OPENAI_API_KEY="your-key"
```

Repository-local paths can be overridden with environment variables. See `.env.example`.

## Important Environment Variables

- `MUTON_VIDEO_ROOT`: input video directory
- `MUTON_FRAME_ROOT`: extracted frame directory
- `MUTON_FACE_ROOT`: cropped face directory
- `MUTON_AUDIO_ROOT`: extracted audio directory
- `MUTON_MULTI_TEXT_JSON`: metadata JSON path
- `MUTON_FUSION_DATASET`: output dataset path
- `MUTON_FUSION_MODEL`: fusion checkpoint path
- `MUTON_SELFATTN_MODEL`: self-attention training output path

## Typical Workflows

### 1. Extract audio

```bash
py -3 preprocessing/extract_audio.py
```

### 2. Extract face crops

```bash
py -3 preprocessing/extract_face.py
```

### 3. Build fusion dataset

```bash
py -3 scripts/build_fusion_dataset.py
```

### 4. Train the Korean fusion head

```bash
py -3 src/train_fusion_ko_final.py --pre_ckpt out/fusion_meld_pretrain_attn/best.pt --ko_pt data/fusion_dataset.pt
```

### 5. Train the experimental fusion-to-text decoder

This path uses the multimodal fusion hidden states directly as encoder memory for a seq2seq decoder, instead of prompting GPT from the predicted class/arousal/valence outputs.

Stage A: align the decoder on the larger MELD split.

```bash
py -3 scripts/train_fusion_seq2seq.py --pre_ckpt out/fusion_ko_final/final.pt --train_pt out/meld_train.pt --val_pt out/meld_dev.pt --emotion_space meld --decoder_model google/mt5-small --out_pt out/fusion_seq2seq/meld_best.pt
```

Stage B: fine-tune on the small Korean fused dataset.

```bash
py -3 scripts/train_fusion_seq2seq.py --pre_ckpt out/fusion_seq2seq/meld_best.pt --train_pt out/fusion_dataset.pt --emotion_space ko --decoder_model google/mt5-small --freeze_fusion_backbone --unfreeze_last_nlayers 1 --out_pt out/fusion_seq2seq/ko_best.pt
```

You can also run both stages in one shot:

```bash
py -3 scripts/train_fusion_seq2seq_two_stage.py
```

Notes:

- `fusion_dataset.pt` and the MELD `.pt` files must include `target_text` entries. `src/Embedding.py` already writes that field when the source JSON contains summary text.
- This is an experimental training path for research iteration. It does not replace the current FastAPI inference server yet.

### 6. Run the FastAPI server

```bash
py -3 scripts/run_server.py
```

## Remote URL Workflow For Android

If you cannot keep a fixed public server address and must use a temporary tunnel URL, keep the app pointed at this GitHub raw file instead of hardcoding the tunnel directly:

```text
https://raw.githubusercontent.com/Ai-pre/MUTON_cpy/server/backend_url.json
```

Update the JSON whenever the tunnel URL changes:

```bash
python scripts/update_backend_url.py https://xxxxx.trycloudflare.com
git add backend_url.json
git commit -m "Update backend URL"
git push origin server
```

The Android app should fetch `backend_url.json`, read the `base_url` field, and use that value as its backend base URL.

## Notes

- This is research code, not a fully packaged production service.
- Some scripts assume prepared datasets and checkpoints already exist.
- The recent cleanup focused on making local development less brittle by removing machine-specific absolute paths from the main workflows.

## Recent Cleanup

- Added a shared `muton` helper package for config/import compatibility
- Replaced main `/home/...` paths with environment-based paths
- Added stable script entrypoints under `scripts/`
- Fixed fusion training utilities to match the current model layout
- Restored a readable README with an updated workflow
