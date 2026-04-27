# Training

This repository keeps both legacy fusion training scripts and the current Qwen2.5-Omni adaptation path.

## Legacy Fusion Experiments

The P-project and early Graduation Project 2 experiments trained custom fusion and seq2seq structures from extracted face, audio, and text features.

Related scripts:

```text
scripts/train_fusion_seq2seq.py
scripts/train_fusion_seq2seq_two_stage.py
scripts/train_rich_fusion_seq2seq.py
scripts/train_rich_fusion_seq2seq_two_stage.py
src/train_fusion_ko_final.py
src/train_fusion_ko_kfold.py
src/train_fusion_meld.py
```

These files remain in the repository as baselines and as evidence of the project transition from direct fusion modeling to pretrained multimodal generation.

## Qwen2.5-Omni Adaptation

The recommended Graduation Project 2 model path uses Qwen2.5-Omni with LoRA adaptation.

Related scripts:

```text
scripts/train_qwen_omni_lora.py
scripts/train_qwen_omni_lora_two_stage.py
src/train_qwen_omni_lora.py
```

The adapted output is expected through:

```bash
export MUTON_QWEN_ADAPTER=/path/to/out/qwen_omni_lora/ko_stage
```

## Inference Check

Before running the full Android demo, a server-side inference check can be performed with:

```text
scripts/run_qwen_omni_inference.py
```

The live server entrypoint remains:

```text
scripts/run_qwen_server.py
```

## Current Recommendation

For the live mobile demo, use:

- `OpenAI whisper-1` for STT
- `Qwen2.5-Omni + ko_stage LoRA` for multimodal summary
- the legacy fusion scripts only as comparison or report material
