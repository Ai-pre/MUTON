# Evaluation

MUTON should be evaluated as both a model pipeline and a real-time mobile service.

## STT Evaluation

STT quality is important because summary generation depends on the finalized transcript.

Recommended checks:

- Korean transcription accuracy
- repeated-token suppression
- hallucination filtering
- behavior in quiet versus noisy environments
- utterance segmentation timing
- latency from speech end to subtitle display

The current recommended STT path is `OpenAI whisper-1`. The local Korean Whisper backend can be used as a comparison path.

## Multimodal Summary Evaluation

The summary should be checked for whether it reflects:

- transcript meaning
- facial expression
- audio tone or urgency
- conversational intent
- Korean fluency
- stability across repeated requests

Useful comparison:

- P-project fusion Transformer output
- Graduation Project 2 Qwen2.5-Omni output
- text-only summary baseline

## Service Evaluation

Because MUTON is a mobile real-time system, service-level evaluation is as important as model quality.

Recommended checks:

- Android-to-server request stability
- Cloudflare Tunnel availability
- response latency for `/process_audio_chunk`
- response latency for `/process_video_chunk`
- response latency for `/get_fusion_analysis`
- behavior when the backend URL changes
- behavior when there is no visual input
- behavior when STT confidence is low

## User-Centered Evaluation

For the final project, user-centered evaluation can focus on whether the output helps users understand conversation context better than plain STT.

Possible survey criteria:

- subtitle readability
- summary usefulness
- emotional/contextual clarity
- perceived latency
- trust in the output
- preference compared with STT-only captions
