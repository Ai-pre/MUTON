# Architecture

## Runtime Pipeline

1. the client streams camera frames to `/process_video_chunk`
2. the client streams PCM audio chunks to `/process_audio_chunk`
3. the server buffers audio with VAD until one utterance is considered complete
4. STT produces a transcript
5. the server commits an utterance snapshot:
   - transcript
   - utterance waveform
   - latest face image
6. the client calls `/get_fusion_analysis`
7. Qwen2.5-Omni generates a Korean multimodal summary from the committed snapshot

## Why The Pipeline Is Split

- STT and multimodal summary have different strengths
- `whisper-1` is currently more robust for noisy subtitle transcription
- Qwen2.5-Omni performs better as a multimodal reasoning and summary model

## Confidence Handling

The server drops unreliable transcripts before they become summaries.

- low transcript confidence: subtitle suppressed
- low committed confidence: summary returns `Low Confidence`

This reduces garbage summaries caused by noisy audio.
