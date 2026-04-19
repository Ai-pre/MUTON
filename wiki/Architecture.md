# Architecture

## System Evolution

MUTON has two important pipeline stages in its development history.

### P-project

P-project used separate face, audio, and text encoders followed by a directly designed multimodal fusion Transformer. This version proved that a full streaming pipeline could be built and connected to the Android client, but summary quality was still limited by feature compression and a handcrafted generation flow.

### Graduation Project 2

Graduation Project 2 kept the mobile streaming structure but changed the summary engine. After richer sequence experiments, the project moved to a Qwen2.5-Omni based path that accepts raw multimodal inputs more naturally and generates summaries with stronger pretrained multimodal reasoning.

## Figure Placement

Add the pipeline comparison figures in this order:

1. Insert the P-project pipeline figure immediately below this section.
2. Insert the Graduation Project 2 pipeline figure directly below the P-project figure.

If the final wiki layout allows a side-by-side arrangement, that is the best presentation. If not, keep them in vertical order so the transition is easy to read.

## Current Runtime Pipeline

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
- separating the two stages reduces error propagation and makes model replacement easier

## Confidence Handling

The server drops unreliable transcripts before they become summaries.

- low transcript confidence: subtitle suppressed
- low committed confidence: summary returns `Low Confidence`

This reduces garbage summaries caused by noisy audio.
