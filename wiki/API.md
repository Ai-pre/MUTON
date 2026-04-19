# API

The current backend is implemented in `src/server_qwen.py`.

Base URL example:

```text
http://127.0.0.1:5000
```

FastAPI interactive docs:

```text
http://127.0.0.1:5000/docs
```

## `GET /health`

### Description

Checks whether the backend is alive.

### Response

```json
{
  "status": "ok",
  "backend": "qwen_omni"
}
```

## `POST /process_video_chunk`

### Description

Receives one JPEG frame and updates the latest visual state used by the summary model.

### Request

- Content-Type: `multipart/form-data`
- Field:
  - `frame`: JPEG image file

### Response

```json
{
  "status": "ok",
  "image_source": "face_crop",
  "emotion": "Happy"
}
```

### Notes

- `emotion` is a 6-class mapped visual label for the mobile UI.
- The image is cached as the latest face image for the next summary step.

## `POST /process_audio_chunk`

### Description

Receives raw PCM audio chunks. The server buffers them until an utterance boundary is detected, then runs STT.

### Request

- Content-Type: `multipart/form-data`
- Field:
  - `audio`: raw PCM bytes sampled at `16kHz`, mono, `int16`

### Response

```json
{
  "text": "hello, where are you going now?",
  "stt_confidence": 0.82,
  "prosody": [],
  "content": [],
  "speaker": [],
  "fusion_emotion": "",
  "summary": ""
}
```

### Notes

- `text` is empty until the server considers the utterance complete.
- `stt_confidence` is used to suppress unreliable summaries.
- In the current Qwen runtime path, `prosody`, `content`, and `speaker` remain for app compatibility and are not the main summary inputs.

## `POST /get_fusion_analysis`

### Description

Generates a multimodal Korean summary using the committed utterance snapshot:

- finalized transcript
- utterance waveform
- latest face image at utterance-final time

### Request

- Content-Type: `multipart/form-data`
- Fields:
  - `text`: transcript string
  - `prosody`: JSON string, currently `"[]"`
  - `content`: JSON string, currently `"[]"`
  - `speaker`: JSON string, currently `"[]"`

### Response

Successful case:

```json
{
  "fusion_emotion": "",
  "fusion_confidence": 0.81,
  "arousal": 0.0,
  "valence": 0.0,
  "summary": "The speaker sounds tense and appears to be explaining the situation carefully.",
  "cls_attn": []
}
```

Low-confidence case:

```json
{
  "fusion_emotion": "Low Confidence",
  "fusion_confidence": 0.31,
  "arousal": 0.0,
  "valence": 0.0,
  "summary": "",
  "cls_attn": []
}
```

No visual input case:

```json
{
  "fusion_emotion": "No Visual Input",
  "summary": ""
}
```

## Recommended Runtime Configuration

For the best current mobile-demo behavior:

- `MUTON_QWEN_STT_BACKEND=openai`
- Qwen is used for multimodal summary generation
- `whisper-1` is used for STT
