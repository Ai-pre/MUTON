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
- The image is also cached as the latest face image for the next multimodal summary step.

## `POST /process_audio_chunk`

### Description

Receives raw PCM audio chunks. The server buffers them until it detects an utterance boundary with VAD, then runs STT.

### Request

- Content-Type: `multipart/form-data`
- Field:
  - `audio`: raw PCM bytes sampled at `16kHz`, mono, `int16`

### Response

```json
{
  "text": "오늘 너무 피곤해.",
  "stt_confidence": 0.82,
  "prosody": [],
  "content": [],
  "speaker": [],
  "fusion_emotion": "",
  "summary": ""
}
```

### Notes

- `text` is empty until the server decides the utterance is complete.
- `stt_confidence` is a server-side confidence score used to suppress unreliable summaries.
- In the current Qwen path, `prosody`, `content`, and `speaker` are placeholders kept for app compatibility.

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
  "summary": "눈을 크게 뜨고 당황한 표정으로 상황을 되묻고 있다.",
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
- Qwen used only for multimodal summary generation
- `whisper-1` used only for STT
