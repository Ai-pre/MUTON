# Examples

## 1. Health Check

```bash
curl http://127.0.0.1:5000/health
```

## 2. Send One Video Frame

```bash
curl -X POST http://127.0.0.1:5000/process_video_chunk \
  -F "frame=@sample.jpg"
```

## 3. Send One Audio Chunk

```bash
curl -X POST http://127.0.0.1:5000/process_audio_chunk \
  -F "audio=@chunk.pcm"
```

## 4. Request Summary

```bash
curl -X POST http://127.0.0.1:5000/get_fusion_analysis \
  -F "text=The speaker sounds upset." \
  -F "prosody=[]" \
  -F "content=[]" \
  -F "speaker=[]"
```

## 5. Python Client Example

See:

- `examples/python_api_client.py`

Example usage:

```bash
python examples/python_api_client.py \
  --base_url http://127.0.0.1:5000 \
  --image sample.jpg \
  --pcm chunk.pcm \
  --text "The speaker sounds upset."
```
