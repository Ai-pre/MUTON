from __future__ import annotations

import os
import sys
import time
import uuid
import wave
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from muton.config import env_path, env_str
from muton.encoders import AudioEncoder, FaceEncoder
from src.build_rich_dataset import crop_face_bgr
from src.hyperclovax_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_INSTRUCTION,
    create_client,
    request_summary,
)


HC_BASE_URL = env_str("MUTON_HC_BASE_URL", DEFAULT_BASE_URL)
HC_API_KEY = env_str("MUTON_HC_API_KEY", "not-needed")
HC_MODEL_NAME = env_str("MUTON_HC_MODEL_NAME", DEFAULT_MODEL_NAME)
HC_SYSTEM_PROMPT = env_str("MUTON_HC_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
HC_USER_INSTRUCTION = env_str("MUTON_HC_USER_PROMPT", DEFAULT_USER_INSTRUCTION)
HC_MAX_TOKENS = int(env_str("MUTON_HC_MAX_TOKENS", "96"))
HC_STT_BACKEND = env_str(
    "MUTON_HC_STT_BACKEND",
    env_str("MUTON_QWEN_STT_BACKEND", "openai"),
).lower()
CACHE_TTL_SEC = float(env_str("MUTON_CACHE_TTL_SEC", "3.0"))
STT_SUMMARY_MIN_CONFIDENCE = float(env_str("MUTON_STT_SUMMARY_MIN_CONFIDENCE", "0.55"))
MEDIA_CACHE_DIR = env_path("MUTON_HC_MEDIA_CACHE", "out/hyperclovax_media_cache")
MEDIA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def map_visual_emotion_to_ko6(emotion: str) -> str:
    mapping = {
        "Angry": "Angry",
        "Disgust": "Dislike",
        "Happy": "Happy",
        "Neutral": "Neutral",
        "Sad": "Sad",
        "Surprise": "Surprise",
        "Fear": "Unknown",
    }
    return mapping.get(emotion, emotion or "Unknown")


def pcm_bytes_to_waveform(raw_bytes: bytes) -> np.ndarray:
    pcm_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    if pcm_np.size == 0:
        return np.zeros(0, dtype=np.float32)
    return pcm_np / 32768.0


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=str(MEDIA_CACHE_DIR)), name="media")

client = create_client(HC_BASE_URL, HC_API_KEY)
face_encoder = FaceEncoder()
audio_encoder = AudioEncoder()

latest_face_path: Path | None = None
latest_face_timestamp = 0.0
latest_audio_waveform: np.ndarray | None = None
latest_audio_timestamp = 0.0
latest_transcript = ""

committed_face_path: Path | None = None
committed_audio_path: Path | None = None
committed_transcript = ""
committed_transcript_confidence = 0.0
committed_timestamp = 0.0

last_summary_key: tuple[float, str] | None = None
last_summary_text = ""


def stage_face_image(jpeg_bytes: bytes) -> tuple[Path | None, str]:
    frame_bgr = face_encoder.decode_jpeg(jpeg_bytes)
    if frame_bgr is None:
        return None, "decode_failed"

    crop_bgr = crop_face_bgr(face_encoder, frame_bgr)
    if crop_bgr is None:
        image = Image.fromarray(frame_bgr[:, :, ::-1]).convert("RGB")
        source = "full_frame"
    else:
        image = Image.fromarray(crop_bgr[:, :, ::-1]).convert("RGB")
        source = "face_crop"

    image_path = MEDIA_CACHE_DIR / f"face_{uuid.uuid4().hex}.jpg"
    image.save(image_path, format="JPEG", quality=95)
    return image_path, source


def stage_audio_waveform(waveform: np.ndarray | None) -> Path | None:
    if waveform is None or waveform.size == 0:
        return None

    audio_path = MEDIA_CACHE_DIR / f"audio_{uuid.uuid4().hex}.wav"
    pcm_np = np.clip(waveform * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(pcm_np.tobytes())
    return audio_path


def public_media_url(request: Request, path: Path | None) -> str:
    if path is None:
        return ""
    base = str(request.base_url).rstrip("/")
    return f"{base}/media/{path.name}"


def commit_utterance_snapshot(transcript: str, waveform: np.ndarray | None, confidence: float) -> None:
    global committed_face_path, committed_audio_path, committed_transcript, committed_transcript_confidence
    global committed_timestamp, last_summary_key, last_summary_text

    committed_timestamp = time.time()
    committed_transcript = transcript.strip()
    committed_transcript_confidence = float(confidence)
    committed_face_path = latest_face_path
    committed_audio_path = stage_audio_waveform(waveform)
    last_summary_key = None
    last_summary_text = ""


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "backend": "hyperclovax_omni", "hc_base_url": HC_BASE_URL}


@app.post("/process_video_chunk")
async def process_video_chunk(frame: UploadFile = File(...)) -> dict[str, Any]:
    global latest_face_path, latest_face_timestamp

    jpeg = await frame.read()
    face_result = face_encoder.encode_jpeg_bytes(jpeg)
    image_path, source = stage_face_image(jpeg)
    if image_path is None:
        return {"status": "error", "reason": source}

    latest_face_path = image_path
    latest_face_timestamp = time.time()

    emotion = "Unknown"
    if isinstance(face_result, dict) and face_result.get("status") == "ok":
        emotion = str(face_result.get("emotion", "Unknown") or "Unknown")
    emotion = map_visual_emotion_to_ko6(emotion)

    return {
        "status": "ok",
        "image_source": source,
        "emotion": emotion,
    }


@app.post("/process_audio_chunk")
async def process_audio_chunk(audio: UploadFile = File(...)) -> dict[str, Any]:
    global latest_audio_waveform, latest_audio_timestamp, latest_transcript

    pcm = await audio.read()
    text = ""
    stt_confidence = 0.0

    if HC_STT_BACKEND == "openai":
        transcript, utterance_waveform, stt_confidence = audio_encoder.consume_buffered_speech_openai(pcm)
    else:
        transcript, utterance_waveform, stt_confidence = audio_encoder.consume_buffered_speech(pcm)

    if utterance_waveform is not None:
        latest_audio_waveform = utterance_waveform
        latest_audio_timestamp = time.time()

    if transcript:
        text = transcript
        latest_transcript = text
        commit_utterance_snapshot(text, utterance_waveform, stt_confidence)

    return {
        "text": text,
        "stt_confidence": stt_confidence,
        "prosody": [],
        "content": [],
        "speaker": [],
        "fusion_emotion": "",
        "summary": "",
    }


@app.post("/get_fusion_analysis")
async def get_fusion_analysis(
    request: Request,
    text: str = Form(...),
    prosody: str = Form("[]"),
    content: str = Form("[]"),
    speaker: str = Form("[]"),
) -> dict[str, Any]:
    global last_summary_key, last_summary_text
    del prosody, content, speaker

    now = time.time()
    if committed_face_path is None or (now - committed_timestamp) > CACHE_TTL_SEC:
        return {"fusion_emotion": "No Visual Input", "summary": ""}

    script = (text or "").strip() or committed_transcript.strip()
    if not script:
        return {"fusion_emotion": "", "summary": ""}

    if committed_transcript_confidence < STT_SUMMARY_MIN_CONFIDENCE:
        return {
            "fusion_emotion": "Low Confidence",
            "fusion_confidence": committed_transcript_confidence,
            "arousal": 0.0,
            "valence": 0.0,
            "summary": "",
            "cls_attn": [],
        }

    summary_key = (committed_timestamp, script)
    if last_summary_key == summary_key and last_summary_text:
        summary = last_summary_text
    else:
        image_url = public_media_url(request, committed_face_path)
        audio_url = public_media_url(request, committed_audio_path)
        summary = request_summary(
            client=client,
            model_name=HC_MODEL_NAME,
            image_url=image_url,
            audio_url=audio_url,
            script=script,
            system_prompt=HC_SYSTEM_PROMPT,
            instruction=HC_USER_INSTRUCTION,
            max_tokens=HC_MAX_TOKENS,
        )
        last_summary_key = summary_key
        last_summary_text = summary

    return {
        "fusion_emotion": "",
        "fusion_confidence": committed_transcript_confidence,
        "arousal": 0.0,
        "valence": 0.0,
        "summary": summary,
        "cls_attn": [],
    }


if __name__ == "__main__":
    host = os.environ.get("MUTON_HOST", "0.0.0.0")
    port = int(os.environ.get("MUTON_PORT", "5000"))
    uvicorn.run(app, host=host, port=port, reload=False)
