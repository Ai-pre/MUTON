from __future__ import annotations

from contextlib import nullcontext
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from muton.config import env_path, env_str
from muton.encoders import AudioEncoder, FaceEncoder
from src.build_rich_dataset import crop_face_bgr


QWEN_DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
QWEN_USER_INSTRUCTION = env_str(
    "MUTON_QWEN_USER_PROMPT",
    "얼굴 이미지, 음성, 대사를 함께 참고해서 화자의 감정, 태도, 상황을 한국어 한 문장으로 설명해라.",
)
QWEN_MODEL_NAME = env_str("MUTON_QWEN_MODEL_NAME", "Qwen/Qwen2.5-Omni-7B")
QWEN_ADAPTER = str(env_path("MUTON_QWEN_ADAPTER", "out/qwen_omni_lora/ko_stage"))
QWEN_MAX_NEW_TOKENS = int(env_str("MUTON_QWEN_MAX_NEW_TOKENS", "64"))
QWEN_DTYPE = env_str("MUTON_QWEN_TORCH_DTYPE", "bfloat16")
QWEN_STT_BACKEND = env_str("MUTON_QWEN_STT_BACKEND", "whisper").lower()
QWEN_STT_MAX_NEW_TOKENS = int(env_str("MUTON_QWEN_STT_MAX_NEW_TOKENS", "128"))
QWEN_STT_USE_ADAPTER = env_str("MUTON_QWEN_STT_USE_ADAPTER", "false").lower() == "true"
CACHE_TTL_SEC = float(env_str("MUTON_CACHE_TTL_SEC", "3.0"))
QWEN_STT_INSTRUCTION = env_str(
    "MUTON_QWEN_STT_PROMPT",
    "음성 내용을 한국어 자막용 문장으로 정확히 받아써라. 설명하지 말고 전사 결과만 출력해라.",
)


def parse_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def build_runtime_messages(
    image: Image.Image | None,
    audio: np.ndarray | None,
    script: str,
) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    if audio is not None and audio.size > 0:
        user_content.append({"type": "audio", "audio": audio.astype(np.float32, copy=False)})

    prompt_text = f"{QWEN_USER_INSTRUCTION}\n\n대사: {script.strip()}"
    user_content.append({"type": "text", "text": prompt_text})

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": QWEN_DEFAULT_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


def build_stt_messages(audio: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": QWEN_DEFAULT_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio.astype(np.float32, copy=False)},
                {"type": "text", "text": QWEN_STT_INSTRUCTION},
            ],
        },
    ]


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

_torch_dtype = parse_torch_dtype(QWEN_DTYPE)
_processor = Qwen2_5OmniProcessor.from_pretrained(QWEN_ADAPTER if Path(QWEN_ADAPTER).exists() else QWEN_MODEL_NAME)
_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    QWEN_MODEL_NAME,
    torch_dtype=_torch_dtype,
    device_map="auto",
)
if Path(QWEN_ADAPTER).exists():
    from peft import PeftModel

    _model = PeftModel.from_pretrained(_model, QWEN_ADAPTER)
_model.eval()

_face_encoder = FaceEncoder()
_audio_encoder = AudioEncoder()
_audio_sampling_rate = getattr(_processor.feature_extractor, "sampling_rate", 16000)

latest_face_image: Image.Image | None = None
latest_face_timestamp = 0.0
latest_audio_waveform: np.ndarray | None = None
latest_audio_timestamp = 0.0
latest_transcript = ""


def _generate_from_messages(
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    use_adapter: bool,
) -> str:
    inputs = _processor.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(_model.device) if torch.is_tensor(value) else value for key, value in dict(inputs).items()}

    context = nullcontext()
    if not use_adapter and hasattr(_model, "disable_adapter"):
        context = _model.disable_adapter()

    with context:
        with torch.no_grad():
            generated = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                eos_token_id=_processor.tokenizer.eos_token_id,
                pad_token_id=_processor.tokenizer.pad_token_id or _processor.tokenizer.eos_token_id,
            )

    prompt_len = inputs["input_ids"].shape[1]
    generated_text = _processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)[0]
    for stop_marker in [
        "\nHuman",
        "\nAssistant",
        "Human\n",
        "Assistant\n",
        "Human",
        "Assistant",
    ]:
        if stop_marker in generated_text:
            generated_text = generated_text.split(stop_marker, 1)[0]
    return generated_text.strip()


def get_cached_face_image(jpeg_bytes: bytes) -> tuple[Image.Image | None, str]:
    frame_bgr = _face_encoder.decode_jpeg(jpeg_bytes)
    if frame_bgr is None:
        return None, "decode_failed"

    crop_bgr = crop_face_bgr(_face_encoder, frame_bgr)
    if crop_bgr is None:
        rgb = Image.fromarray(frame_bgr[:, :, ::-1]).convert("RGB")
        return rgb, "full_frame"

    crop_rgb = Image.fromarray(crop_bgr[:, :, ::-1]).convert("RGB")
    return crop_rgb, "face_crop"


def generate_qwen_summary(script: str) -> str:
    face_image = latest_face_image
    audio = latest_audio_waveform
    messages = build_runtime_messages(face_image, audio, script)
    return _generate_from_messages(messages, max_new_tokens=QWEN_MAX_NEW_TOKENS, use_adapter=True)


def generate_qwen_transcript(audio: np.ndarray) -> str:
    messages = build_stt_messages(audio)
    return _generate_from_messages(
        messages,
        max_new_tokens=QWEN_STT_MAX_NEW_TOKENS,
        use_adapter=QWEN_STT_USE_ADAPTER,
    )


def consume_audio_buffer_for_qwen_stt(raw_bytes: bytes) -> tuple[str | None, np.ndarray | None]:
    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    if len(audio_int16) > 0:
        chunk_energy = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))
    else:
        chunk_energy = 0.0

    if len(_audio_encoder.audio_buffer) == 0 and chunk_energy < _audio_encoder.min_energy_threshold:
        return None, None

    _audio_encoder.audio_buffer.extend(raw_bytes)

    is_speech = False
    if chunk_energy > _audio_encoder.min_energy_threshold:
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        window_size = 512
        for i in range(0, len(audio_float32), window_size):
            chunk = audio_float32[i : i + window_size]
            if len(chunk) < window_size:
                break
            tensor_chunk = torch.from_numpy(chunk).to(_audio_encoder.device).unsqueeze(0)
            speech_prob = _audio_encoder.vad_model(tensor_chunk, 16000).item()
            if speech_prob > _audio_encoder.speech_threshold:
                is_speech = True
                break

    if is_speech:
        _audio_encoder.silence_chunks = 0
    else:
        _audio_encoder.silence_chunks += 1

    should_send = False
    if len(_audio_encoder.audio_buffer) > 32000 and _audio_encoder.silence_chunks > _audio_encoder.max_silence_chunks:
        should_send = True
    elif len(_audio_encoder.audio_buffer) > 320000:
        should_send = True

    if not should_send:
        return None, None

    if len(_audio_encoder.audio_buffer) < 25000:
        _audio_encoder.audio_buffer = bytearray()
        _audio_encoder.silence_chunks = 0
        return None, None

    full_buffer = bytes(_audio_encoder.audio_buffer)
    full_buffer_int16 = np.frombuffer(full_buffer, dtype=np.int16)
    full_energy = np.sqrt(np.mean(full_buffer_int16.astype(np.float32) ** 2)) if len(full_buffer_int16) > 0 else 0.0
    if full_energy < 300:
        _audio_encoder.audio_buffer = bytearray()
        _audio_encoder.silence_chunks = 0
        return None, None

    waveform = pcm_bytes_to_waveform(full_buffer)
    _audio_encoder.audio_buffer = bytearray()
    _audio_encoder.silence_chunks = 0

    try:
        transcript = generate_qwen_transcript(waveform)
    except Exception as exc:
        print(f"Qwen STT error: {exc}")
        return None, waveform

    transcript = transcript.strip()
    if not transcript:
        return None, waveform
    return transcript, waveform


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "backend": "qwen_omni"}


@app.post("/process_video_chunk")
async def process_video_chunk(frame: UploadFile = File(...)) -> dict[str, Any]:
    global latest_face_image, latest_face_timestamp

    jpeg = await frame.read()
    image, source = get_cached_face_image(jpeg)
    if image is None:
        return {"status": "error", "reason": source}

    latest_face_image = image
    latest_face_timestamp = time.time()
    return {"status": "ok", "image_source": source}


@app.post("/process_audio_chunk")
async def process_audio_chunk(audio: UploadFile = File(...)) -> dict[str, Any]:
    global latest_audio_waveform, latest_audio_timestamp, latest_transcript

    pcm = await audio.read()
    text = ""

    if QWEN_STT_BACKEND == "qwen":
        transcript, utterance_waveform = consume_audio_buffer_for_qwen_stt(pcm)
        if utterance_waveform is not None:
            latest_audio_waveform = utterance_waveform
            latest_audio_timestamp = time.time()
        if transcript:
            text = transcript
            latest_transcript = text
    else:
        latest_audio_waveform = pcm_bytes_to_waveform(pcm)
        latest_audio_timestamp = time.time()
        text = _audio_encoder.stt_with_api(pcm) or ""
        if text:
            latest_transcript = text

    return {
        "text": text,
        "prosody": [],
        "content": [],
        "speaker": [],
        "fusion_emotion": "",
        "summary": "",
    }


@app.post("/get_fusion_analysis")
async def get_fusion_analysis(
    text: str = Form(...),
    prosody: str = Form("[]"),
    content: str = Form("[]"),
    speaker: str = Form("[]"),
) -> dict[str, Any]:
    del prosody, content, speaker

    now = time.time()
    if latest_face_image is None or (now - latest_face_timestamp) > CACHE_TTL_SEC:
        return {"fusion_emotion": "No Visual Input", "summary": ""}

    script = (text or "").strip() or latest_transcript.strip()
    if not script:
        return {"fusion_emotion": "", "summary": ""}

    if latest_audio_waveform is None or (now - latest_audio_timestamp) > CACHE_TTL_SEC:
        summary = generate_qwen_summary(script)
    else:
        summary = generate_qwen_summary(script)

    return {
        "fusion_emotion": "",
        "fusion_confidence": 0.0,
        "arousal": 0.0,
        "valence": 0.0,
        "summary": summary,
        "cls_attn": [],
    }


if __name__ == "__main__":
    host = os.environ.get("MUTON_HOST", "0.0.0.0")
    port = int(os.environ.get("MUTON_PORT", "5000"))
    uvicorn.run(app, host=host, port=port, reload=False)
