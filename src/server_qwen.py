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
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel
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
QWEN_DEVICE_MAP_MODE = env_str("MUTON_QWEN_DEVICE_MAP", "single").lower()
QWEN_STT_BACKEND = env_str("MUTON_QWEN_STT_BACKEND", "whisper").lower()
QWEN_STT_MAX_NEW_TOKENS = int(env_str("MUTON_QWEN_STT_MAX_NEW_TOKENS", "128"))
QWEN_STT_USE_ADAPTER = env_str("MUTON_QWEN_STT_USE_ADAPTER", "false").lower() == "true"
CACHE_TTL_SEC = float(env_str("MUTON_CACHE_TTL_SEC", "3.0"))
STT_SUMMARY_MIN_CONFIDENCE = float(env_str("MUTON_STT_SUMMARY_MIN_CONFIDENCE", "0.55"))
QWEN_STT_INSTRUCTION = env_str(
    "MUTON_QWEN_STT_PROMPT",
    "음성 내용을 한국어 자막용 문장으로 정확히 받아써라. 설명하지 말고 전사 결과만 출력해라.",
)
RECORD_SUMMARY_MODEL = env_str("MUTON_RECORD_SUMMARY_MODEL", "gpt-4o-mini")
RECORD_SUMMARY_INSTRUCTION = env_str(
    "MUTON_RECORD_SUMMARY_PROMPT",
    "You summarize conversations in Korean. Return exactly one concise Korean sentence that captures the overall conversation topic and intent. Do not add labels, quotes, bullets, or explanations.",
)
SPEAKER_CLUSTERING_ENABLED = env_str("MUTON_ENABLE_SPEAKER_CLUSTERING", "true").lower() == "true"
SPEAKER_CLUSTER_THRESHOLD = float(env_str("MUTON_SPEAKER_CLUSTER_THRESHOLD", "0.80"))
SPEAKER_CLUSTER_RESET_SEC = float(env_str("MUTON_SPEAKER_CLUSTER_RESET_SEC", "45.0"))
SPEAKER_CLUSTER_MAX_SPEAKERS = int(env_str("MUTON_SPEAKER_CLUSTER_MAX_SPEAKERS", "6"))
SPEAKER_CLUSTER_EMA_ALPHA = float(env_str("MUTON_SPEAKER_CLUSTER_EMA_ALPHA", "0.25"))


def parse_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def resolve_qwen_device_map() -> Any:
    if not torch.cuda.is_available():
        return None
    if QWEN_DEVICE_MAP_MODE == "auto":
        return "auto"
    # Single visible GPU is the stable default for LoRA loading in this repo.
    return {"": 0}


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


class ConversationSummaryRequest(BaseModel):
    conversation_text: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_torch_dtype = parse_torch_dtype(QWEN_DTYPE)
_device_map = resolve_qwen_device_map()
_processor = Qwen2_5OmniProcessor.from_pretrained(QWEN_ADAPTER if Path(QWEN_ADAPTER).exists() else QWEN_MODEL_NAME)
_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    QWEN_MODEL_NAME,
    torch_dtype=_torch_dtype,
    device_map=_device_map,
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
committed_face_image: Image.Image | None = None
committed_audio_waveform: np.ndarray | None = None
committed_transcript = ""
committed_transcript_confidence = 0.0
committed_timestamp = 0.0
last_summary_key: tuple[float, str] | None = None
last_summary_text = ""
_record_summary_client: OpenAI | None = None
speaker_prototypes: list[np.ndarray] = []
speaker_counts: list[int] = []
last_speaker_cluster_timestamp = 0.0


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


def generate_qwen_summary(script: str, face_image: Image.Image | None, audio: np.ndarray | None) -> str:
    messages = build_runtime_messages(face_image, audio, script)
    return _generate_from_messages(messages, max_new_tokens=QWEN_MAX_NEW_TOKENS, use_adapter=True)


def commit_utterance_snapshot(transcript: str, waveform: np.ndarray | None, confidence: float) -> None:
    global committed_face_image, committed_audio_waveform, committed_transcript, committed_transcript_confidence
    global committed_timestamp
    global last_summary_key, last_summary_text

    committed_timestamp = time.time()
    committed_transcript = transcript.strip()
    committed_transcript_confidence = float(confidence)
    committed_audio_waveform = None if waveform is None else np.array(waveform, copy=True)
    committed_face_image = latest_face_image.copy() if latest_face_image is not None else None
    last_summary_key = None
    last_summary_text = ""


def generate_qwen_transcript(audio: np.ndarray) -> str:
    messages = build_stt_messages(audio)
    return _generate_from_messages(
        messages,
        max_new_tokens=QWEN_STT_MAX_NEW_TOKENS,
        use_adapter=QWEN_STT_USE_ADAPTER,
    )


def get_record_summary_client() -> OpenAI:
    global _record_summary_client

    if _record_summary_client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        _record_summary_client = OpenAI(api_key=api_key)
    return _record_summary_client


def _extract_response_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", "") or ""
    if output_text:
        return output_text.strip()

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", "") or ""
            if text:
                return text.strip()
    return ""


def generate_conversation_record_title(conversation_text: str) -> str:
    normalized_text = conversation_text.strip()
    if not normalized_text:
        return ""

    client = get_record_summary_client()
    response = client.responses.create(
        model=RECORD_SUMMARY_MODEL,
        instructions=RECORD_SUMMARY_INSTRUCTION,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": normalized_text,
                    }
                ],
            }
        ],
        max_output_tokens=60,
        truncation="auto",
    )
    return _extract_response_output_text(response)


def _normalize_speaker_embedding(values: list[float] | np.ndarray) -> np.ndarray | None:
    embedding = np.asarray(values, dtype=np.float32)
    if embedding.size == 0:
        return None
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm < 1e-6:
        return None
    return embedding / norm


def reset_speaker_clusters() -> None:
    global speaker_prototypes, speaker_counts, last_speaker_cluster_timestamp

    speaker_prototypes = []
    speaker_counts = []
    last_speaker_cluster_timestamp = 0.0


def assign_speaker_label(utterance_waveform: np.ndarray | None) -> tuple[str, float, int]:
    global last_speaker_cluster_timestamp

    if not SPEAKER_CLUSTERING_ENABLED or utterance_waveform is None or utterance_waveform.size == 0:
        return "", 0.0, len(speaker_prototypes)

    now = time.time()
    if last_speaker_cluster_timestamp and (now - last_speaker_cluster_timestamp) > SPEAKER_CLUSTER_RESET_SEC:
        reset_speaker_clusters()

    try:
        feats = _audio_encoder.extract_features_from_pcm(utterance_waveform.astype(np.float32, copy=False))
    except Exception as exc:
        print(f"Speaker feature extraction error: {exc}")
        return "", 0.0, len(speaker_prototypes)

    embedding = _normalize_speaker_embedding(feats.get("speaker", []))
    if embedding is None:
        return "", 0.0, len(speaker_prototypes)

    if not speaker_prototypes:
        speaker_prototypes.append(embedding.copy())
        speaker_counts.append(1)
        last_speaker_cluster_timestamp = now
        return "speaker_0", 1.0, 1

    similarities = [float(np.dot(proto, embedding)) for proto in speaker_prototypes]
    best_idx = int(np.argmax(similarities))
    best_similarity = similarities[best_idx]

    if best_similarity >= SPEAKER_CLUSTER_THRESHOLD:
        alpha = float(np.clip(SPEAKER_CLUSTER_EMA_ALPHA, 0.0, 1.0))
        updated = (1.0 - alpha) * speaker_prototypes[best_idx] + alpha * embedding
        normalized_updated = _normalize_speaker_embedding(updated)
        if normalized_updated is not None:
            speaker_prototypes[best_idx] = normalized_updated
        speaker_counts[best_idx] += 1
        last_speaker_cluster_timestamp = now
        return f"speaker_{best_idx}", best_similarity, len(speaker_prototypes)

    if len(speaker_prototypes) < SPEAKER_CLUSTER_MAX_SPEAKERS:
        speaker_prototypes.append(embedding.copy())
        speaker_counts.append(1)
        last_speaker_cluster_timestamp = now
        new_idx = len(speaker_prototypes) - 1
        return f"speaker_{new_idx}", best_similarity, len(speaker_prototypes)

    last_speaker_cluster_timestamp = now
    return f"speaker_{best_idx}", best_similarity, len(speaker_prototypes)


def consume_audio_buffer_for_qwen_stt(raw_bytes: bytes) -> tuple[str | None, np.ndarray | None, float]:
    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
    if len(audio_int16) > 0:
        chunk_energy = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))
    else:
        chunk_energy = 0.0

    if len(_audio_encoder.audio_buffer) == 0 and chunk_energy < _audio_encoder.min_energy_threshold:
        return None, None, 0.0

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
    if (
        len(_audio_encoder.audio_buffer) > _audio_encoder.min_buffer_bytes
        and _audio_encoder.silence_chunks > _audio_encoder.max_silence_chunks
    ):
        should_send = True
    elif len(_audio_encoder.audio_buffer) > _audio_encoder.max_buffer_bytes:
        should_send = True

    if not should_send:
        return None, None, 0.0

    if len(_audio_encoder.audio_buffer) < _audio_encoder.min_duration_bytes:
        _audio_encoder.audio_buffer = bytearray()
        _audio_encoder.silence_chunks = 0
        return None, None, 0.0

    full_buffer = bytes(_audio_encoder.audio_buffer)
    full_buffer_int16 = np.frombuffer(full_buffer, dtype=np.int16)
    full_energy = np.sqrt(np.mean(full_buffer_int16.astype(np.float32) ** 2)) if len(full_buffer_int16) > 0 else 0.0
    if full_energy < _audio_encoder.min_full_energy:
        _audio_encoder.audio_buffer = bytearray()
        _audio_encoder.silence_chunks = 0
        return None, None, 0.0

    waveform = pcm_bytes_to_waveform(full_buffer)
    _audio_encoder.audio_buffer = bytearray()
    _audio_encoder.silence_chunks = 0

    try:
        transcript = generate_qwen_transcript(waveform)
    except Exception as exc:
        print(f"Qwen STT error: {exc}")
        return None, waveform, 0.0

    transcript = transcript.strip()
    confidence = _audio_encoder._estimate_transcript_confidence(transcript, waveform, float(full_energy))
    if transcript and confidence < _audio_encoder.min_transcript_confidence:
        print(f"Discarded because Qwen transcript confidence is too low (conf={confidence:.2f}, text={transcript})")
        return None, waveform, confidence
    if not transcript:
        return None, waveform, confidence
    return transcript, waveform, confidence


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "backend": "qwen_omni"}


@app.post("/reset_speaker_clusters")
async def reset_speaker_clusters_endpoint() -> dict[str, Any]:
    cleared = len(speaker_prototypes)
    reset_speaker_clusters()
    return {"status": "ok", "cleared_speakers": cleared}


@app.post("/process_video_chunk")
async def process_video_chunk(frame: UploadFile = File(...)) -> dict[str, Any]:
    global latest_face_image, latest_face_timestamp

    jpeg = await frame.read()
    face_result = _face_encoder.encode_jpeg_bytes(jpeg)
    image, source = get_cached_face_image(jpeg)
    if image is None:
        return {"status": "error", "reason": source}

    latest_face_image = image
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
    speaker_label = ""
    speaker_similarity = 0.0
    known_speakers = len(speaker_prototypes)

    if QWEN_STT_BACKEND == "qwen":
        transcript, utterance_waveform, stt_confidence = consume_audio_buffer_for_qwen_stt(pcm)
        if utterance_waveform is not None:
            latest_audio_waveform = utterance_waveform
            latest_audio_timestamp = time.time()
        if transcript:
            text = transcript
            latest_transcript = text
            commit_utterance_snapshot(text, utterance_waveform, stt_confidence)
            speaker_label, speaker_similarity, known_speakers = assign_speaker_label(utterance_waveform)
    elif QWEN_STT_BACKEND == "openai":
        transcript, utterance_waveform, stt_confidence = _audio_encoder.consume_buffered_speech_openai(pcm)
        if utterance_waveform is not None:
            latest_audio_waveform = utterance_waveform
            latest_audio_timestamp = time.time()
        if transcript:
            text = transcript
            latest_transcript = text
            commit_utterance_snapshot(text, utterance_waveform, stt_confidence)
            speaker_label, speaker_similarity, known_speakers = assign_speaker_label(utterance_waveform)
    else:
        transcript, utterance_waveform, stt_confidence = _audio_encoder.consume_buffered_speech(pcm)
        if utterance_waveform is not None:
            latest_audio_waveform = utterance_waveform
            latest_audio_timestamp = time.time()
        if transcript:
            text = transcript
            latest_transcript = text
            commit_utterance_snapshot(text, utterance_waveform, stt_confidence)
            speaker_label, speaker_similarity, known_speakers = assign_speaker_label(utterance_waveform)

    if text and speaker_label:
        print(
            f"Speaker clustering assigned {speaker_label} "
            f"(sim={speaker_similarity:.3f}, known={known_speakers}) for transcript: {text}"
        )

    return {
        "text": text,
        "stt_confidence": stt_confidence,
        "prosody": [],
        "content": [],
        "speaker": speaker_label,
        "speaker_similarity": speaker_similarity,
        "known_speakers": known_speakers,
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
    global last_summary_key, last_summary_text
    del prosody, content, speaker

    now = time.time()
    if committed_face_image is None or (now - committed_timestamp) > CACHE_TTL_SEC:
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
        summary = generate_qwen_summary(script, committed_face_image, committed_audio_waveform)
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


@app.post("/summarize_conversation_record")
async def summarize_conversation_record(payload: ConversationSummaryRequest) -> dict[str, str]:
    conversation_text = payload.conversation_text.strip()
    if not conversation_text:
        return {"title": ""}

    try:
        title = generate_conversation_record_title(conversation_text)
    except Exception as exc:
        print(f"Conversation record summary error: {exc}")
        return {"title": "", "error": "summary_failed"}

    return {"title": title}


if __name__ == "__main__":
    host = os.environ.get("MUTON_HOST", "0.0.0.0")
    port = int(os.environ.get("MUTON_PORT", "5000"))
    uvicorn.run(app, host=host, port=port, reload=False)
