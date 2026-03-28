# Server/src/muton/encoders.py
from __future__ import annotations

import io
import json
import os
import re
import wave
import collections
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import torch
import cv2
import mediapipe as mp

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
    AutoModelForImageClassification,
    WavLMModel,
)
from openai import OpenAI

from pyannote.audio import Model, Inference, Pipeline
from scipy.spatial.distance import cosine
import torchaudio

# =========================
# Common
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Face Encoder (원본 face.py 로직 그대로)
# =========================
@dataclass
class FaceConfig:
    resize_width: int = 256  # 원본에 있었음 :contentReference[oaicite:4]{index=4}


class FaceEncoder:
    def __init__(self, config: FaceConfig = None):
        self.config = config or FaceConfig()

        print("Loading Face Mesh (MediaPipe)...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        print("Loading Emotion Model (dima806/ViT)...")
        model_id = "dima806/facial_emotions_image_detection"
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        self.model = (
            AutoModelForImageClassification.from_pretrained(model_id).to(DEVICE).eval()
        )

        self.mar_history = collections.deque(maxlen=5)
        self.MOVEMENT_THRESHOLD = (
            0.0003  # 원본 그대로 :contentReference[oaicite:5]{index=5}
        )

    def decode_jpeg(self, jpeg_bytes: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        # ★ 원본처럼 90도 회전 유지 :contentReference[oaicite:6]{index=6}
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def encode_jpeg_bytes(self, jpeg_bytes: bytes) -> Dict[str, Any]:
        frame = self.decode_jpeg(jpeg_bytes)
        if frame is None:
            return {"status": "error", "reason": "decode_failed"}
        result = self.encode_frame(frame)
        if result is None:
            return {"status": "no_face"}
        result["status"] = "ok"
        return result

    def calculate_mar(self, landmarks):
        top = landmarks[13]
        bottom = landmarks[14]
        left = landmarks[61]
        right = landmarks[291]
        vertical = np.linalg.norm(
            np.array([top.x, top.y]) - np.array([bottom.x, bottom.y])
        )
        horizontal = np.linalg.norm(
            np.array([left.x, left.y]) - np.array([right.x, right.y])
        )
        if horizontal == 0:
            return 0.0
        return vertical / horizontal

    def align_face(self, frame, landmarks):
        img_h, img_w = frame.shape[:2]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        l_x, l_y = int(left_eye.x * img_w), int(left_eye.y * img_h)
        r_x, r_y = int(right_eye.x * img_w), int(right_eye.y * img_h)
        dy = r_y - l_y
        dx = r_x - l_x
        angle = np.degrees(np.arctan2(dy, dx))
        center = ((l_x + r_x) // 2, (l_y + r_y) // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, M, (img_w, img_h), flags=cv2.INTER_CUBIC)

    @torch.no_grad()
    def encode_frame(self, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        img_h, img_w = frame_bgr.shape[:2]
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            self.mar_history.clear()
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        # [A] speaking (mouth movement)
        mar = self.calculate_mar(landmarks)
        self.mar_history.append(mar)
        is_speaking = False
        variance = 0.0
        if len(self.mar_history) >= 3:
            variance = float(np.var(list(self.mar_history)))
            if variance > self.MOVEMENT_THRESHOLD:
                is_speaking = True

        # [B] align
        aligned_frame = self.align_face(frame_bgr, landmarks)

        # [C] crop (bbox + padding, 원본 그대로) :contentReference[oaicite:7]{index=7}
        x_list = [l.x for l in landmarks]
        y_list = [l.y for l in landmarks]
        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)

        cx = int((x_min + x_max) / 2 * img_w)
        cy = int((y_min + y_max) / 2 * img_h)
        w = int((x_max - x_min) * img_w)
        h = int((y_max - y_min) * img_h)

        padding = max(w, h) * 0.6
        x1 = max(0, int(cx - w / 2 - padding))
        y1 = max(0, int(cy - h / 2 - padding))
        x2 = min(img_w, int(cx + w / 2 + padding))
        y2 = min(img_h, int(cy + h / 2 + padding))

        face_bgr = aligned_frame[y1:y2, x1:x2]
        if face_bgr is None or face_bgr.size == 0:
            return None

        if face_bgr.shape[0] < 10 or face_bgr.shape[1] < 10:
            return {
                "face_vec": [0.0] * 768,
                "face_emotion_logits": [0.0] * 7,
                "emotion": "Unknown",
                "emotion_probs": [],
                "is_speaking": is_speaking,
                "mar_variance": variance,
                "status": "ok",
            }

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # [D] ViT forward (embedding + logits) :contentReference[oaicite:8]{index=8}
        inputs = self.processor(images=face_rgb, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        # ✅ 768 embedding (CLS)
        face_embedding = outputs.hidden_states[-1][:, 0, :].squeeze(0)  # (768,)
        # ✅ 7 logits
        logits = outputs.logits.squeeze(0)  # (7,)

        probs = torch.softmax(logits, dim=0)
        scores = {
            self.model.config.id2label[i].lower(): float(probs[i])
            for i in range(len(probs))
        }

        # 감정 증폭 로직(원본 유지) :contentReference[oaicite:9]{index=9}
        final_emotion = "Neutral"
        if scores.get("surprise", 0) > 0.15:
            final_emotion = "Surprise"
        elif scores.get("angry", 0) > 0.230:
            final_emotion = "Angry"
        elif scores.get("disgust", 0) > 0.15:
            final_emotion = "Disgust"
        elif scores.get("sad", 0) > 0.20:
            final_emotion = "Sad"
        elif scores.get("fear", 0) > 0.20:
            final_emotion = "Fear"
        else:
            top_emotion = max(scores, key=scores.get)
            label_map = {
                "sad": "Sad",
                "disgust": "Disgust",
                "angry": "Angry",
                "neutral": "Neutral",
                "fear": "Fear",
                "surprise": "Surprise",
                "happy": "Happy",
            }
            final_emotion = label_map.get(top_emotion, top_emotion.capitalize())

        return {
            "face_vec": face_embedding.detach().cpu().tolist(),  # (768,)
            "face_emotion_logits": logits.detach().cpu().tolist(),  # (7,)
            "emotion": final_emotion,
            "emotion_probs": probs.detach().cpu().tolist(),
            "is_speaking": is_speaking,
            "mar_variance": variance,
            "status": "ok",
        }


# =========================
# Audio Encoder (원본 audio.py 로직 그대로 + 키만 env로)
# =========================
class AudioEncoder:
    def __init__(self, device: str = DEVICE):
        self.device = device
        self.sample_rate = 16000

        print("Connecting to OpenAI Whisper API...")
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            # 서버/embedding에서 STT 안 쓰는 경우도 있으니 에러로 죽이지 않고 경고만
            print("OPENAI_API_KEY not set. STT will return None.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)

        self.audio_buffer = bytearray()

        print("Loading Silero VAD...")
        self.vad_model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self.vad_model.to(self.device)

        self.speech_threshold = 0.8
        self.min_energy_threshold = 500
        self.silence_chunks = 0
        self.max_silence_chunks = 4
        self.file_counter = 1

        self.noise_words = [
            "MBC 뉴스",
            "MBC뉴스",
            "시청해주셔서 감사합니다",
            "구독과 좋아요",
            "알림 설정",
            "투데이 별별영상",
            "자막뉴스",
            "YTN",
            "KBS",
            "SBS",
            "유료광고",
            "포함하고 있습니다",
            "주식시황",
            "오늘의 주식",
            "뉴스 스토리",
            "이덕영",
            "기자",
            "보도국",
        ]
        self.filler_words = [
            "음...",
            "음",
            "어...",
            "어",
            "그...",
            "그",
            "아...",
            "아",
            "저...",
            "저",
            "에...",
            "에",
        ]

        print("Loading WavLM-base-plus...")
        self.wavlm = (
            WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
            .to(self.device)
            .eval()
        )

        # [추가] 화자 분리 및 인증 모델 로드
        # your_token에 토큰 추가필요
        self.hf_token = os.environ.get("HF_TOKEN", "your_token")
        print("Loading Pyannote Models...")

        # sim.py 로직: 화자 특징 추출
        self.verify_model = Model.from_pretrained(
            "pyannote/embedding", use_auth_token=self.hf_token
        )
        self.inference = Inference(self.verify_model, window="whole")

        # test.py 로직: 화자 분리 파이프라인
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=self.hf_token
        )
        self.diarization_pipeline.to(torch.device(self.device))

        # 기준 화자 임베딩 (초기에는 None, 나중에 파일로 로드 가능)
        self.target_embedding = None

    # 유사도 검사 기능을 서버용으로 변환
    def set_target_speaker(self, target_wav_path: str):
        """기준이 되는 '나'의 목소리를 등록합니다."""
        self.target_embedding = self.inference(target_wav_path)
        print("Target speaker embedding registered.")

    def analyze_speakers_in_audio(self, wav_path: str):
        """음성 파일 내에서 화자를 분리하고 기준 화자와의 유사도를 측정합니다."""
        if self.target_embedding is None:
            return []

        # 1. 화자 분리 (Diarization)
        diarization = self.diarization_pipeline(wav_path)
        waveform, sr = torchaudio.load(wav_path)

        # 1. 채널 통합 (스테레오 -> 모노)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 2. 볼륨 정규화 (소리 크기를 일정하게 맞춤)
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)

        # 3. 샘플 레이트 강제 고정 (중요!)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000

        results = []
        # 2. 각 화자 구간별로 유사도 체크
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            start_sample = int(turn.start * sr)
            end_sample = int(turn.end * sr)

            # 구간 추출
            segment = waveform[:, start_sample:end_sample]

            # 임시 세그먼트 저장 (pyannote inference는 파일이나 메모리 포맷을 요구함)
            tmp_seg_path = "tmp_seg.wav"
            torchaudio.save(tmp_seg_path, segment, sr)

            # 유사도 계산
            current_emb = self.inference(tmp_seg_path)
            similarity = 1 - cosine(self.target_embedding, current_emb)

            results.append(
                {
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                    "speaker_label": speaker_label,
                    "similarity": round(float(similarity), 4),
                    "is_target": bool(similarity > 0.8),  # sim.py의 임계값 기준
                }
            )

            if os.path.exists(tmp_seg_path):
                os.remove(tmp_seg_path)

        return results

    def stt_with_api(self, raw_bytes: bytes) -> Optional[str]:
        if self.client is None:
            return None

        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        if len(audio_int16) > 0:
            chunk_energy = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))
        else:
            chunk_energy = 0

        if len(self.audio_buffer) == 0 and chunk_energy < self.min_energy_threshold:
            return None

        self.audio_buffer.extend(raw_bytes)

        is_speech = False
        if chunk_energy > self.min_energy_threshold:
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            WINDOW_SIZE = 512
            for i in range(0, len(audio_float32), WINDOW_SIZE):
                chunk = audio_float32[i : i + WINDOW_SIZE]
                if len(chunk) < WINDOW_SIZE:
                    break
                tensor_chunk = torch.from_numpy(chunk).to(self.device).unsqueeze(0)
                speech_prob = self.vad_model(tensor_chunk, 16000).item()
                if speech_prob > self.speech_threshold:
                    is_speech = True
                    break

        if is_speech:
            self.silence_chunks = 0
        else:
            self.silence_chunks += 1

        MIN_BUFFER = 32000
        MAX_BUFFER = 320000
        should_send = False

        if (
            len(self.audio_buffer) > MIN_BUFFER
            and self.silence_chunks > self.max_silence_chunks
        ):
            should_send = True
        elif len(self.audio_buffer) > MAX_BUFFER:
            should_send = True
            print("Detected: 강제 전송 (Buffer Full)")

        if not should_send:
            return None

        MIN_DURATION_BYTES = 25000
        if len(self.audio_buffer) < MIN_DURATION_BYTES:
            print(
                f"Discarded because the chunk is too short (size={len(self.audio_buffer)})"
            )
            self.audio_buffer = bytearray()
            self.silence_chunks = 0
            return None

        full_buffer_int16 = np.frombuffer(self.audio_buffer, dtype=np.int16)
        full_energy = np.sqrt(np.mean(full_buffer_int16.astype(np.float32) ** 2))
        if full_energy < 300:
            print(
                f"Discarded because full energy is too low (energy={int(full_energy)})"
            )
            self.audio_buffer = bytearray()
            self.silence_chunks = 0
            return None

        filename = f"speech_{self.file_counter}.wav"
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(self.audio_buffer)

        self.file_counter += 1
        wav_io.seek(0)
        wav_io.name = filename

        try:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_io,
                language="ko",
                response_format="verbose_json",
                prompt="대화 내용입니다. 핵심 내용만 적으세요.",
                temperature=0.0,
            )

            self.audio_buffer = bytearray()
            self.silence_chunks = 0

            raw_text = (
                getattr(transcript, "text", None)
                or (transcript.get("text") if isinstance(transcript, dict) else "")
                or ""
            ).strip()

            segments = getattr(transcript, "segments", None)
            if segments is None and isinstance(transcript, dict):
                segments = transcript.get("segments", None)

            if not segments:
                return None

            seg0 = segments[0]

            def _get(seg, key, default=None):
                if isinstance(seg, dict):
                    return seg.get(key, default)
                return getattr(seg, key, default)

            avg_logprob = _get(seg0, "avg_logprob", None)
            no_speech_prob = _get(seg0, "no_speech_prob", None)

            if avg_logprob is not None and avg_logprob < -1.0:
                print(
                    f"Discarded because avg_logprob is too low ({avg_logprob:.2f}): {raw_text}"
                )
                return None

            if no_speech_prob is not None and no_speech_prob > 0.8:
                print(
                    f"Discarded because no_speech_prob is too high ({no_speech_prob:.2f}): {raw_text}"
                )
                return None

            if not raw_text:
                return None

            filtered_text = raw_text
            for noise in self.noise_words:
                filtered_text = filtered_text.replace(noise, "")
            for filler in self.filler_words:
                filtered_text = filtered_text.replace(f"{filler} ", "").replace(
                    f" {filler}", ""
                )
                if filtered_text == filler:
                    filtered_text = ""

            filtered_text = filtered_text.strip()
            filtered_text = re.sub(r"^[,\.]+", "", filtered_text).strip()

            if any(x in raw_text for x in ["유료광고", "구독", "기자", "뉴스"]):
                return None
            if len(filtered_text) < 2 and not any(c.isalnum() for c in filtered_text):
                return None

            print(f"API transcript: {filtered_text}")
            return filtered_text

        except Exception as e:
            print(f"OpenAI API Error: {e}")
            self.audio_buffer = bytearray()
            self.silence_chunks = 0
            return None

    # WavLM feature extractor (원본 그대로) :contentReference[oaicite:10]{index=10}
    def extract_features_from_pcm(self, pcm_np: np.ndarray) -> Dict[str, Any]:
        with torch.no_grad():
            wav = torch.tensor(
                pcm_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            out = self.wavlm(wav, output_hidden_states=True)
            hidden = out.last_hidden_state
            T = hidden.size(1)

            content = hidden.mean(dim=1).squeeze(0)

            sec = len(pcm_np) / self.sample_rate
            frames_per_sec = T / sec if sec > 0 else T
            n = int(frames_per_sec)
            if n > T:
                n = T
            speaker = hidden[:, -n:, :].mean(dim=1).squeeze(0)

            wav_cpu = wav.squeeze(0).cpu()
            frame_len = int(0.025 * 16000)
            hop = int(0.010 * 16000)
            energies = [
                wav_cpu[i : i + frame_len].abs().mean().item()
                for i in range(0, wav_cpu.numel() - frame_len, hop)
            ]
            if len(energies) == 0:
                energies = np.ones(1)

            x_old = np.linspace(0, 1, len(energies))
            x_new = np.linspace(0, 1, T)
            e_interp = np.interp(x_new, x_old, energies)
            w = e_interp / (e_interp.sum() + 1e-9)
            w_t = torch.tensor(w, device=self.device).unsqueeze(1)
            prosody = (hidden.squeeze(0) * w_t).sum(dim=0)

            return {
                "prosody": prosody.cpu().numpy().tolist(),
                "content": content.cpu().numpy().tolist(),
                "speaker": speaker.cpu().numpy().tolist(),
            }


# =========================
# Text Encoder (embedding.py에서 쓰던 mean-pool 그대로 래핑)
# =========================
class TextEncoder:
    def __init__(
        self,
        model_name: str = "klue/roberta-small",
        device: str = DEVICE,
        max_length: int = 64,
    ):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        ).to(self.device)

        out = self.model(**inputs, return_dict=True)
        last_hidden = out.last_hidden_state
        mask = inputs["attention_mask"]

        mask_f = mask.unsqueeze(-1).float()
        summed = (last_hidden * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1e-6)
        vec = (summed / denom).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return vec  # (768,)

    @staticmethod
    def load_translate_cache(path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    @staticmethod
    def save_translate_cache(path: str, cache: dict):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @staticmethod
    def get_korean_script(sample_id: str, utterance_en: str, cache: dict) -> str:
        # 1) 캐시에 있으면 그거 씀
        if sample_id in cache:
            return cache[sample_id]

        # 2) 없으면 일단 영어 그대로(=placeholder)
        # TODO: 여기서 OpenAI/로컬번역 호출로 바꾸면 됨
        ko = utterance_en

        cache[sample_id] = ko
        return ko
