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
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WavLMModel,
    pipeline,
)


# =========================
# Common
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Face Encoder (?먮낯 face.py 濡쒖쭅 洹몃?濡?
# =========================
@dataclass
class FaceConfig:
    resize_width: int = 256  # ?먮낯???덉뿀??:contentReference[oaicite:4]{index=4}


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
        self.model = AutoModelForImageClassification.from_pretrained(model_id).to(DEVICE).eval()

        self.mar_history = collections.deque(maxlen=5)
        self.MOVEMENT_THRESHOLD = 0.0003  # ?먮낯 洹몃?濡?:contentReference[oaicite:5]{index=5}

    def decode_jpeg(self, jpeg_bytes: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        # ???먮낯泥섎읆 90???뚯쟾 ?좎? :contentReference[oaicite:6]{index=6}
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
        vertical = np.linalg.norm(np.array([top.x, top.y]) - np.array([bottom.x, bottom.y]))
        horizontal = np.linalg.norm(np.array([left.x, left.y]) - np.array([right.x, right.y]))
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

        # [C] crop (bbox + padding, ?먮낯 洹몃?濡? :contentReference[oaicite:7]{index=7}
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

        # ??768 embedding (CLS)
        face_embedding = outputs.hidden_states[-1][:, 0, :].squeeze(0)  # (768,)
        # ??7 logits
        logits = outputs.logits.squeeze(0)  # (7,)

        probs = torch.softmax(logits, dim=0)
        scores = {
            self.model.config.id2label[i].lower(): float(probs[i])
            for i in range(len(probs))
        }

        # 媛먯젙 利앺룺 濡쒖쭅(?먮낯 ?좎?) :contentReference[oaicite:9]{index=9}
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
            "face_vec": face_embedding.detach().cpu().tolist(),          # (768,)
            "face_emotion_logits": logits.detach().cpu().tolist(),       # (7,)
            "emotion": final_emotion,
            "emotion_probs": probs.detach().cpu().tolist(),
            "is_speaking": is_speaking,
            "mar_variance": variance,
            "status": "ok",
        }


# =========================
# Audio Encoder (?먮낯 audio.py 濡쒖쭅 洹몃?濡?+ ?ㅻ쭔 env濡?
# =========================
class AudioEncoder:
    def __init__(self, device: str = DEVICE):
        self.device = device
        self.sample_rate = 16000

        self.audio_buffer = bytearray()
        self.stt_model_name = os.environ.get(
            "MUTON_STT_MODEL_NAME",
            "ghost613/whisper-large-v3-turbo-korean",
        ).strip()
        self.stt_device = os.environ.get("MUTON_STT_DEVICE", self.device).strip()
        self.stt_language = os.environ.get("MUTON_STT_LANGUAGE", "ko").strip()
        self.stt_prompt = os.environ.get(
            "MUTON_STT_PROMPT",
            "Transcribe Korean speech faithfully as subtitles. Output only the spoken utterance.",
        ).strip()
        self.stt_max_new_tokens = int(os.environ.get("MUTON_STT_MAX_NEW_TOKENS", "64").strip())
        stt_dtype_name = os.environ.get(
            "MUTON_STT_TORCH_DTYPE",
            "float16" if self.stt_device.startswith("cuda") else "float32",
        ).strip()
        self.stt_torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(stt_dtype_name, torch.float16 if self.stt_device.startswith("cuda") else torch.float32)

        print(f"Loading Korean Whisper STT ({self.stt_model_name})...")
        self.stt_processor = AutoProcessor.from_pretrained(self.stt_model_name)
        self.stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.stt_model_name,
            torch_dtype=self.stt_torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.stt_device).eval()
        # Older Whisper checkpoints often carry max_length=20 in generation config,
        # which triggers noisy warnings when we drive decoding with max_new_tokens.
        try:
            self.stt_model.generation_config.max_length = None
        except Exception:
            pass
        try:
            self.stt_processor.tokenizer.set_prefix_tokens(
                language=self.stt_language,
                task="transcribe",
            )
        except Exception:
            pass

        pipeline_device = -1
        if self.stt_device.startswith("cuda"):
            pipeline_device = int(self.stt_device.split(":", 1)[1]) if ":" in self.stt_device else 0

        self.stt_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.stt_model,
            tokenizer=self.stt_processor.tokenizer,
            feature_extractor=self.stt_processor.feature_extractor,
            torch_dtype=self.stt_torch_dtype,
            device=pipeline_device,
        )

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

        self.noise_words = [
            "MBC 뉴스",
            "시청해주셔서 감사합니다",
            "구독과 좋아요",
            "알림 설정",
            "YTN",
            "KBS",
            "SBS",
            "유료광고",
            "기자",
            "보도국",
        ]
        self.filler_words = ["음", "어", "아", "그", "저", "으음"]

        print("Loading WavLM-base-plus...")
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(self.device).eval()

    def stt_with_api(self, raw_bytes: bytes) -> Optional[str]:
        transcript, _ = self.consume_buffered_speech(raw_bytes)
        return transcript

    def _filter_transcript(self, raw_text: str) -> Optional[str]:
        if not raw_text:
            return None

        filtered_text = raw_text
        for noise in self.noise_words:
            filtered_text = filtered_text.replace(noise, "")
        for filler in self.filler_words:
            filtered_text = filtered_text.replace(f"{filler} ", "").replace(f" {filler}", "")
            if filtered_text == filler:
                filtered_text = ""

        filtered_text = filtered_text.strip()
        filtered_text = re.sub(r"^[,\.]+", "", filtered_text).strip()
        filtered_text = re.sub(r"\s+", " ", filtered_text)

        repeated_char_match = re.fullmatch(r"(.{1,2})\1{2,}", filtered_text)
        if repeated_char_match:
            return None

        raw_tokens = filtered_text.split()
        if raw_tokens:
            longest_repeat = 1
            current_repeat = 1
            for prev_token, token in zip(raw_tokens, raw_tokens[1:]):
                if token == prev_token:
                    current_repeat += 1
                    longest_repeat = max(longest_repeat, current_repeat)
                else:
                    current_repeat = 1
            if longest_repeat >= 4:
                return None

        tokens = raw_tokens
        if tokens:
            collapsed_tokens: list[str] = []
            prev_token = None
            for token in tokens:
                if token == prev_token:
                    continue
                collapsed_tokens.append(token)
                prev_token = token
            tokens = collapsed_tokens
            filtered_text = " ".join(tokens)

        if tokens:
            max_count = max(tokens.count(token) for token in set(tokens))
            unique_ratio = len(set(tokens)) / max(len(tokens), 1)
            if len(tokens) >= 4 and max_count / len(tokens) >= 0.6:
                return None
            if len(tokens) >= 6 and unique_ratio < 0.4:
                return None

        if any(x in raw_text for x in ["유료광고", "구독", "기자", "뉴스"]):
            return None
        if len(filtered_text) < 2 and not any(c.isalnum() for c in filtered_text):
            return None

        return filtered_text or None

    def _transcribe_waveform(self, waveform: np.ndarray) -> Optional[str]:
        if waveform.size == 0:
            return None

        try:
            result = self.stt_pipe(
                {"array": waveform.astype(np.float32, copy=False), "sampling_rate": self.sample_rate},
                max_length=self.stt_max_new_tokens,
                return_timestamps=False,
            )
        except Exception as e:
            print(f"Local Whisper STT Error: {e}")
            return None

        if isinstance(result, dict):
            return (result.get("text") or "").strip()
        return str(result).strip()

    def consume_buffered_speech(self, raw_bytes: bytes) -> tuple[Optional[str], Optional[np.ndarray]]:
        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        if len(audio_int16) > 0:
            chunk_energy = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2))
        else:
            chunk_energy = 0.0

        if len(self.audio_buffer) == 0 and chunk_energy < self.min_energy_threshold:
            return None, None

        self.audio_buffer.extend(raw_bytes)

        is_speech = False
        if chunk_energy > self.min_energy_threshold:
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            window_size = 512
            for i in range(0, len(audio_float32), window_size):
                chunk = audio_float32[i : i + window_size]
                if len(chunk) < window_size:
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

        min_buffer = 32000
        max_buffer = 320000
        should_send = False

        if len(self.audio_buffer) > min_buffer and self.silence_chunks > self.max_silence_chunks:
            should_send = True
        elif len(self.audio_buffer) > max_buffer:
            should_send = True
            print("Detected: force send (buffer full)")

        if not should_send:
            return None, None

        min_duration_bytes = 25000
        if len(self.audio_buffer) < min_duration_bytes:
            print(f"Discarded because the chunk is too short (size={len(self.audio_buffer)})")
            self.audio_buffer = bytearray()
            self.silence_chunks = 0
            return None, None

        full_buffer_int16 = np.frombuffer(self.audio_buffer, dtype=np.int16)
        full_energy = np.sqrt(np.mean(full_buffer_int16.astype(np.float32) ** 2))
        if full_energy < 300:
            print(f"Discarded because full energy is too low (energy={int(full_energy)})")
            self.audio_buffer = bytearray()
            self.silence_chunks = 0
            return None, None

        utterance_bytes = bytes(self.audio_buffer)
        waveform = np.frombuffer(utterance_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_buffer = bytearray()
        self.silence_chunks = 0

        raw_text = self._transcribe_waveform(waveform)
        filtered_text = self._filter_transcript(raw_text or "")
        if filtered_text:
            print(f"Local Whisper transcript: {filtered_text}")
        return filtered_text, waveform

    # WavLM feature extractor (?먮낯 洹몃?濡? :contentReference[oaicite:10]{index=10}
    def extract_features_from_pcm(self, pcm_np: np.ndarray) -> Dict[str, Any]:
        with torch.no_grad():
            wav = torch.tensor(pcm_np, dtype=torch.float32, device=self.device).unsqueeze(0)
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
                wav_cpu[i: i + frame_len].abs().mean().item()
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
# Text Encoder (embedding.py?먯꽌 ?곕뜕 mean-pool 洹몃?濡??섑븨)
# =========================
class TextEncoder:
    def __init__(self, model_name: str = "klue/roberta-small", device: str = DEVICE, max_length: int = 64):
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
        # 1) 罹먯떆???덉쑝硫?洹멸굅 ?
        if sample_id in cache:
            return cache[sample_id]

        # 2) ?놁쑝硫??쇰떒 ?곸뼱 洹몃?濡?=placeholder)
        # TODO: ?ш린??OpenAI/濡쒖뺄踰덉뿭 ?몄텧濡?諛붽씀硫???
        ko = utterance_en

        cache[sample_id] = ko
        return ko

