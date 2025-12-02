import io
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torchaudio
import soundfile as sf
from transformers import (
    AutoProcessor,
    WavLMModel,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

# =========================================
# 환경 / 공통 설정
# =========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000


@dataclass
class AudioConfig:
    wavlm_name: str = "microsoft/wavlm-large"
    whisper_name: str = "openai/whisper-small"
    # WavLM에서 어떤 레이어를 어떤 용도로 쓸지
    prosody_layer: int = 8         # 억양/리듬 정보가 잘 나오는 중간 레이어
    speaker_layer: int = 3         # 화자 스타일/목소리 정보
    content_from_last_hidden: bool = True  # content는 last_hidden_state에서 뽑기
    # 길이 줄이기용 (프레임 시퀀스를 전부 쓸지, 평균값만 쓸지)
    return_frame_level: bool = False  # True면 시퀀스도 반환, False면 mean pool만


class AudioEncoder:
    """
    - WavLM으로 prosody / content / speaker embedding 추출
    - Whisper로 STT 텍스트 추출
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()

        # ---- WavLM 로드 ----
        self.wavlm_processor = AutoProcessor.from_pretrained(self.config.wavlm_name)
        self.wavlm_model = WavLMModel.from_pretrained(
            self.config.wavlm_name,
            output_hidden_states=True,
        ).to(DEVICE)
        self.wavlm_model.eval()

        # ---- Whisper 로드 ----
        self.whisper_processor = WhisperProcessor.from_pretrained(self.config.whisper_name)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            self.config.whisper_name
        ).to(DEVICE)
        self.whisper_model.eval()

    # =========================================
    # 오디오 로딩/전처리
    # =========================================
    def _load_audio_from_bytes(self, file_bytes: bytes) -> torch.Tensor:
        """
        bytes → mono 16kHz waveform (torch.Tensor, shape: (T,))
        """
        wav, sr = sf.read(io.BytesIO(file_bytes))  # np.ndarray, shape: (T,) or (T, C)

        # stereo → mono
        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        wav = torch.tensor(wav, dtype=torch.float32)

        # resample to 16k
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(
                wav.unsqueeze(0), sr, SAMPLE_RATE
            ).squeeze(0)

        return wav

    # =========================================
    # WavLM 특징 추출
    # =========================================
    @torch.no_grad()
    def _encode_wavlm(self, wav: torch.Tensor) -> Dict[str, Any]:
        """
        wav: (T,) float32, 16kHz
        return:
            - prosody_vec: (1024,)
            - content_vec: (1024,)
            - speaker_vec: (1024,)
            - (옵션) prosody_seq: (frames, 1024)
            - (옵션) content_seq: (frames, 1024)
        """
        inputs = self.wavlm_processor(
            wav,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).to(DEVICE)

        outputs = self.wavlm_model(**inputs)
        hidden_states = outputs.hidden_states      # tuple[layer] → (1, frames, 1024)
        last_hidden = outputs.last_hidden_state    # (1, frames, 1024)

        # prosody: 중간 레이어 사용 (억양/리듬)
        prosody_seq = hidden_states[self.config.prosody_layer].squeeze(0)  # (frames, 1024)
        prosody_vec = prosody_seq.mean(dim=0)                              # (1024,)

        # content: last_hidden_state 기반 (내용/음소)
        if self.config.content_from_last_hidden:
            content_seq = last_hidden.squeeze(0)       # (frames, 1024)
        else:
            # 원하면 다른 레이어로 바꿀 수도 있음
            content_seq = hidden_states[-2].squeeze(0)

        content_vec = content_seq.mean(dim=0)          # (1024,)

        # speaker: 더 낮은 레이어에서 추출 (화자 스타일/목소리)
        speaker_seq = hidden_states[self.config.speaker_layer].squeeze(0)
        speaker_vec = speaker_seq.mean(dim=0)

        result: Dict[str, Any] = {
            "prosody_vec": prosody_vec,      # torch.Tensor, (1024,)
            "content_vec": content_vec,      # torch.Tensor, (1024,)
            "speaker_vec": speaker_vec,      # torch.Tensor, (1024,)
        }

        if self.config.return_frame_level:
            result["prosody_seq"] = prosody_seq        # (frames, 1024)
            result["content_seq"] = content_seq        # (frames, 1024)
            result["speaker_seq"] = speaker_seq        # (frames, 1024)

        return result

    # =========================================
    # Whisper STT
    # =========================================
    @torch.no_grad()
    def _encode_whisper(self, wav: torch.Tensor) -> Dict[str, Any]:
        """
        wav: (T,) float32, 16kHz
        return:
            - text: str
        """
        # WhisperProcessor는 log-mel 스펙트로그램으로 변환해줌
        inputs = self.whisper_processor(
            wav,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(DEVICE)  # (1, 80, frames)

        generated_ids = self.whisper_model.generate(
            input_features,
            max_new_tokens=64,  # 청크 기준이니까 짧게
        )

        text = self.whisper_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return {
            "text": text.strip(),
        }

    # =========================================
    # 통합 인코딩 API
    # =========================================
    def encode_audio_bytes(self, file_bytes: bytes) -> Dict[str, Any]:
        """
        안드로이드에서 올라온 WAV 청크(bytes)를 넣으면
        - STT 텍스트
        - prosody embedding
        - content embedding
        - speaker embedding
        을 모두 반환.
        """
        wav = self._load_audio_from_bytes(file_bytes)  # (T,)

        wavlm_feats = self._encode_wavlm(wav)
        whisper_feats = self._encode_whisper(wav)

        # Torch 텐서는 나중에 바로 Fusion Transformer에 넣을 거면 그대로 두고,
        # JSON으로 내보낼 거면 .tolist() 해서 변환하면 됨.
        result: Dict[str, Any] = {
            "text": whisper_feats["text"],
            "prosody_vec": wavlm_feats["prosody_vec"],
            "content_vec": wavlm_feats["content_vec"],
            "speaker_vec": wavlm_feats["speaker_vec"],
        }

        if self.config.return_frame_level:
            result["prosody_seq"] = wavlm_feats["prosody_seq"]
            result["content_seq"] = wavlm_feats["content_seq"]
            result["speaker_seq"] = wavlm_feats["speaker_seq"]

        return result

    def encode_audio_file(self, path: str) -> Dict[str, Any]:
        """
        디버깅용: 로컬 wav 파일 경로를 넣어서 바로 결과 뽑기
        """
        with open(path, "rb") as f:
            data = f.read()
        return self.encode_audio_bytes(data)


# =========================================
# 단독 테스트용
# =========================================
if __name__ == "__main__":
    enc = AudioEncoder()
    # 여기에 테스트용 파일 경로 넣어서 확인
    test_path = "test.wav"  # 16kHz mono 아니어도 됨, 자동 변환됨
    out = enc.encode_audio_file(test_path)

    print("STT Text:", out["text"])
    print("Prosody vec shape:", out["prosody_vec"].shape)
    print("Content vec shape:", out["content_vec"].shape)
    print("Speaker vec shape:", out["speaker_vec"].shape)
