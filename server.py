from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from audio import AudioEncoder  # <-- audio.py 불러오기

# ==========================
# FastAPI 설정
# ==========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Android 접근 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Audio Encoder 로드 (딱 1번만)
# ==========================
print("🔊 Loading AudioEncoder... (WavLM + Whisper)")
encoder = AudioEncoder()
print("✓ AudioEncoder Loaded")


# ==========================
# API: 실시간 오디오 청크 처리
# ==========================
@app.post("/process_audio_chunk")
async def process_audio_chunk(audio: UploadFile = File(...)):
    """
    안드로이드에서 보내는 0.5초 WAV 청크를 받아서
    - STT 텍스트
    - prosody 임베딩 (WavLM)
    - content 임베딩 (WavLM)
    - speaker 임베딩 (WavLM)
    을 전부 반환
    """
    file_bytes = await audio.read()

    # audio.py에서 한번에 전체 특징 추출
    feats = encoder.encode_audio_bytes(file_bytes)

    # Torch Tensor → list 변환해서 JSON으로 보낼 수 있게 함
    return {
        "text": feats["text"],                         # Whisper STT
        "prosody": feats["prosody_vec"].tolist(),      # 1024차원
        "content": feats["content_vec"].tolist(),      # 1024차원
        "speaker": feats["speaker_vec"].tolist(),      # 1024차원
    }


# ==========================
# 서버 실행
# ==========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
