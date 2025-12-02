from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import cv2
import numpy as np

from audio import AudioEncoder       # audio.py
from face import FaceVideoProcessor  # face.py


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 개발용: 어디서든 접근 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# 전역 인코더 / 비디오 프로세서 로드 (1번만)
# ==========================
print("🔊 Loading AudioEncoder... (WavLM + Whisper)")
encoder = AudioEncoder()
print("✓ AudioEncoder Loaded")

print("🎥 Loading FaceVideoProcessor...")
face_processor = FaceVideoProcessor()
print("✓ FaceVideoProcessor Loaded")


# ==========================
# 1) 단일 프레임 업로드 → 화면에 표시 (/upload_frame)
# ==========================
@app.post("/upload_frame")
async def upload_frame(frame: UploadFile = File(...)):
    """
    PC 또는 안드로이드에서 JPEG 한 장을 보내면
    노트북 화면에 실시간으로 띄워줌.
    """
    img_bytes = await frame.read()

    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"status": "error", "reason": "cannot decode image"}

    cv2.imshow("Phone Camera Stream", img)
    cv2.waitKey(1)

    return {"status": "ok"}


# 안드로이드에서 현재 /process_video_chunk 로 보내고 있으니 alias 추가
@app.post("/process_video_chunk")
async def process_video_chunk(frame: UploadFile = File(...)):
    """
    안드로이드에서 'process_video_chunk'로 보내도
    결국 upload_frame과 같은 처리를 하도록 alias.
    """
    return await upload_frame(frame)


# ==========================
# 2) 동영상 파일 전체 → 1초 단위 프레임 리스트 (/process_face_video)
# ==========================
@app.post("/process_face_video")
async def process_face_video(video: UploadFile = File(...)):
    """
    mp4 같은 동영상 파일 전체를 보내면
    1초마다 프레임을 잘라서 base64(JPEG) 리스트를 반환.
    """
    video_bytes = await video.read()
    frame_b64_list = face_processor.extract_frames_base64(video_bytes)

    return {
        "num_frames": len(frame_b64_list),
        "frames": frame_b64_list,
    }


# ==========================
# 3) 오디오 청크 → STT + 임베딩 (/process_audio_chunk)
# ==========================
@app.post("/process_audio_chunk")
async def process_audio_chunk(audio: UploadFile = File(...)):
    """
    안드로이드에서 올라온 0.5초 WAV 청크를 받아서
    - Whisper STT 텍스트
    - WavLM prosody/content/speaker 임베딩
    을 반환.
    """
    file_bytes = await audio.read()

    feats = encoder.encode_audio_bytes(file_bytes)

    return {
        "text": feats["text"],
        "prosody": feats["prosody_vec"].tolist(),   # 1024차원
        "content": feats["content_vec"].tolist(),   # 1024차원
        "speaker": feats["speaker_vec"].tolist(),   # 1024차원
    }


# ==========================
# 서버 실행
# ==========================
if __name__ == "__main__":
    # 0.0.0.0:5000 에서 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=5000)