# embedding.py
# ------------------------------------------------------
# 목적:
# - (얼굴 이미지, 음성 wav, 텍스트 script)을 각각 임베딩으로 바꾼 뒤
# - 하나의 샘플(dict)로 묶어서 리스트로 저장하고
# - 최종적으로 torch.save로 fusion_dataset.pt로 저장하는 데이터셋 생성 스크립트
# ------------------------------------------------------

import os                      # 파일/폴더 경로 조작, 존재 여부 확인 등
import json                    # multi_text.json 로드/파싱
import cv2                     # OpenCV: 얼굴 crop 이미지 로드
import torch                   # 텐서 변환 및 저장(torch.save)
import numpy as np             # 오디오 PCM 처리(형 변환 등)
import soundfile as sf         # wav 파일 로드(scipy 대신 간편)
from transformers import AutoTokenizer, AutoModel  # 텍스트 임베딩용 RoBERTa 로드
from tqdm import tqdm          # 진행바 출력

from muton.encoders import FaceEncoder, AudioEncoder, TextEncoder

# ------------------------------------------------------
# DEVICE 설정:
# - GPU가 있으면 cuda 사용, 없으면 cpu 사용
# - 텍스트 모델은 DEVICE에서 추론
# ------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# Paths (데이터 위치 및 출력 위치)
# ======================================================

FACE_ROOT = "/home/jaesang/p_project/data/face_crops"   # 사람별 얼굴 crop 이미지 폴더 루트 (ex: face_crops/<pid>/<name>.jpg)
AUDIO_ROOT = "/home/jaesang/p_project/data/audio"       # 사람별 wav 파일 폴더 루트 (ex: audio/<pid>/<name>.wav)
JSON_PATH  = "/home/jaesang/p_project/data/multi_text.json"  # 메타데이터/라벨이 들어있는 JSON
OUT_PATH   = "/home/jaesang/p_project/data/fusion_dataset.pt" # 최종 저장될 torch 데이터셋 파일

# ======================================================
# Models (텍스트/얼굴/오디오 인코더 로드)
# ======================================================

print("⚙️ Loading models...")  # 로딩 시작 로그

# 텍스트 토크나이저 로드: 입력 문장을 토큰/어텐션마스크로 변환
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")

# 텍스트 임베딩 모델 로드: klue/roberta-small
# - to(DEVICE)로 GPU/CPU 올리고
# - eval()로 dropout 등 비활성화 (추론 모드)
text_model = AutoModel.from_pretrained(
    "klue/roberta-small"
).to(DEVICE).eval()

# 얼굴 인코더 로드 (내부에서 얼굴 임베딩 768, 감정 logits 7을 생성)
f_encoder = face_encoder.FaceEncoder()

# 오디오 인코더 로드 (내부에서 content/speaker/prosody 각각 768 생성)
a_encoder = audio_encoder.AudioEncoder()

print("✅ Models loaded")  # 로딩 완료 로그

# ======================================================
# Utils (헬퍼 함수들)
# ======================================================

def mean_pool(last_hidden, mask):
    """
    mean pooling:
    - last_hidden: (B, T, H)  transformer의 token-wise hidden states
    - mask:        (B, T)     attention_mask (패딩=0, 유효토큰=1)
    동작:
    - 패딩 토큰은 합산에서 제외하고, 유효 토큰들만 평균내어 문장 임베딩을 만든다.
    """
    mask = mask.unsqueeze(-1).float()          # (B, T) -> (B, T, 1) 로 차원 확장 후 float 변환
    summed = (last_hidden * mask).sum(dim=1)   # 마스크를 곱해 패딩은 0 처리 후, T 차원으로 합산 -> (B, H)
    denom = mask.sum(dim=1).clamp(min=1e-6)    # 유효 토큰 개수(분모). 0 방지 위해 최소값 1e-6으로 클램프 -> (B, 1)
    return summed / denom                      # 평균 -> (B, H)

def get_summary_text(item):
    """
    target_text로 쓸 요약문/설명 텍스트를 item에서 찾는다.
    우선순위 키:
    - "summary_text"
    - "summary_textc"
    - "accessible_emotion_desc"
    조건:
    - 해당 키가 존재하고
    - 문자열이고
    - 공백 제거 후 비어있지 않아야 사용
    """
    for k in ["summary_text", "summary_textc", "accessible_emotion_desc"]:
        if k in item and isinstance(item[k], str) and item[k].strip():  # 유효한 텍스트인지 검사
            return item[k]  # 가장 먼저 발견한 유효 텍스트 반환
    return ""               # 없으면 빈 문자열 반환

# ======================================================
# Main (실제 데이터셋 생성 루프)
# ======================================================

def main():
    # JSON 파일을 열어 전체 메타데이터를 로드
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)  # data는 보통 dict 형태 (예: {"clip_001": [item1, item2, ...], ...})

    dataset = []  # 최종적으로 torch.save로 저장할 샘플 리스트
    print("🚀 Start embedding extraction")  # 시작 로그

    # data.items(): (clip_key, items) 순회
    # - clip_key: "clip_001" 같은 키
    # - items: 해당 클립에 속한 발화/샘플 리스트
    for clip_key, items in tqdm(data.items()):
        # clip_로 시작하는 키만 처리 (그 외 메타 키가 있을 수 있어서 필터)
        if not clip_key.startswith("clip_"):
            continue

        # person_id(pid) 추출:
        # "clip_001" -> "001"
        pid = clip_key.replace("clip_", "").strip()

        # 얼굴/오디오 폴더 경로 구성:
        # face_crops/<pid>, audio/<pid>
        face_dir  = os.path.join(FACE_ROOT, pid)
        audio_dir = os.path.join(AUDIO_ROOT, pid)

        # 둘 중 하나라도 폴더가 없으면 스킵
        if not os.path.isdir(face_dir) or not os.path.isdir(audio_dir):
            continue

        # 클립 내부 item들(샘플들)을 순회
        for item in items:
            # 각 샘플의 name(파일명 stem 역할)을 추출
            name = item.get("name", "").strip()
            if not name:
                continue  # name이 없으면 파일을 찾을 수 없으니 스킵

            # 얼굴 이미지 경로:
            # 기본은 .jpg를 먼저 시도, 없으면 .jpeg 시도
            face_path = os.path.join(face_dir, f"{name}.jpg")
            if not os.path.exists(face_path):
                face_path = os.path.join(face_dir, f"{name}.jpeg")

            # 오디오 wav 경로:
            audio_path = os.path.join(audio_dir, f"{name}.wav")

            # 얼굴/오디오 파일 둘 중 하나라도 없으면 스킵
            if not os.path.exists(face_path) or not os.path.exists(audio_path):
                continue

            # --------------------------------------------------
            # Face (embedding + emotion logits)
            # --------------------------------------------------

            # OpenCV로 이미지 로드 (BGR numpy array)
            img = cv2.imread(face_path)
            if img is None:
                continue  # 로드 실패 시 스킵

            # 얼굴 인코더로 임베딩/감정로짓 추출
            # 반환 형태 예:
            # {"face_vec": np.array(768,), "face_emotion_logits": np.array(7,)}
            face_out = f_encoder.encode_frame(img)
            if face_out is None:
                continue  # 인코더가 얼굴을 못 찾았거나 실패하면 스킵

            # face_vec을 torch tensor로 변환 (float32)
            face_vec = torch.tensor(face_out["face_vec"], dtype=torch.float32)

            # 얼굴 감정 logits(7 클래스)을 torch tensor로 변환
            face_emo_logits = torch.tensor(
                face_out["face_emotion_logits"], dtype=torch.float32
            )

            # shape sanity check:
            # - 얼굴 임베딩은 (768,)
            # - 감정 logits는 (7,)
            if face_vec.shape != (768,) or face_emo_logits.shape != (7,):
                continue

            # --------------------------------------------------
            # Audio (3 × 768)
            # --------------------------------------------------

            # wav 로드 -> pcm (numpy), sr (샘플링레이트)
            pcm, _ = sf.read(audio_path)

            # 스테레오 등 다채널이면 평균으로 mono 변환
            if pcm.ndim > 1:
                pcm = pcm.mean(axis=1)

            # 오디오 인코더로 feature 추출
            # - 입력은 float32 PCM 1D array
            # - 반환 형태 예:
            #   {"content": (768,), "speaker": (768,), "prosody": (768,)}
            feats = a_encoder.extract_features_from_pcm(
                pcm.astype(np.float32)
            )

            # 각 feature를 torch tensor로 변환
            audio_content = torch.tensor(feats["content"], dtype=torch.float32)
            audio_speaker = torch.tensor(feats["speaker"], dtype=torch.float32)
            audio_prosody = torch.tensor(feats["prosody"], dtype=torch.float32)

            # content 임베딩 shape 체크 (나머지도 체크하고 싶으면 추가 가능)
            if audio_content.shape != (768,):
                continue

            # --------------------------------------------------
            # Text (768)
            # --------------------------------------------------

            # 원문 스크립트(발화 텍스트)를 가져옴
            script = item.get("script", "")

            # tokenizer로 텍스트를 텐서로 변환
            # - truncation=True: 길면 자름
            # - padding="max_length": 64 길이로 패딩
            # - max_length=64: 최대 토큰 길이 64
            inputs = tokenizer(
                script,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=64
            ).to(DEVICE)  # input_ids/attention_mask 등을 DEVICE로 이동

            # 텍스트 모델 추론은 gradient 불필요 -> no_grad
            with torch.no_grad():
                # out.last_hidden_state: (B, T, H)
                out = text_model(**inputs, return_dict=True)

                # mean_pool로 문장 임베딩 생성 -> (B, H)
                # squeeze(0)로 배치 차원 제거 -> (H,)
                # cpu()로 CPU로 옮겨 저장(데이터셋 저장 시 GPU 텐서 방지)
                text_vec = mean_pool(
                    out.last_hidden_state,
                    inputs["attention_mask"]
                ).squeeze(0).cpu()

            # --------------------------------------------------
            # Labels (타깃/라벨 구성)
            # --------------------------------------------------

            # target_text로 사용할 요약 텍스트를 찾음
            summary = get_summary_text(item)

            # 요약 텍스트가 없으면 supervised 학습 타깃이 없으므로 스킵
            if not summary:
                continue

            # 최종 샘플(dict) 구성:
            # - id/person_id: 식별용
            # - Inputs: 얼굴/오디오/텍스트 임베딩
            # - Targets: 감정/각성/쾌정도/요약문(생성 타깃)/원문 script
            dataset.append({
                "id": name,                           # 샘플 id (파일명 기반)
                "person_id": pid,                     # 사람(클립) id

                # ===== Inputs =====
                "face_vec": face_vec,                 # 얼굴 임베딩 (768)
                "face_emo_logits": face_emo_logits,   # 얼굴 감정 로짓 (7)
                "audio_content": audio_content,       # 오디오 content 임베딩 (768)
                "audio_speaker": audio_speaker,       # 오디오 speaker 임베딩 (768)
                "audio_prosody": audio_prosody,       # 오디오 prosody 임베딩 (768)
                "text": text_vec,                     # 텍스트 임베딩 (768)

                # ===== Targets =====
                "emotion": item.get("emotion", "Neutral"),  # 범주 감정 라벨(없으면 Neutral)
                "arousal": float(item.get("arousal", 5.0)), # 각성도(없으면 5.0)
                "valence": float(item.get("valence", 5.0)), # 쾌/불쾌(없으면 5.0)
                "target_text": summary,               # 생성 타깃 텍스트(요약문)
                # ===== meta data =====
                "script": script                      # 원문 발화 텍스트(입력 텍스트)
            })

    # dataset(list of dicts)을 torch 파일로 저장
    torch.save(dataset, OUT_PATH)

    # 저장 로그 및 총 샘플 수 출력
    print(f"🎉 Saved dataset → {OUT_PATH}")
    print(f"Total samples: {len(dataset)}")

# 스크립트로 실행될 때만 main()을 호출 (import될 때는 실행 방지)
if __name__ == "__main__":
    main()
