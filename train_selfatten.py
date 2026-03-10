# ------------------------------------------------------
# 멀티모달(얼굴/오디오/텍스트) 임베딩을 6개 토큰으로 보고,
# CLS 토큰을 붙인 뒤 Self-Attention(Transformer block)로 융합해서
# 감정 분류(7-class) + arousal/valence 회귀를 동시에 학습하는 코드
# ------------------------------------------------------

import torch                                  # PyTorch 핵심
import torch.nn as nn                         # 신경망 모듈들
import torch.optim as optim                   # 옵티마이저
from torch.utils.data import Dataset, DataLoader, random_split  # Dataset/Loader, 데이터 분할
from muton.encoders import FaceEncoder, AudioEncoder, TextEncoder

# ======================================================
# 1. 설정 (경로/하이퍼파라미터/라벨 매핑)
# ======================================================

DATA_PATH = "/home/jaesang/p_project/data/fusion_dataset.pt"        # embedding.py에서 저장한 데이터셋(pt) 경로
MODEL_SAVE_PATH = "/home/jaesang/p_project/my_transformer_fusion.pth"  # 학습된 모델 가중치 저장 경로

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 있으면 cuda, 없으면 cpu

BATCH_SIZE = 8        # 배치 크기
EPOCHS = 50           # 학습 에폭 수
LR = 5e-5             # 학습률 (AdamW)

# 감정 문자열 라벨을 정수 인덱스로 매핑
# - 학습용 분류 타깃은 0~6 범위
# - dislike는 neutral(0)로 합쳐버린 설정
EMOTION_MAP = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "surprise": 4,
    "fear": 5,
    "disgust": 6,
    "dislike": 0
}

# ======================================================
# 2. Dataset (fusion_dataset.pt에서 샘플을 꺼내 모델 입력/라벨로 반환)
# ======================================================

class FusionDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)     # pt 파일에서 리스트[dict] 형태로 로드 (embedding.py 결과)

    def __len__(self):
        return len(self.data)            # 전체 샘플 개수 반환

    def __getitem__(self, idx):
        item = self.data[idx]            # idx번째 샘플(dict) 가져오기

        # -------- 입력: 얼굴 --------
        face_vec = item["face_vec"]                 # 얼굴 임베딩 (768,)
        face_emo = item["face_emo_logits"]          # 얼굴 감정 logits (7,)

        # -------- 입력: 오디오 --------
        ac  = item["audio_content"]                 # 오디오 content 임베딩 (768,)
        asp = item["audio_speaker"]                 # 오디오 speaker 임베딩 (768,)
        apr = item["audio_prosody"]                 # 오디오 prosody 임베딩 (768,)

        # -------- 입력: 텍스트 --------
        text = item["text"]                         # 텍스트 임베딩 (768,)

        # -------- 라벨: emotion (분류) --------
        # item["emotion"]을 문자열로 가져와 소문자 변환 후 매핑
        # 매핑 실패하면 기본값 0(neutral)
        emo = EMOTION_MAP.get(
            str(item.get("emotion", "neutral")).lower(), 0
        )

        # -------- 라벨: arousal / valence (회귀) --------
        aro = float(item.get("arousal", 5.0))       # 각성도(기본 5.0)
        val = float(item.get("valence", 5.0))       # 쾌/불쾌(기본 5.0)

        # DataLoader가 batch로 묶을 수 있도록 텐서로 변환해서 반환
        return (
            face_vec, face_emo,                    # 얼굴 입력 2개
            ac, asp, apr, text,                    # 오디오 3개 + 텍스트 1개 (총 4개)
            torch.tensor(emo, dtype=torch.long),   # 분류 라벨 (정수)
            torch.tensor(aro, dtype=torch.float),  # 회귀 라벨 (float)
            torch.tensor(val, dtype=torch.float)   # 회귀 라벨 (float)
        )

# ======================================================
# 3. Transformer Block (Self-Attention + FFN + Residual + LayerNorm)
# ======================================================

class FusionBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, dropout=0.1):
        super().__init__()

        # Multi-Head Self-Attention 레이어
        # - embed_dim: 토큰 임베딩 차원(d_model)
        # - num_heads: 헤드 수
        # - batch_first=True: 입력 shape을 (B, T, D)로 받겠다
        # - dropout: 어텐션 내부 dropout
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True,
            dropout=dropout
        )

        # 첫 번째 LayerNorm (Attn residual 뒤)
        self.norm1 = nn.LayerNorm(d_model)

        # 두 번째 LayerNorm (FFN residual 뒤)
        self.norm2 = nn.LayerNorm(d_model)

        # Position-wise FeedForward Network (Transformer의 FFN)
        # - D -> 4D 확장 후 비선형(ReLU) -> Dropout -> D로 축소
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        # residual에 적용할 dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        # - 여기서는 CLS 포함 7토큰이므로 seq_len=7이 됨

        # Self-Attention 수행:
        # Q=K=V=x 로 넣으면 self-attention
        # need_weights=True: attention weight를 반환
        # average_attn_weights=False: head별 weight 유지 -> (B, nhead, T, T) 형태
        attn_out, attn_weights = self.attn(
            x, x, x,
            need_weights=True,
            average_attn_weights=False
        )

        # Residual + Dropout + LayerNorm (Attention 블록)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN 적용
        ffn_out = self.ffn(x)

        # Residual + Dropout + LayerNorm (FFN 블록)
        x = self.norm2(x + self.dropout(ffn_out))

        # 변환된 토큰 시퀀스와 attention weight 반환
        return x, attn_weights

# ======================================================
# 4. Multimodal Transformer (CLS + 6 Tokens 입력)
# ======================================================

class MultimodalTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_classes=7):
        super().__init__()

        # ---- Projection layers ----
        # 각 모달 입력의 원래 차원을 d_model로 투영(projection)해서
        # Transformer가 동일 차원 토큰으로 처리하게 만든다.

        self.face_proj     = nn.Linear(768, d_model)  # 얼굴 임베딩 768 -> d_model
        self.face_emo_proj = nn.Linear(7,   d_model)  # 얼굴 감정 logits 7 -> d_model

        self.audio_c_proj = nn.Linear(768, d_model)   # 오디오 content 768 -> d_model
        self.audio_s_proj = nn.Linear(768, d_model)   # 오디오 speaker 768 -> d_model
        self.audio_p_proj = nn.Linear(768, d_model)   # 오디오 prosody 768 -> d_model

        self.text_proj = nn.Linear(768, d_model)      # 텍스트 임베딩 768 -> d_model

        # ---- CLS token ----
        # 학습 가능한 CLS 토큰 파라미터
        # - (1,1,d_model) 하나를 만들어두고, forward에서 batch 크기만큼 expand해서 사용
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # ---- Transformer block ----
        # 단일 FusionBlock (레이어 1개만 사용)
        self.block = FusionBlock(d_model=d_model, nhead=nhead)

        # ---- Heads ----
        # CLS 출력 벡터를 기반으로 3가지 태스크 헤드
        # 1) emotion 분류 logits
        # 2) arousal 회귀 (스칼라 1개)
        # 3) valence 회귀 (스칼라 1개)
        self.emotion_head = nn.Linear(d_model, num_classes)
        self.arousal_head = nn.Linear(d_model, 1)
        self.valence_head = nn.Linear(d_model, 1)

    def forward(self, fv, fe, ac, asp, apr, t, return_attn=False):
        # fv: (B,768)  얼굴 임베딩
        # fe: (B,7)    얼굴 감정 logits
        # ac: (B,768)  오디오 content
        # asp:(B,768)  오디오 speaker
        # apr:(B,768)  오디오 prosody
        # t:  (B,768)  텍스트 임베딩
        # return_attn: attention weight도 반환할지 여부

        B = fv.size(0)  # 배치 크기

        # 각 모달 입력을 d_model로 projection해서 "토큰"으로 만든다.
        tokens = [
            self.face_proj(fv),          # (B,d)
            self.face_emo_proj(fe),      # (B,d)
            self.audio_c_proj(ac),       # (B,d)
            self.audio_s_proj(asp),      # (B,d)
            self.audio_p_proj(apr),      # (B,d)
            self.text_proj(t)            # (B,d)
        ]

        # 리스트에 있는 6개 (B,d)를 토큰 축으로 stack -> (B,6,d)
        tokens = torch.stack(tokens, dim=1)

        # CLS 토큰을 배치 크기만큼 복제 -> (B,1,d)
        cls = self.cls_token.expand(B, -1, -1)

        # CLS + 6 토큰을 concat -> (B,7,d)
        seq = torch.cat([cls, tokens], dim=1)

        # Transformer block 통과 -> out: (B,7,d)
        # attn_weights: (B, nhead, 7, 7)
        out, attn_weights = self.block(seq)

        # CLS 위치(0번 토큰)의 출력만 뽑아서 대표 벡터로 사용 -> (B,d)
        cls_out = out[:, 0]

        # 3개 헤드로 예측
        pred_emo = self.emotion_head(cls_out)  # (B,7) 분류 logits
        pred_aro = self.arousal_head(cls_out)  # (B,1) arousal
        pred_val = self.valence_head(cls_out)  # (B,1) valence

        # attention weight까지 보고 싶으면 함께 반환
        if return_attn:
            return pred_emo, pred_aro, pred_val, attn_weights

        # 기본 반환: 예측 3종
        return pred_emo, pred_aro, pred_val

# ======================================================
# 5. Train (학습 루프)
# ======================================================

def train():
    dataset = FusionDataset(DATA_PATH)  # 전체 데이터셋 로드

    # 데이터가 충분히 많으면 90%만 train으로 사용 (나머지는 버림)
    # (검증셋을 쓰려면 여기서 _ 대신 val_data를 받아서 평가하면 됨)
    if len(dataset) > 50:
        train_size = int(0.9 * len(dataset))
        train_data, _ = random_split(
            dataset, [train_size, len(dataset) - train_size]
        )
    else:
        train_data = dataset  # 데이터가 적으면 전체를 train으로 사용

    # DataLoader 생성:
    # - shuffle=True로 매 epoch마다 섞어서 학습
    loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 모델 생성 후 DEVICE로 이동
    model = MultimodalTransformer().to(DEVICE)

    # AdamW 옵티마이저 설정
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # 분류 손실: CrossEntropyLoss (logits + 정수 라벨)
    criterion_cls = nn.CrossEntropyLoss()

    # 회귀 손실: MSELoss
    criterion_reg = nn.MSELoss()

    print("🔥 Training multimodal fusion transformer...")  # 학습 시작 로그

    # 에폭 반복
    for epoch in range(EPOCHS):
        model.train()           # 학습 모드
        total_loss = 0.0        # 에폭 누적 loss

        # 배치 반복 (FusionDataset.__getitem__ 반환 순서와 동일)
        for fv, fe, ac, asp, apr, t, labels, aro, val in loader:
            # 입력 텐서들을 DEVICE로 이동
            fv, fe, ac, asp, apr, t = (
                fv.to(DEVICE),
                fe.to(DEVICE),
                ac.to(DEVICE),
                asp.to(DEVICE),
                apr.to(DEVICE),
                t.to(DEVICE)
            )
            

            # 라벨(분류)도 DEVICE로 이동
            labels = labels.to(DEVICE)

            # aro/val은 (B,) 형태로 들어오므로 (B,1)로 reshape
            aro = aro.to(DEVICE).unsqueeze(1)
            val = val.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()  # 이전 step의 gradient 초기화

            # forward -> 예측값 3종
            pred_emo, pred_aro, pred_val = model(
                fv, fe, ac, asp, apr, t
            )

            # 총 loss 구성:
            # - emotion 분류 loss 1.0
            # - arousal MSE 0.5 가중치
            # - valence MSE 0.5 가중치
            loss = (
                criterion_cls(pred_emo, labels)
                + 0.5 * criterion_reg(pred_aro, aro)
                + 0.5 * criterion_reg(pred_val, val)
            )

            loss.backward()      # 역전파로 gradient 계산
            optimizer.step()     # 파라미터 업데이트

            total_loss += loss.item()  # 배치 loss 누적

        # 10에폭마다 평균 loss 출력
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Loss: {total_loss/len(loader):.4f}"
            )

    # 학습 끝나면 모델 가중치 저장(state_dict만 저장)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"✅ Model saved → {MODEL_SAVE_PATH}")  # 저장 완료 로그

# ======================================================
# 엔트리포인트: 파일을 직접 실행할 때만 train() 수행
# ======================================================
if __name__ == "__main__":
    train()
