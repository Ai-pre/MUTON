# -*- coding: utf-8 -*-
"""
train_fusion_ko_final.py
- MELD pretrain(best.pt)을 불러와서 backbone 고정(head-only)로
- 한국어 fusion_dataset.pt 전체(약 30개)로 최종 1개 모델(final.pt) 생성

출력 ckpt 포맷:
{
  "model": state_dict,
  "pre_ckpt": <path>,
  "ko_emo2id": {...},
  "args": {...},
  "backbone_cfg": {...},
}
"""

import os
import argparse
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_fusion_meld import FusionTransformer  # 네 MELD pretrain 구조 그대로

# ===== 너 한국어 라벨(소문자) =====
KO_EMO2ID = {
    "angry": 0,
    "dislike": 1,
    "happy": 2,
    "neutral": 3,
    "sad": 4,
    "surprise": 5,
}
KO_ID2EMO = {v: k for k, v in KO_EMO2ID.items()}

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _to_f32(batch, key):
    return torch.from_numpy(np.stack([np.asarray(b[key], dtype=np.float32) for b in batch], axis=0))

def ko_collate_fn(batch):
    face_vec = _to_f32(batch, "face_vec")
    face_emo = _to_f32(batch, "face_emo_logits")
    a_cont  = _to_f32(batch, "audio_content")
    a_spk   = _to_f32(batch, "audio_speaker")
    a_pros  = _to_f32(batch, "audio_prosody")
    text    = _to_f32(batch, "text")

    emo = torch.tensor(
        [KO_EMO2ID.get(str(b["emotion"]).strip().lower(), KO_EMO2ID["neutral"]) for b in batch],
        dtype=torch.long
    )
    arousal = torch.tensor([float(b.get("arousal", 5.0)) for b in batch], dtype=torch.float32)
    valence = torch.tensor([float(b.get("valence", 5.0)) for b in batch], dtype=torch.float32)

    return {
        "face_vec": face_vec,
        "face_emo": face_emo,
        "a_cont": a_cont,
        "a_spk": a_spk,
        "a_pros": a_pros,
        "text": text,
        "emo": emo,
        "arousal": arousal,
        "valence": valence,
    }

class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def freeze_backbone(model: FusionTransformer, unfreeze_last_nlayers: int = 0):
    """
    기본: head만 학습.
    옵션: encoder 마지막 n layer만 풀어 미세조정(데이터 적으면 0 추천)
    """
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if "head_" in name:
            p.requires_grad = True

    if unfreeze_last_nlayers > 0:
        layers = model.encoder.layers
        n = len(layers)
        for li in range(max(0, n - unfreeze_last_nlayers), n):
            for p in layers[li].parameters():
                p.requires_grad = True

@torch.no_grad()
def eval_trainset(model, loader, device):
    model.eval()
    correct = 0
    n = 0
    mse_a = 0.0
    mse_v = 0.0
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)

        emo_logits, a_pred, v_pred = model(batch)
        pred = emo_logits.argmax(dim=-1)
        y = batch["emo"]

        correct += (pred == y).sum().item()
        n += y.numel()

        mse_a += F.mse_loss(a_pred, batch["arousal"], reduction="sum").item()
        mse_v += F.mse_loss(v_pred, batch["valence"], reduction="sum").item()

    acc = correct / max(1, n)
    mse_a = mse_a / max(1, n)
    mse_v = mse_v / max(1, n)
    return acc, mse_a, mse_v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre_ckpt", type=str, required=True, help="MELD pretrain best.pt")
    ap.add_argument("--ko_pt", type=str, required=True, help="Korean fusion_dataset.pt")
    ap.add_argument("--out_pt", type=str, default="out/fusion_ko_final/final.pt")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unfreeze_last_nlayers", type=int, default=0)
    ap.add_argument("--w_emo", type=float, default=1.0)
    ap.add_argument("--w_arousal", type=float, default=0.5)
    ap.add_argument("--w_valence", type=float, default=0.5)
    ap.add_argument("--save_every", type=int, default=0, help=">0이면 n epoch마다 중간 저장")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_pt), exist_ok=True)
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load ko data
    data = torch.load(args.ko_pt)
    print("KO N =", len(data))
    print("KO label dist:", Counter([str(x["emotion"]).strip().lower() for x in data]))

    ds = ListDataset(data)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0, pin_memory=True, collate_fn=ko_collate_fn)

    # ---- load MELD pretrain
    pkg = torch.load(args.pre_ckpt, map_location="cpu")
    cfg = pkg.get("args", {})

    backbone_cfg = {
        "d_model": cfg.get("d_model", 256),
        "nhead": cfg.get("nhead", 8),
        "nlayers": cfg.get("nlayers", 4),
    }

    model = FusionTransformer(
        d_model=backbone_cfg["d_model"],
        nhead=backbone_cfg["nhead"],
        nlayers=backbone_cfg["nlayers"],
        num_emotions=7,
    ).to(device)
    model.load_state_dict(pkg["model"], strict=True)

    # ---- replace head to 6-class
    d_in = model.head_emo.in_features
    model.head_emo = nn.Linear(d_in, len(KO_EMO2ID)).to(device)

    # ---- freeze backbone (head-only by default)
    freeze_backbone(model, unfreeze_last_nlayers=args.unfreeze_last_nlayers)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print("trainable params:", sum(p.numel() for p in trainable))

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    # ---- train
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0

        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

            emo_logits, a_pred, v_pred = model(batch)

            loss_emo = F.cross_entropy(emo_logits, batch["emo"])
            loss_a = F.mse_loss(a_pred, batch["arousal"])
            loss_v = F.mse_loss(v_pred, batch["valence"])

            loss = args.w_emo * loss_emo + args.w_arousal * loss_a + args.w_valence * loss_v

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            total += loss.item()
            steps += 1

        avg_loss = total / max(1, steps)

        # trainset 모니터링(30개라 과적합 체크용)
        acc, mse_a, mse_v = eval_trainset(model, loader, device)

        if epoch == 1 or epoch % 20 == 0:
            print(f"[epoch {epoch}] loss={avg_loss:.4f} train_acc={acc:.4f} mse_a={mse_a:.4f} mse_v={mse_v:.4f}")

        # best는 train loss 기준(데이터 너무 작아서 dev 없으면 이게 현실적)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # 중간 저장 옵션
        if args.save_every and (epoch % args.save_every == 0):
            mid_path = args.out_pt.replace(".pt", f"_e{epoch}.pt")
            torch.save({
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "pre_ckpt": args.pre_ckpt,
                "ko_emo2id": KO_EMO2ID,
                "args": vars(args),
                "backbone_cfg": backbone_cfg,
            }, mid_path)
            print("saved mid:", mid_path)

    # ---- save final(best)
    ckpt = {
        "model": best_state if best_state is not None else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "pre_ckpt": args.pre_ckpt,
        "ko_emo2id": KO_EMO2ID,
        "args": vars(args),
        "backbone_cfg": backbone_cfg,
    }
    torch.save(ckpt, args.out_pt)
    print("SAVED:", args.out_pt)
    print("BEST train_loss:", best_loss)

if __name__ == "__main__":
    main()
