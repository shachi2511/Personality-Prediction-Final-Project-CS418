#!/usr/bin/env python3
# train_sentence_context.py
# Sentence-level personality classifier that avoids majority-class collapse.

import os
# Silence the tokenizers parallelism warning when using DataLoader workers/fork
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import json
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, DataCollatorWithPadding

# -------------------------- Constants --------------------------
QUESTION_TEXTS = [
    "I would never audition to be on a game show.",
    "I am not much of a flirt.",
    "I have to psych myself up before I am brave enough to make a phone call.",
    "I would hate living with room mates.",
    "I mostly listen to people in conversations.",
    "I reveal little about myself.",
    "I spend hours alone with my hobbies.",
    "I prefer to eat alone.",
    "I have trouble finding people I want to be friends with.",
    "I prefer to socialize 1 on 1, than with a group.",
    "I sometimes speak so quietly people sometimes have trouble hearing me.",
    "I do not like to get my picture taken.",
    "I can keep a conversation going with anyone about anything.",
    "I want a huge social circle.",
    "I talk to people when waiting in lines.",
    "I act wild and crazy.",
    "I am a bundle of joy.",
    "I love excitement.",
    "I'd like to be in a parade.",
    "I am a flamboyant person.",
    "I am good at making impromptu speeches.",
    "I naturally emerge as a leader.",
    "I am spontaneous.",
    "I would enjoy being a sports team coach.",
    "I have a strong personality.",
    "I am excited by many different activities.",
    "I spend most of my time in fantasy worlds.",
    "I often feel lucky.",
    "I don't make eye contact when I talk with people.",
    "I have a monotone voice.",
    "I am a touchy feely person.",
    "I would like to try bungee jumping.",
    "I tend to be admired by others.",
    "I make big physical movements whenever I get excited.",
    "I am brave.",
    "I am always in the moment.",
    "I am involved with my community.",
    "I am good at entertaining children.",
    "I like formal occasions.",
    "I would have to be lost for a very long time before asking help.",
    "I do not care about sports.",
    "I prefer individual sports to team sports.",
    "My parents know nothing about my love life.",
    "I mostly listen to people in conversations.",
    "I never leave the door to my room open.",
    "I make a lot of hand motions when I talk.",
    "I take lots of pictures of my activities.",
    "When I was a child, I put on fake concerts and plays with my friends.",
    "I really like dancing.",
    "I would have difficulty describing myself to someone.",
    "My life would not make a good story.",
    "I am hesitant to give suggestions.",
    "I tire out quickly.",
    "I never tell people the important things about myself.",
    "I avoid going to unknown places.",
    "Going to the doctor is always awkward for me.",
    "I have not kept up with my old friends over the years.",
    "I have not been joyful for quite some time.",
    "I hate to ask for help.",
    "If I were to die, I would not want there to be a memorial for me.",
    "I hate shopping.",
    "I love to do impressions.",
    "I would be pleased if asked to speak at a funeral.",
    "I would never go to a dance club.",
    "I find it very hard to tell people I find them attractive.",
    "I hate people.",
    "I was an outcast in school.",
    "I would enjoy being a librarian.",
    "I am usually not single.",
    "I am able to stand up for myself.",
    "I would go surfing regularly if I lived on a beach.",
    "I have wanted to be a stand-up comedian.",
    "I am a high status person.",
    "I work out regularly.",
    "I laugh a lot.",
    "I like pranks.",
    "I am happy with my life.",
    "I am never at a loss for words.",
    "I feel healthy and vibrant most of the time.",
    "I love large parties.",
    "I am quiet around strangers.",
    "I don't talk a lot.",
    "I keep in the background.",
    "I don't like to draw attention to myself.",
    "I have little to say.",
    "I often feel blue.",
    "I am not really interested in others.",
    "I make people feel at ease.",
    "I don't mind being the center of attention.",
    "I start conversations.",
    "I talk to a lot of different people at parties.",
]
QUESTION_MAP = {i + 1: q for i, q in enumerate(QUESTION_TEXTS)}
SCORE_TEXT = {1: "strongly disagree", 2: "disagree", 3: "neutral", 4: "agree", 5: "strongly agree"}
IE_MAP = {1: "Introvert", 2: "Extravert", 3: "Ambivert"}

LABEL2ID = {"Introvert": 0, "Extravert": 1, "Ambivert": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# -------------------------- Utils --------------------------
def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_item_weights(df: pd.DataFrame) -> Dict[int, float]:
    """
    Diagnosticity of each question wrt I/E using Spearman correlation on IE∈{1,2}.
    Returns weights in ~[0.5, 1.5] (neutral 1.0 if signal is weak).
    """
    weights = {}
    df_ie = df[df["IE"].isin([1, 2])].copy()
    if df_ie.empty:
        return {i: 1.0 for i in range(1, 92)}
    y = df_ie["IE"].map({1: 0, 2: 1})
    for i in range(1, 92):
        col = f"Q{i}A"
        if col not in df_ie.columns:
            weights[i] = 1.0
            continue
        s = df_ie[col]
        mask = s.notna()
        if mask.sum() < 50:
            weights[i] = 1.0
            continue
        rho = pd.Series(s[mask].astype(float)).corr(pd.Series(y[mask].astype(float)), method="spearman")
        weights[i] = float(0.8 + 0.4 * min(1.0, abs(rho))) if (rho is not None and not np.isnan(rho)) else 1.0
    return weights


def build_sentence_rows(df: pd.DataFrame,
                        item_w: Dict[int, float],
                        teacher_user_probs: Optional[np.ndarray] = None,
                        user_order: Optional[List[int]] = None):
    """
    Build sentence-level rows with:
      text, label_id, user_id, q_idx, sample_weight, (optional) teacher_probs
    """
    rows = []
    if teacher_user_probs is not None and user_order is None:
        raise ValueError("If teacher_user_probs is given, user_order must be provided.")

    for uid, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="Users→sentences")):
        label = IE_MAP.get(getattr(row, "IE", None), None)
        if not label:
            continue
        label_id = LABEL2ID[label]
        soft = None
        if teacher_user_probs is not None:
            soft = teacher_user_probs[user_order.index(uid)]
        for q in range(1, 92):
            col = f"Q{q}A"
            score = getattr(row, col, None)
            if score is None or (isinstance(score, float) and np.isnan(score)):
                continue
            s = int(score)
            text = f"{QUESTION_MAP[q]} – {SCORE_TEXT.get(s, 'neutral')}."
            r = {
                "user_id": uid,
                "text": text,
                "label_id": label_id,
                "q_idx": q,
                "sample_weight": float(item_w.get(q, 1.0))
            }
            if soft is not None:
                r["t_probs"] = [float(soft[LABEL2ID["Introvert"]]),
                                float(soft[LABEL2ID["Extravert"]]),
                                float(soft[LABEL2ID["Ambivert"]])]
            rows.append(r)
    return pd.DataFrame(rows)


@torch.no_grad()
def teacher_user_softprobs(df: pd.DataFrame, teacher_dir: str, device, batch_size: int = 64):
    profiles = []
    for _, row in df.iterrows():
        label = IE_MAP.get(row.get("IE"))
        if not label:
            profiles.append(None)
            continue
        pieces = []
        for i in range(1, 92):
            col = f"Q{i}A"
            if col in row and pd.notna(row[col]):
                s = int(row[col])
                pieces.append(f"{QUESTION_MAP[i]} – {SCORE_TEXT.get(s, 'neutral')}.")
        profiles.append(" ".join(pieces) if pieces else None)

    tok = AutoTokenizer.from_pretrained(teacher_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(teacher_dir).to(device).eval()
    t_id2label = mdl.config.id2label
    t_lab2id = {v: int(k) for k, v in t_id2label.items()}

    out = np.zeros((len(profiles), 3), dtype=np.float32)
    texts, idxs = [], []
    for i, txt in enumerate(profiles):
        if txt:
            texts.append(txt); idxs.append(i)

    for start in tqdm(range(0, len(texts), batch_size), desc="Teacher scoring", unit="batch"):
        end = min(start + batch_size, len(texts))
        enc = tok(texts[start:end], truncation=True, padding=True, return_tensors="pt", max_length=512).to(device)
        probs = torch.softmax(mdl(**enc).logits, dim=-1).detach().cpu().numpy()
        for j in range(end - start):
            vec = np.zeros(3, dtype=np.float32)
            for lab, s_idx in LABEL2ID.items():
                t_idx = t_lab2id.get(lab, None)
                vec[s_idx] = probs[j, t_idx] if t_idx is not None else 1.0 / 3
            s = vec.sum(); out[idxs[start + j]] = vec / s if s > 0 else vec
    return out


# -------------------------- Data classes --------------------------
class SentenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_len: int = 128, use_teacher: bool = False):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len
        self.use_teacher = use_teacher

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        # Note: no padding here → we will do dynamic batch padding in collate_fn
        enc = self.tok(
            r["text"],
            truncation=True,
            padding=False,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(r["label_id"]), dtype=torch.long)
        item["sample_weight"] = torch.tensor(float(r["sample_weight"]), dtype=torch.float32)
        if self.use_teacher and "t_probs" in r and r["t_probs"] is not None:
            item["t_probs"] = torch.tensor(r["t_probs"], dtype=torch.float32)
        else:
            item["t_probs"] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        return item


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    unique, counts = np.unique(labels, return_counts=True)
    freq = {u: c for u, c in zip(unique, counts)}
    weights = np.array([1.0 / freq[y] for y in labels], dtype=np.float32)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)


# -------------------------- Collate (FIX) --------------------------
def make_hf_collate(tokenizer: AutoTokenizer):
    """
    Dynamic padding collate function:
    - Pads input_ids / attention_mask (and token_type_ids if present) to max length in batch.
    - Stacks labels, sample_weight, t_probs.
    This prevents the 'resize storage' error from default_collate.
    """
    padder = DataCollatorWithPadding(tokenizer)

    def collate(batch):
        # Separate tokenizer fields from extra fields
        features = []
        labels = []
        sample_w = []
        t_probs = []
        for x in batch:
            feat = {k: x[k] for k in list(x.keys()) if k in ("input_ids", "attention_mask", "token_type_ids")}
            features.append(feat)
            labels.append(x["labels"])
            sample_w.append(x["sample_weight"])
            t_probs.append(x["t_probs"])
        padded = padder(features)
        padded["labels"] = torch.stack(labels) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels, dtype=torch.long)
        padded["sample_weight"] = torch.stack(sample_w) if isinstance(sample_w[0], torch.Tensor) else torch.tensor(sample_w, dtype=torch.float32)
        padded["t_probs"] = torch.stack(t_probs) if isinstance(t_probs[0], torch.Tensor) else torch.tensor(t_probs, dtype=torch.float32)
        return padded

    return collate


# -------------------------- Model --------------------------
class SentenceEncoder(nn.Module):
    def __init__(self, base_model="sentence-transformers/all-MiniLM-L6-v2",
                 num_labels=3, unfreeze_last=0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)
        for p in self.encoder.parameters():
            p.requires_grad = False
        if unfreeze_last > 0 and hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            for layer in self.encoder.encoder.layer[-unfreeze_last:]:
                for p in layer.parameters():
                    p.requires_grad = True

    def mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = self.mean_pool(out.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# -------------------------- Losses --------------------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean", label_smoothing=0.05):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        num_classes = logits.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.label_smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)
        logp = F.log_softmax(logits, dim=-1)
        ce = -(true_dist * logp)
        if self.weight is not None:
            ce = ce * self.weight.unsqueeze(0)
        pt = torch.exp(-ce.sum(dim=1))
        focal = ((1 - pt) ** self.gamma) * ce.sum(dim=1)
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        else:
            return focal


def kd_kl_loss(student_logits, teacher_probs, T: float = 2.0, reduction="batchmean"):
    """
    KL( teacher || student ) at temperature T.
    Returns per-sample loss (B,) when reduction='none'.
    teacher_probs expected shape: (B, C) with rows summing to 1.
    """
    if (teacher_probs is None) or torch.all(teacher_probs <= 0):
        # No teacher provided (or dummy zeros) → return zeros
        B = student_logits.size(0)
        z = torch.zeros(B, device=student_logits.device)
        if reduction == "none":
            return z
        return z.sum()  # scalar 0

    # normalize any non-normalized rows just in case
    teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

    logp_s_T = F.log_softmax(student_logits / T, dim=-1)     # (B, C)
    # elementwise KL, then sum over classes → per-sample (B,)
    per_ex = F.kl_div(logp_s_T, teacher_probs, reduction="none").sum(dim=1) * (T * T)

    if reduction == "none":
        return per_ex                      # (B,)
    elif reduction == "sum":
        return per_ex.sum()                # scalar
    else:  # 'mean' or 'batchmean'
        return per_ex.mean()               # scalar

# -------------------------- Train/Eval --------------------------
def train_epoch(model, loader, optimizer, device, class_weights, alpha=0.5, T=2.0):
    model.train()
    focal = FocalLoss(weight=None, gamma=0.0, label_smoothing=0.05, reduction="none")
    running = 0.0
    n = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        for k in batch:
            batch[k] = batch[k].to(device)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       token_type_ids=batch.get("token_type_ids", None))
        ce_vec = focal(logits, batch["labels"])
        kd_vec = kd_kl_loss(logits, batch["t_probs"], T=T, reduction="none")  # (B,)
        loss_vec = (alpha * kd_vec + (1.0 - alpha) * ce_vec) * batch["sample_weight"]
        loss = loss_vec.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
        optimizer.step()

        running += loss.item() * logits.size(0)
        n += logits.size(0)

    return running / max(1, n)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        for k in batch:
            batch[k] = batch[k].to(device)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       token_type_ids=batch.get("token_type_ids", None))
        preds = logits.argmax(dim=-1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(batch["labels"].cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred,
        labels=[0,1,2], target_names=["Introvert","Extravert","Ambivert"],
        zero_division=0, digits=3
    )
    return acc, macro_f1, report


# -------------------------- Main --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="../dataset/questions.csv")
    ap.add_argument("--out_dir", type=str, default="./sentence_context_model")
    ap.add_argument("--student_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--teacher_dir", type=str, default=None, help="Optional: profile teacher dir for soft labels")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.5, help="KD weight (0=hard only, 1=soft only)")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--unfreeze_last", type=int, default=0, help="Unfreeze last N transformer layers")
    args = ap.parse_args()
    # No teacher → no KD
    if not args.teacher_dir:
        args.alpha = 0.0
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = pick_device()
    print(f"Device: {device}")

    df = pd.read_csv(args.data, sep="\t")
    df = df[df["IE"].isin([1,2,3])].reset_index(drop=True)
    print(f"Loaded {len(df)} users with valid IE labels.")

    print("Computing item weights…")
    item_w = compute_item_weights(df)

    teacher_probs = None
    user_order = list(range(len(df)))
    if args.teacher_dir:
        print("Scoring teacher soft labels (one per user)…")
        teacher_probs = teacher_user_softprobs(df, args.teacher_dir, device)
        print("Teacher soft labels shape:", teacher_probs.shape)

    print("Building sentence dataset…")
    rows_df = build_sentence_rows(df, item_w, teacher_user_probs=teacher_probs, user_order=user_order)
    print(f"Total sentence examples: {len(rows_df):,}")

    print("Group-aware train/val split…")
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    groups = rows_df["user_id"].values
    idx_train, idx_val = next(gss.split(rows_df, groups=groups))
    train_df = rows_df.iloc[idx_train].reset_index(drop=True)
    val_df   = rows_df.iloc[idx_val].reset_index(drop=True)
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    train_ds = SentenceDataset(train_df, tokenizer, max_len=args.max_len, use_teacher=(args.teacher_dir is not None))
    val_ds   = SentenceDataset(val_df,   tokenizer, max_len=args.max_len, use_teacher=(args.teacher_dir is not None))

    # Balanced sampling
    sampler = make_weighted_sampler(train_df["label_id"].values)

    # >>> FIX: dynamic-padding collate_fn and safer loader settings on macOS/MPS
    collate_fn = make_hf_collate(tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, sampler=sampler,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )

    model = SentenceEncoder(base_model=args.student_model, num_labels=3, unfreeze_last=args.unfreeze_last).to(device)

    cls_counts = train_df["label_id"].value_counts().sort_index().to_numpy(dtype=np.float32)
    cls_weights = (cls_counts.sum() / (len(cls_counts) * np.clip(cls_counts, 1.0, None))).astype(np.float32)
    class_weights_t = torch.tensor(cls_weights, dtype=torch.float32, device=device)

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.01)

    best_f1 = -1.0
    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss = train_epoch(model, train_loader, optim, device, class_weights_t, alpha=args.alpha, T=args.temperature)
        acc, macro_f1, report = eval_epoch(model, val_loader, device)
        print(f"Train loss: {tr_loss:.4f} | Val acc: {acc:.3f} | Val macro-F1: {macro_f1:.3f}")
        print(report)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            save_path = os.path.join(args.out_dir, "best_model.pt")
            torch.save({"state_dict": model.state_dict(),
                        "label2id": LABEL2ID,
                        "id2label": ID2LABEL,
                        "student_model": args.student_model,
                        "max_len": args.max_len}, save_path)
            tokenizer.save_pretrained(args.out_dir)
            with open(os.path.join(args.out_dir, "label_map.json"), "w") as f:
                json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f)
            print(f"✓ Saved best model → {save_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()