#!/usr/bin/env python3

import os
import sys
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
)

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


# --------------------- utils ---------------------
def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_latest_checkpoint(root_dir):
    """Return the checkpoint-* subfolder with the highest step number, or root_dir if none."""
    try:
        checkpoints = [d for d in os.listdir(root_dir) if d.startswith("checkpoint-")]
    except FileNotFoundError:
        return root_dir
    if not checkpoints:
        return root_dir
    latest = max(checkpoints, key=lambda name: int(name.split("-")[1]))
    return os.path.join(root_dir, latest)


def mbti_to_ie(mbti: str) -> str:
    """Map an MBTI code to Introvert/Extravert (binary ground truth)."""
    if mbti and mbti[0].upper() == "I":
        return "Introvert"
    if mbti and mbti[0].upper() == "E":
        return "Extravert"
    # Fallback; unusual for MBTI to not start with I/E
    return "Introvert"


def softmax_np(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def _normalize_label(s: str) -> str:
    return str(s).strip().lower()


def find_label_index(id2label, swap_ie=False):
    """
    Return indices for Introvert/Extravert/Ambivert robustly, with optional swap.
    Handles spelling variants like Extrovert/Extravert.
    """
    i_idx = e_idx = a_idx = None
    for k, v in id2label.items():
        name = _normalize_label(v)
        if name in {"introvert", "i"}:
            i_idx = int(k)
        elif name in {"extravert", "extrovert", "e"}:
            e_idx = int(k)
        elif name in {"ambivert", "a"}:
            a_idx = int(k)
    if i_idx is None or e_idx is None:
        raise ValueError(
            f"id2label must contain Introvert and Extravert (found: {list(id2label.values())})"
        )
    if swap_ie:
        i_idx, e_idx = e_idx, i_idx
    return i_idx, e_idx, a_idx


# --------------------- loading ---------------------
def load_model_and_tokenizer(model_dir, device):
    """
    Load either:
      - a custom model saved as best_model.pt, or
      - a standard HF checkpoint (must have config.json).
    Returns: (tokenizer, model, id2label: dict[int,str], max_len: int)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    best_path = os.path.join(model_dir, "best_model.pt")

    if os.path.exists(best_path):
        # Custom PyTorch checkpoint path
        ckpt = torch.load(best_path, map_location=device)
        base = ckpt.get("student_model", "sentence-transformers/all-MiniLM-L6-v2")
        max_len = int(ckpt.get("max_len", 128))
        id2label = ckpt.get("id2label", {0: "Introvert", 1: "Extravert", 2: "Ambivert"})
        # Ensure dict[int,str]
        if isinstance(id2label, dict):
            id2label = {int(k): str(v) for k, v in id2label.items()}
        else:
            id2label = {int(i): str(v) for i, v in enumerate(id2label)}

        class SentenceEncoder(nn.Module):
            def __init__(self, base_model, num_labels=3):
                super().__init__()
                self.encoder = AutoModel.from_pretrained(base_model)
                for p in self.encoder.parameters():
                    p.requires_grad = False
                hidden = self.encoder.config.hidden_size
                self.classifier = nn.Linear(hidden, num_labels)

            def mean_pool(self, last_hidden_state, attention_mask):
                mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
                summed = (last_hidden_state * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp_min(1e-9)
                return summed / denom

            def forward(self, input_ids, attention_mask, token_type_ids=None):
                out = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
                pooled = self.mean_pool(out.last_hidden_state, attention_mask)
                return self.classifier(pooled)  # logits

        model = SentenceEncoder(base, num_labels=len(id2label)).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        return tokenizer, model, id2label, max_len

    # Standard HF checkpoint path (expects config.json)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    return tokenizer, model, id2label, 512


@torch.inference_mode()
def predict_batch_logits(texts, tokenizer, model, device, max_len=None):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len if max_len else None,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    logits = out.logits if hasattr(out, "logits") else out  # support both HF and custom model
    return logits.detach().cpu().numpy()


# --------------------- decision rules ---------------------
def redistribute_amb_from_logits(avg_logits, i_idx, e_idx, a_idx, alpha=0.7):
    """
    Reassign a fraction alpha of Ambivert prob to I/E in proportion to their two-class softmax.
    Returns pI_hat, pE_hat, pA, wIE (2-vector of I/E weights).
    """
    z = np.asarray(avg_logits)
    p = softmax_np(z)
    pI = p[i_idx]
    pE = p[e_idx]
    pA = p[a_idx] if a_idx is not None else 0.0

    # Two-class softmax over I/E (closeness among the two)
    z2 = np.array([z[i_idx], z[e_idx]])
    w = np.exp(z2 - z2.max())
    w = w / w.sum()

    pI_hat = pI + alpha * w[0] * pA
    pE_hat = pE + alpha * w[1] * pA
    return pI_hat, pE_hat, pA, w


def ie_score_logit_margin(avg_logits, i_idx, e_idx):
    """
    Pure logit margin: score = logit(E) - logit(I).
    """
    return float(avg_logits[e_idx] - avg_logits[i_idx])


def decide_ie(score, threshold=0.0, extro_bias=0.0):
    """
    Predict Extravert if score + extro_bias > threshold else Introvert.
    """
    return "Extravert" if (score + extro_bias) > threshold else "Introvert"


def calibrate_threshold(scores, y_true, grid_size=200):
    """
    Choose a threshold to maximize balanced accuracy on provided data.
    scores: array-like of shape (n_samples,)
    y_true: array-like with values in {"Introvert","Extravert"}
    """
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    if scores.size == 0:
        return 0.0
    smin, smax = scores.min(), scores.max()
    if smin == smax:
        return 0.0  # degenerate
    candidates = np.linspace(smin - 1e-6, smax + 1e-6, num=grid_size)

    best_thr, best_balacc = 0.0, -1.0
    for thr in candidates:
        y_pred = np.where(scores > thr, "Extravert", "Introvert")
        bal = balanced_accuracy_score(y_true, y_pred)
        if bal > best_balacc:
            best_balacc, best_thr = bal, thr
    return float(best_thr)


# --------------------- main ---------------------
def main():
    parser = argparse.ArgumentParser(
        description="Binary Introvert vs Extravert by collapsing a 3-way (I/E/A) model, with Ambivert redistribution + threshold calibration."
    )
    parser.add_argument("--input", type=str, default="tests_dataset.csv",
                        help="Path to tab-separated test file (MBTI<TAB>msg1|||msg2|||-…).")
    parser.add_argument("--output", type=str, default="test_results.csv",
                        help="Output CSV file.")
    parser.add_argument("--text", nargs="+",
                        help="One or more raw text strings for ad-hoc inference (overrides batch mode)")
    parser.add_argument("--model_dir", type=str, default="sentence_context_model",
                        help="Model directory (custom best_model.pt or HF checkpoint).")
    parser.add_argument("--calibrate", action="store_true",
                        help="Calibrate threshold on a validation split of the input file.")
    parser.add_argument("--calibrate_frac", type=float, default=0.2,
                        help="Fraction of rows for calibration if --calibrate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--amb_alpha", type=float, default=0.7,
                        help="Fraction of Ambivert prob redistributed to I/E (0..1).")
    parser.add_argument("--extro_bias", type=float, default=0.0,
                        help="Additive bias to score favoring Extravert.")
    parser.add_argument("--use_logit_margin", action="store_true",
                        help="Use pure logit margin (E-I) without Amb redistribution.")
    parser.add_argument("--swap_ie", action="store_true",
                        help="Swap Introvert/Extravert label indices (for mis-saved id2label).")
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    # Load model & tokenizer
    model_folder = get_latest_checkpoint(args.model_dir)
    print(f"Model folder: {model_folder}")
    tokenizer, model, id2label, max_len = load_model_and_tokenizer(model_folder, device)
    print(f"id2label: {id2label}")
    i_idx, e_idx, a_idx = find_label_index(id2label, swap_ie=args.swap_ie)

    # ---- Ad-hoc inference
    if args.text:
        logits = predict_batch_logits(args.text, tokenizer, model, device, max_len=max_len)
        probs = softmax_np(logits)
        for text, z, p in zip(args.text, logits, probs):
            if args.use_logit_margin or a_idx is None:
                score = ie_score_logit_margin(z, i_idx, e_idx)
                pI_hat, pE_hat = p[i_idx], p[e_idx]
                extra = {"mode": "logit_margin"}
            else:
                pI_hat, pE_hat, pA, w = redistribute_amb_from_logits(z, i_idx, e_idx, a_idx, alpha=args.amb_alpha)
                score = pE_hat - pI_hat
                extra = {"mode": "redistribute", "pA": pA, "wI": w[0], "wE": w[1]}

            pred = decide_ie(score, threshold=0.0, extro_bias=args.extro_bias)
            print(f"\nText: {text}")
            for k in sorted(id2label):
                print(f"  {id2label[k]:<10}: {p[k]:.3f}")
            print(f"  adjusted pI: {pI_hat:.3f}  adjusted pE: {pE_hat:.3f}  score(E-I): {score:+.4f}  {extra}")
            print(f"-> Pred (binary I/E): {pred}")
        return

    # ---- Batch mode: read rows
    try:
        rows = []
        with open(args.input, newline="", encoding="utf-8") as fin:
            reader = csv.reader(fin, delimiter="\t")
            for row in reader:
                if not row or len(row) < 2:
                    continue
                mbti_code    = row[0].strip()
                messages_raw = row[1]
                messages = [m.strip() for m in messages_raw.split("|||") if m.strip()]
                if not messages:
                    continue
                rows.append((mbti_code, messages))
        if not rows:
            print("No valid rows found in input.", file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    # Score all rows (aggregate over messages)
    all_mbti, all_scores, all_pIhat, all_pEhat = [], [], [], []
    all_probs = []
    print("Encoding & scoring rows...")
    for mbti_code, messages in tqdm(rows, desc="Scoring"):
        logit_mat = predict_batch_logits(messages, tokenizer, model, device, max_len=max_len)  # (N,3)
        avg_logits = logit_mat.mean(axis=0)
        p = softmax_np(avg_logits)

        if args.use_logit_margin or a_idx is None:
            score = ie_score_logit_margin(avg_logits, i_idx, e_idx)
            pI_hat, pE_hat = p[i_idx], p[e_idx]
        else:
            pI_hat, pE_hat, _, _ = redistribute_amb_from_logits(avg_logits, i_idx, e_idx, a_idx, alpha=args.amb_alpha)
            score = pE_hat - pI_hat

        all_mbti.append(mbti_code)
        all_scores.append(score)
        all_pIhat.append(pI_hat)
        all_pEhat.append(pE_hat)
        all_probs.append(p)

    all_scores = np.asarray(all_scores)
    all_probs = np.vstack(all_probs)
    y_true = np.array([mbti_to_ie(m) for m in all_mbti])

    # Show class balance of ground truth
    gt_unique, gt_counts = np.unique(y_true, return_counts=True)
    print(f"Ground-truth distribution: {dict(zip(gt_unique, gt_counts))}")

    # Calibrate threshold if requested
    if args.calibrate:
        rng = np.random.RandomState(args.seed)
        n = len(all_scores)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_cal = max(1, int(n * float(args.calibrate_frac)))
        cal_idx = idx[:n_cal]
        threshold = calibrate_threshold(all_scores[cal_idx], y_true[cal_idx])
        print(f"Calibrated threshold on {n_cal} samples: {threshold:+.6f}")
    else:
        threshold = 0.0
        print(f"No calibration; using default threshold: {threshold:+.6f}")

    # Final preds with optional extro bias
    y_pred = np.where(all_scores + args.extro_bias > threshold, "Extravert", "Introvert")

    # Write CSV
    labels_all = [id2label[k] for k in sorted(id2label)]
    idx_order  = sorted(id2label)
    with open(args.output, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["mbti", "orig_ie", "pred_ie", "score_adj", "threshold", "extro_bias",
                         "pI_hat", "pE_hat"] + [f"p_{lab}" for lab in labels_all])
        for mbti_code, pred, score, pIh, pEh, probs in zip(all_mbti, y_pred, all_scores, all_pIhat, all_pEhat, all_probs):
            writer.writerow(
                [mbti_code, mbti_to_ie(mbti_code), pred, f"{score:.6f}", f"{threshold:.6f}", f"{args.extro_bias:.6f}",
                 f"{pIh:.6f}", f"{pEh:.6f}"] + [f"{probs[i]:.6f}" for i in idx_order]
            )

    print(f"Done → results written to {args.output}")

    # --- Metrics & plots (Binary I/E only) ---
    labels_binary = ["Introvert", "Extravert"]

    cm = confusion_matrix(y_true, y_pred, labels=labels_binary)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Introvert vs Extravert)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(len(labels_binary)), labels_binary)
    plt.yticks(np.arange(len(labels_binary)), labels_binary)
    for i in range(len(labels_binary)):
        for j in range(len(labels_binary)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("confusion_matrix_personality_classifier.png")
    plt.close()

    # Binary accuracy summary
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    print("\n=== Binary Classification Report (Introvert vs Extravert) ===")
    print(classification_report(y_true, y_pred, labels=labels_binary, zero_division=0))
    print(f"Balanced Accuracy: {bal_acc:.4f}")

    # Score histogram to inspect collapse
    plt.figure()
    plt.hist(all_scores, bins=40)
    plt.title("Score Distribution (E−I; after redistribution if enabled)")
    plt.xlabel("score = adjusted_pE - adjusted_pI  (positive → Extravert)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("score_hist.png")
    plt.close()
    print("Plots saved: confusion_matrix_personality_classifier.png, score_hist.png")

    # Sanity: show prediction distribution
    unique_preds, counts = np.unique(y_pred, return_counts=True)
    pred_dist = dict(zip(unique_preds, counts))
    print(f"Prediction distribution: {pred_dist}")
    if len(unique_preds) == 1:
        print("WARNING: All predictions collapsed to a single class.")
        print("Try: --calibrate, increase --amb_alpha (e.g., 0.9), add --extro_bias (e.g., 0.1), or --swap_ie.")
        print("Also inspect score_hist.png to see if scores are all negative or near zero.")


if __name__ == "__main__":
    main()