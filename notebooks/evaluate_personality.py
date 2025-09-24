#!/usr/bin/env python3

import os
import sys
import csv
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

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
    """Map an MBTI code to Introvert/Extravert/Ambivert."""
    if mbti and mbti[0].upper() == "I":
        return "Introvert"
    elif mbti and mbti[0].upper() == "E":
        return "Extravert"
    else:
        return "Ambivert"


def softmax_np(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def ambivert_flip(avg_probs, id2label, margin=0.12, amb_min=0.30):
    """
    If |p(I) - p(E)| < margin and p(Amb) >= amb_min, choose Ambivert; else argmax.
    """
    label2idx = {v: k for k, v in id2label.items()}
    i_idx = label2idx.get("Introvert", 0)
    e_idx = label2idx.get("Extravert", 1)
    a_idx = label2idx.get("Ambivert", 2)
    if abs(avg_probs[i_idx] - avg_probs[e_idx]) < margin and avg_probs[a_idx] >= amb_min:
        return a_idx
    return int(np.argmax(avg_probs))


# --------------------- loading ---------------------
def load_model_and_tokenizer(model_dir, device):
    """
    Load either:
      - a custom model saved as best_model.pt (with tokenizer files in the same folder), or
      - a standard HF checkpoint (must have config.json).
    Returns: (tokenizer, model, id2label: dict, max_len: int)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    best_path = os.path.join(model_dir, "best_model.pt")

    if os.path.exists(best_path):
        # Custom PyTorch checkpoint path
        ckpt = torch.load(best_path, map_location=device)
        base = ckpt.get("student_model", "sentence-transformers/all-MiniLM-L6-v2")
        max_len = int(ckpt.get("max_len", 128))
        id2label = ckpt.get("id2label", {0: "Introvert", 1: "Extravert", 2: "Ambivert"})
        id2label = {int(k): v for k, v in (id2label.items() if isinstance(id2label, dict) else enumerate(id2label))}

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
    id2label = model.config.id2label
    return tokenizer, model, id2label, 512


@torch.inference_mode()
def predict_batch_logits(texts, tokenizer, model, device, max_len=None):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len if max_len else None,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    logits = out.logits if hasattr(out, "logits") else out  # support both HF and custom model
    return logits.detach().cpu().numpy()


@torch.inference_mode()
def predict_batch_probs(texts, tokenizer, model, device, max_len=None):
    logits = predict_batch_logits(texts, tokenizer, model, device, max_len=max_len)
    return softmax_np(logits)


# --------------------- main ---------------------
def main():
    parser = argparse.ArgumentParser(
        description="Classify text(s) into Introvert/Extravert/Ambivert."
    )
    parser.add_argument(
        "--input", type=str, default="tests_dataset.csv",
        help="Path to tab-separated test file (MBTI<TAB>msg1|||msg2|||-…). Default: tests_dataset.csv"
    )
    parser.add_argument(
        "--output", type=str, default="test_results.csv",
        help="Output CSV file. Default: test_results.csv"
    )
    parser.add_argument(
        "--text", nargs="+",
        help="One or more raw text strings for ad-hoc inference (overrides batch mode)"
    )
    parser.add_argument(
        "--model_dir", type=str, default="sentence_context_model",
        help="Model directory (custom best_model.pt or HF checkpoint). Default: sentence_context_model"
    )
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    # Load model & tokenizer (supports custom or HF)
    model_folder = get_latest_checkpoint(args.model_dir)
    print(model_folder)
    tokenizer, model, id2label, max_len = load_model_and_tokenizer(model_folder, device)
    idx_order = sorted(id2label.keys())
    labels = [id2label[i] for i in idx_order]

    # ---- Ad-hoc inference
    if args.text:
        probs = predict_batch_probs(args.text, tokenizer, model, device, max_len=max_len)
        for text, p in zip(args.text, probs):
            print(f"Text: {text}")
            for idx in idx_order:
                print(f"  {id2label[idx]:<10}: {p[idx]:.3f}")
            # Ambivert tie-break on single text
            pred_idx = ambivert_flip(p, id2label, margin=0.12, amb_min=0.30)
            print(f"-> Pred: {id2label[pred_idx]}\n")
        return

    # ---- Batch evaluation mode
    try:
        with open(args.input, "r", encoding="utf-8") as f_count:
            total = sum(1 for _ in f_count)
    except FileNotFoundError:
        print(f"Error: input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.input, newline="", encoding="utf-8") as fin, \
            open(args.output, "w", newline="", encoding="utf-8") as fout:

        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout)
        writer.writerow(["mbti", "orig_ie", "pred_ie"] + [f"p_{lab}" for lab in labels])

        for row in tqdm(reader, total=total, desc="Processing rows"):
            if not row or len(row) < 2:
                continue
            mbti_code    = row[0].strip()
            messages_raw = row[1]
            messages = [m.strip() for m in messages_raw.split("|||") if m.strip()]
            if not messages:
                continue

            # Aggregate **logits** across messages → more stable decision
            logit_mat = predict_batch_logits(messages, tokenizer, model, device, max_len=max_len)  # (N,3)
            avg_probs = softmax_np(logit_mat.mean(axis=0))  # (3,)

            # Ambivert tie-break
            pred_idx = ambivert_flip(avg_probs, id2label, margin=0.12, amb_min=0.30)
            pred_ie  = id2label[pred_idx]
            orig_ie  = mbti_to_ie(mbti_code)

            writer.writerow(
                [mbti_code, orig_ie, pred_ie]
                + [f"{avg_probs[i]:.6f}" for i in idx_order]
            )

    print(f"Done → results written to {args.output}")

    # --- Metrics & plots ---
    df = pd.read_csv(args.output)
    y_true = df["orig_ie"].values
    y_pred = df["pred_ie"].values

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(len(labels)), labels)
    plt.yticks(np.arange(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Correct vs Incorrect bar chart
    correct = (y_true == y_pred).sum()
    incorrect = (y_true != y_pred).sum()
    plt.figure()
    plt.bar(["Correct", "Incorrect"], [correct, incorrect])
    plt.title("Prediction Accuracy")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("prediction_accuracy.png")
    plt.close()

    # Report to console (handy)
    print("\n=== 3-WAY Classification Report ===")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print("Plots saved: confusion_matrix.png, prediction_accuracy.png")


if __name__ == "__main__":
    main()