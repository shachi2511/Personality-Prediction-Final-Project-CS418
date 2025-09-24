#!/usr/bin/env python3

import os
import sys
import csv
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tqdm import tqdm

def get_latest_checkpoint(root_dir):
    ckpts = [d for d in os.listdir(root_dir) if d.startswith("checkpoint-")]
    return os.path.join(root_dir, max(ckpts, key=lambda n: int(n.split("-")[1]))) if ckpts else root_dir

def mbti_to_ie(mbti: str) -> str:
    return "Introvert" if mbti.upper().startswith("I") else "Extravert"

def predict_batch(texts, tokenizer, model):
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=-1).cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Profile-level evaluation of personality model")
    parser.add_argument("--input",     type=str,   default="tests_dataset.csv")
    parser.add_argument("--output",    type=str,   default="test_results.csv")
    parser.add_argument("--model_dir", type=str,   default="oversampled_profile_model")
    args = parser.parse_args()

    # Load the fine-tuned profile-level model
    ckpt      = get_latest_checkpoint(args.model_dir)
    print(ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model     = AutoModelForSequenceClassification.from_pretrained(ckpt)
    print(model)
    model.eval()
    id2lab = model.config.id2label  # e.g. {0:"Introvert",1:"Extravert",2:"Ambivert"}

    # Read all rows
    with open(args.input, encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    total = len(rows)

    # Prepare output
    out = []
    for row in tqdm(rows, total=total, desc="Evaluating profiles"):
        if len(row) < 2:
            continue
        mbti_code   = row[0].strip()
        msgs_raw    = row[1]
        messages    = [m.strip() for m in msgs_raw.split("|||") if m.strip()]
        if not messages:
            continue

        # Build one profile text per user
        profile_text = " | ".join(messages)

        # Single inference on full profile
        probs = predict_batch([profile_text], tokenizer, model)[0]
        pred_idx = int(np.argmax(probs))
        pred_ie  = id2lab[pred_idx]
        orig_ie  = mbti_to_ie(mbti_code)

        out.append({
            "mbti": mbti_code,
            "orig_ie": orig_ie,
            "pred_ie": pred_ie,
            "p_Introvert": probs[id2lab.inverse["Introvert"]] if hasattr(id2lab, "inverse") else probs[list(id2lab.keys())[list(id2lab.values()).index("Introvert")]],
            "p_Extravert": probs[id2lab.inverse["Extravert"]] if hasattr(id2lab, "inverse") else probs[list(id2lab.keys())[list(id2lab.values()).index("Extravert")]],
            "p_Ambivert": probs[id2lab.inverse["Ambivert"]] if hasattr(id2lab, "inverse") else probs[list(id2lab.keys())[list(id2lab.values()).index("Ambivert")]],
        })

    df = pd.DataFrame(out)
    df.to_csv(args.output, index=False)
    print(f"Results written to {args.output}")

    # 3-way report
    labels3 = ["Introvert","Extravert","Ambivert"]
    print("\n=== 3-WAY Classification Report ===")
    print(classification_report(df["orig_ie"], df["pred_ie"], labels=labels3, zero_division=0))

    # 3-way confusion matrix
    cm3 = confusion_matrix(df["orig_ie"], df["pred_ie"], labels=labels3)
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm3, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels3, rotation=45)
    ax.set_yticklabels(labels3)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm3[i,j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig("confusion_3way.png")
    plt.close(fig)

    # Binary collapse
    def collapse(pred, row):
        if pred != "Ambivert":
            return pred
        return "Extravert" if row["p_Extravert"] > row["p_Introvert"] else "Introvert"

    df["pred_bin"] = df.apply(lambda r: collapse(r["pred_ie"], r), axis=1)
    acc_bin = accuracy_score(df["orig_ie"], df["pred_bin"])
    print(f"\nBinary (I vs E) accuracy: {acc_bin:.3f}")

    # Binary confusion
    cm2 = confusion_matrix(df["orig_ie"], df["pred_bin"], labels=["Introvert","Extravert"])
    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(cm2, cmap="Oranges")
    ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
    ax2.set_xticklabels(["Introvert","Extravert"])
    ax2.set_yticklabels(["Introvert","Extravert"])
    ax2.set_xlabel("Pred"); ax2.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, cm2[i,j], ha="center", va="center")
    fig2.colorbar(im2, ax=ax2)
    fig2.tight_layout()
    fig2.savefig("confusion_binary.png")
    plt.close(fig2)

    print("Plots saved: confusion_3way.png, confusion_binary.png")

if __name__ == "__main__":
    main()