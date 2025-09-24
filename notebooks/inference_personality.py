# inference_personality.py

import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# 1) Points in to the directory where your best model checkpoint lives
#    If you used `load_best_model_at_end=True`, that will be at ./personality_model
# Helper to get latest checkpoint, or fallback to base dir
def get_latest_checkpoint(root_dir):
    # List all checkpoint directories
    checkpoints = [d for d in os.listdir(root_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return root_dir
    # Pick the checkpoint with the highest step number
    latest = max(checkpoints, key=lambda name: int(name.split("-")[1]))
    return os.path.join(root_dir, latest)

# Automatically select the most recent checkpoint directory
MODEL_DIR = get_latest_checkpoint("./personality_model")


# 2) Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# 3) Your label names in the same order as the model’s outputs
#    You can pull these from model.config.id2label as well.
id2label = model.config.id2label
# e.g. {0: "Introvert", 1: "Extravert", 2: "Ambivert"}

def predict_personality(texts, batch_size: int = 8):
    """
    Given a list of strings, returns a list of dicts:
      [{"text": "...", "scores": {"Introvert":0.02, ...}}, ...]
    """
    results = []
    # tokenize in batch
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits         # shape (batch, 3)
        probs  = torch.softmax(logits, -1)   # shape (batch, 3)
    probs = probs.cpu().numpy()

    for text, prob_vec in zip(texts, probs):
        # build a map label → probability
        score_map = { id2label[i]: float(prob_vec[i]) for i in range(len(prob_vec)) }
        results.append({"text": text, "scores": score_map})
    return results

if __name__ == "__main__":
    # Example usage
    samples = [
        "I love going to crowded parties and talking with many people.",
        "I usually prefer quiet nights in with a book.",
        "Sometimes I like both group events and alone time."
    ]
    preds = predict_personality(samples)
    for p in preds:
        print(f"\nText: {p['text']}")
        for label, score in p["scores"].items():
            print(f"  {label:10s}: {score:.3f}")