import os
import sys
import pandas as pd
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, accuracy_score

# Utility functions
def get_latest_checkpoint(root_dir):
    checkpoints = [d for d in os.listdir(root_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return root_dir
    latest = max(checkpoints, key=lambda name: int(name.split("-")[1]))
    return os.path.join(root_dir, latest)

def mbti_to_ie(mbti: str) -> str:
    if mbti and mbti[0].upper() == "I":
        return "Introvert"
    elif mbti and mbti[0].upper() == "E":
        return "Extravert"
    else:
        return "Ambivert"

def predict_batch(texts, tokenizer, model):
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs

# Load model once
@st.cache(allow_output_mutation=True)
def load_model(model_dir: str):
    folder = get_latest_checkpoint(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(folder)
    model.eval()
    id2label = model.config.id2label
    return tokenizer, model, id2label

# Streamlit UI
st.title("Personality Classifier Dashboard")
model_dir = st.sidebar.text_input("Model Directory", "personality_model")
tokenizer, model, id2label = load_model(model_dir)
labels = [id2label[i] for i in sorted(id2label.keys())]

# Single-sentence inference
st.header("Single Sentence Classification")
user_text = st.text_area("Enter a sentence to classify:")
if st.button("Classify Sentence") and user_text:
    probs = predict_batch([user_text], tokenizer, model)[0]
    df_probs = pd.DataFrame({"Label": labels, "Probability": probs})
    st.write(df_probs)
    fig, ax = plt.subplots()
    ax.bar(df_probs['Label'], df_probs['Probability'])
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Batch evaluation
st.header("Batch Evaluation (Upload File)")
uploaded = st.file_uploader("Upload TSV: MBTI<TAB>msg1|||msg2|||...", type=["csv", "tsv", "txt"])
if uploaded:
    try:
        df_input = pd.read_csv(uploaded, sep="\t", header=None, names=["mbti", "msgs"])
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()
    y_true, y_pred = [], []
    for _, row in df_input.iterrows():
        mbti_code = row['mbti']
        msgs = [m.strip() for m in row['msgs'].split('|||') if m.strip()]
        if not msgs:
            continue
        probs = predict_batch(msgs, tokenizer, model)
        avg = np.mean(probs, axis=0)
        idx = int(np.argmax(avg))
        y_pred.append(id2label[idx])
        y_true.append(mbti_to_ie(mbti_code))
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    st.write(f"**Accuracy:** {acc:.3f}")
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig2, ax2 = plt.subplots()
    im = ax2.imshow(cm, interpolation='nearest')
    ax2.set_xticks(range(len(labels)))
    ax2.set_yticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax2.text(j, i, cm[i, j], ha='center', va='center')
    plt.colorbar(im, ax=ax2)
    st.pyplot(fig2)
    # Correct vs Incorrect
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    incorrect = len(y_true) - correct
    fig3, ax3 = plt.subplots()
    ax3.bar(["Correct", "Incorrect"], [correct, incorrect])
    ax3.set_ylabel("Count")
    st.pyplot(fig3)
