#!/usr/bin/env python3
# train_profile_oversample.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score

# 1) Load and build one-profile-per-person examples
df = pd.read_csv("../dataset/questions.csv", sep="\t")
question_map = {i+1: q for i, q in enumerate([
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
    "I talk to a lot of different people at parties."
])}
score_map = {1:"disagree",2:"slightly disagree",3:"neutral",4:"slightly agree",5:"agree"}
ie_map    = {1:"Introvert",2:"Extravert",3:"Ambivert"}

rows = []
for _, row in df.iterrows():
    ie = ie_map.get(row.get("IE"))
    if ie is None:
        continue
    parts = []
    for i, q in question_map.items():
        s = row.get(f"Q{i}A")
        if pd.notna(s):
            parts.append(f"{q} – {score_map[int(s)]}")
    if parts:
        rows.append({"text": " | ".join(parts), "label": ie})

data = pd.DataFrame(rows)
le = LabelEncoder()
data["labels"] = le.fit_transform(data["label"])

# 2) Split off validation 10% (stratified)
train_df, val_df = train_test_split(
    data, test_size=0.1, stratify=data["labels"], random_state=42
)

# 3) Oversample minority classes in train_df
max_count = train_df["label"].value_counts().max()
balanced_parts = []
for cls, grp in train_df.groupby("label"):
    balanced_parts.append(grp.sample(max_count, replace=True, random_state=42))
train_df_bal = pd.concat(balanced_parts).sample(frac=1, random_state=42)
print("Balanced train class counts:\n", train_df_bal["label"].value_counts())

# 4) Convert to HF datasets
train_ds = Dataset.from_pandas(train_df_bal[["text","labels"]])
val_ds   = Dataset.from_pandas(val_df[["text","labels"]])

# 5) Tokenize / preprocess
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
def preprocess(batch):
    enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
    enc["labels"] = batch["labels"]
    return enc

train_ds = train_ds.map(preprocess, batched=True, remove_columns=["text","labels"])
val_ds   = val_ds.map(preprocess, batched=True, remove_columns=["text","labels"])

# 6) Load model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(le.classes_),
    id2label={i:lab for i,lab in enumerate(le.classes_)},
    label2id={lab:i for i,lab in enumerate(le.classes_)},
)

# 7) Trainer setup
training_args = TrainingArguments(
    output_dir="./oversampled_profile_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
)

data_collator = DataCollatorWithPadding(tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 8) Train
trainer.train()

# 9) Final eval on val set
metrics = trainer.evaluate()
print(f"\nValidation accuracy: {metrics['eval_accuracy']:.3f}")