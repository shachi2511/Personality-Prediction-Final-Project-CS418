# Install dependencies if not already installed:
# pip install transformers datasets scikit-learn pandas

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score

# ------------------------
# STEP 1: Load the dataset
# ------------------------
df = pd.read_csv("../dataset/questions.csv", sep="\t")

# ------------------------
# STEP 2: Prepare question & response mappings
# ------------------------
# Question texts Q1–Q91
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

# Map raw scores to agreement text
score_map = {
    1: "disagree",
    2: "slightly disagree",
    3: "neutral",
    4: "slightly agree",
    5: "agree"
}

# Label mapping from IE
label_map = {1: "Introvert", 2: "Extravert", 3: "Ambivert"}

# ------------------------
# STEP 3: Build sentence-level dataset
# ------------------------
examples = []
for _, row in df.iterrows():
    label = label_map.get(row["IE"], None)
    if not label:
        continue
    for i in range(1, 92):
        col = f"Q{i}A"
        if col in row and pd.notna(row[col]):
            text = f"{question_map[i]} – {score_map.get(int(row[col]), 'neutral')}"
            examples.append({"text": text, "label": label})

# Convert to DataFrame and encode labels
examples_df = pd.DataFrame(examples)
le = LabelEncoder()
examples_df["labels"] = le.fit_transform(examples_df["label"])

# ------------------------
# STEP 4: Convert to HuggingFace Dataset
# ------------------------
dataset = Dataset.from_pandas(examples_df[["text", "labels"]])
dataset = dataset.train_test_split(test_size=0.1, seed=42)
dataset = {
    "train": dataset["train"],
    "validation": dataset["test"]
}
# ------------------------
# STEP 5: Tokenization + Preprocessing
# ------------------------
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(batch):
    enc = tokenizer(batch["text"], padding=True, truncation=True)
    enc["labels"] = batch["labels"]
    return enc

dataset = dataset.map(preprocess, batched=True)

# ------------------------
# STEP 6: Model & Training Arguments
# ------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label={i: label for i, label in enumerate(le.classes_)},
    label2id={label: i for i, label in enumerate(le.classes_)},
)

training_args = TrainingArguments(
     output_dir="./personality_model",
 do_train=True,
 do_eval=True,
 eval_steps=len(dataset["train"]),   # one evaluation per epoch
 logging_steps=100,
 learning_rate=2e-5,
 per_device_train_batch_size=16,
 per_device_eval_batch_size=16,
 num_train_epochs=3,
 weight_decay=0.01,
 logging_dir="./logs",
)
data_collator = DataCollatorWithPadding(tokenizer)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# ------------------------
# STEP 7: Trainer & Train
# ------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()