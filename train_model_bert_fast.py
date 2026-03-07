import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# ---------- CONFIG ----------
MAX_SAMPLES = 200        # keeps training under 10 seconds
EPOCHS = 1
BATCH_SIZE = 8
MODEL_NAME = "distilbert-base-uncased"
# ----------------------------

print("Loading dataset...")

fake = pd.read_csv("dataset/Fake.csv").head(MAX_SAMPLES)
real = pd.read_csv("dataset/True.csv").head(MAX_SAMPLES)

fake["label"] = 0
real["label"] = 1

df = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)

texts = df["text"].tolist()
labels = torch.tensor(df["label"].tolist())

print("Loading tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

dataset = TensorDataset(
    encodings["input_ids"],
    encodings["attention_mask"],
    labels
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Loading model...")
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# 🔒 FREEZE BERT (THIS MAKES IT FAST)
for param in model.distilbert.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=5e-4)

print("Training (fast mode)...")

model.train()
for epoch in range(EPOCHS):
    for input_ids, attention_mask, labels in loader:
        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

print("Saving model...")
model.save_pretrained("model_bert")
tokenizer.save_pretrained("model_bert")

print("✅ FAST BERT TRAINING COMPLETE")