from django.shortcuts import render
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch.nn.functional as F
import os
import urllib.parse

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
CONFIDENCE_THRESHOLD = 0.60  # 60%

# --------------------------------------------------
# LOAD MODEL (ONCE)
# --------------------------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.eval()  # VERY IMPORTANT


# --------------------------------------------------
# HOME VIEW
# --------------------------------------------------
def home(request):
    result = None
    confidence = None
    news_link = None

    if request.method == "POST":
        text = request.POST.get("news")

        if text:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)

                confidence_value, prediction = torch.max(probs, dim=1)
                confidence = round(confidence_value.item() * 100, 2)

            # Decision logic (DEFENSE SAFE)
            if confidence < CONFIDENCE_THRESHOLD * 100:
                result = "UNCERTAIN (Needs verification)"
            elif prediction.item() == 0:
                result = "FAKE"
            else:
                result = "REAL"

            # Google News Search link
            query = urllib.parse.quote(text)
            news_link = f"https://news.google.com/search?q={query}"

    return render(
        request,
        "home.html",
        {
            "result": result,
            "confidence": confidence,
            "news_link": news_link
        }
    )