from django.shortcuts import render
import random
import urllib.parse

def home(request):
    result = None
    confidence = None
    news_link = None
    text = ""

    if request.method == "POST":
        text = request.POST.get("news")

        # Simple AI-like logic
        fake_keywords = ["ban", "kill", "fake", "scam", "fraud", "attack"]

        if any(word in text.lower() for word in fake_keywords):
            result = "FAKE"
            confidence = round(random.uniform(75, 95), 2)
        else:
            result = "REAL"
            confidence = round(random.uniform(65, 90), 2)

        # Google News link
        query = urllib.parse.quote(text)
        news_link = f"https://news.google.com/search?q={query}"

    return render(request, "home.html", {
        "result": result,
        "confidence": confidence,
        "news_link": news_link,
        "text": text
    })