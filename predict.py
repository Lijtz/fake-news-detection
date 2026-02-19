import pickle
import re

# -----------------------------
# 1. Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# 2. Load Saved Model
# -----------------------------
with open("models/model.pkl", "rb") as f:
    bundle = pickle.load(f)

tfidf = bundle["tfidf"]
model = bundle["model"]

# -----------------------------
# 3. Prediction Function
# -----------------------------
def predict_news(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    probability = model.predict_proba(vectorized)[0][1]

    if probability >= 0.5:
        return f"REAL news (Confidence: {probability:.2f})"
    else:
        return f"FAKE news (Confidence: {1 - probability:.2f})"

# -----------------------------
# 4. CLI Interface
# -----------------------------
print("\nFake News Detection System")
print("----------------------------------")

while True:
    user_input = input("\nEnter news headline/article (or type 'exit'): ")

    if user_input.lower() == "exit":
        break

    result = predict_news(user_input)
    print("Prediction:", result)

print("\nSession Ended.")
