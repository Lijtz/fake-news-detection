import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from nltk.corpus import stopwords

# -----------------------------
# 1. Load Dataset
# -----------------------------
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

true_df["label"] = 1
fake_df["label"] = 0

df = pd.concat([true_df, fake_df], ignore_index=True)
df["text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()

# -----------------------------
# 2. Clean Text
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# -----------------------------
# 4. TF-IDF Vectorization
# -----------------------------
stops = stopwords.words("english")

tfidf = TfidfVectorizer(
    stop_words=stops,
    max_df=0.9,
    min_df=5,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -----------------------------
# 5. Logistic Regression Model
# -----------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred = model.predict(X_test_tfidf)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
disp = ConfusionMatrixDisplay.from_estimator(
    model,
    X_test_tfidf,
    y_test
)
plt.title("Fake vs Real News")
plt.show()

# -----------------------------
# 7. Save Model
# -----------------------------
with open("models/model.pkl", "wb") as f:
    pickle.dump({"tfidf": tfidf, "model": model}, f)

print("\nModel saved successfully.")