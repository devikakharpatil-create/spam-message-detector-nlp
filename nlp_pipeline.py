import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# =========================
# 1. Text Cleaning
# =========================
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+', ' URL ', text)   # preserve link signal
    text = re.sub(r'\d+', ' NUM ', text)       # preserve number signal
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# =========================
# 2. Feature Engineering
# =========================
def add_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    df['msg_length'] = df['message'].apply(len)
    df['num_digits'] = df['message'].str.count(r'\d')
    df['num_links'] = df['message'].str.count(r'http|www')
    return df


# =========================
# 3. Train Model
# =========================
def train_model(csv_path: str):
    df = pd.read_csv(csv_path, encoding="latin-1")

    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    df['clean_text'] = df['message'].apply(clean_text)
    df = add_metadata_features(df)

    X_text = df['clean_text']
    y = df['label']

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),
        min_df=3
    )

    X_vec = vectorizer.fit_transform(X_text)

    model = MultinomialNB()
    model.fit(X_vec, y)

    return model, vectorizer


# =========================
# 4. Predict Risk
# =========================
def predict_message(message: str, model, vectorizer):
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])

    proba = model.predict_proba(vector)[0][1]  # spam probability

    if proba >= 0.75:
        risk = "🚨 High Risk"
    elif proba >= 0.40:
        risk = "⚠️ Suspicious"
    else:
        risk = "✅ Safe"

    return {
        "risk": risk,
        "spam_probability": round(proba, 2),
        "cleaned_text": cleaned
    }
