import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import re

def clean_text(text):
    text = text.lower()                          # convert to lowercase
    text = re.sub(r'\d+', '', text)              # remove numbers
    text = re.sub(r'[^\w\s]', '', text)          # remove punctuation like ! ? . ,
    text = text.strip()                          # remove extra spaces
    return text

# ── Load data ──────────────────────────────────────────────
df = pd.read_csv('data/spam.csv', encoding='latin-1')

# ── Clean up columns ───────────────────────────────────────
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df = df.dropna(subset=['message'])
df['message'] = df['message'].apply(clean_text)

print("Null messages before fix:", df['message'].isnull().sum())  # see how many bad rows
df = df.dropna(subset=['message'])                                 # ← THE FIX goes here
print("Null messages after fix:", df['message'].isnull().sum())   # should print 0

# ── Convert labels to numbers ──────────────────────────────
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ── Split into input and output ────────────────────────────
X = df['message']
y = df['label']

# ── Train/test split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── TF-IDF Vectorization ───────────────────────────────────
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=8000,
    ngram_range=(1, 2)      # captures both single words AND two-word phrases
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── Train the model ────────────────────────────────────────
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ── Evaluate ───────────────────────────────────────────────
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Save model & vectorizer ────────────────────────────────
os.makedirs('model', exist_ok=True)   # creates model/ folder if it doesn't exist
with open('model/spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved to model/ folder!")