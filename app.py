from keep_alive import start
start()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle

# ── Load the saved model & vectorizer ─────────────────────
# We load them ONCE when the server starts, not on every request
# This makes the API fast
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'model/spam_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, 'model/vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)

# ── Create the FastAPI app ─────────────────────────────────
app = FastAPI()

# ── CORS middleware ────────────────────────────────────────
# This allows your HTML frontend (running in browser) to talk
# to this API. Without this, the browser will block the request.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend static files ────────────────────────────
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ── Define what the incoming request looks like ────────────
# Pydantic automatically validates that the request has a "message" field
class MessageInput(BaseModel):
    message: str

# ── The prediction endpoint ────────────────────────────────
@app.post("/predict")
def predict(data: MessageInput):
    # Step 1: transform the text using the SAME vectorizer used in training
    transformed = vectorizer.transform([data.message])
    
    # Step 2: predict — returns [0] for ham or [1] for spam
    probability = model.predict_proba(transformed)[0]
    spam_probability = float(probability[1])
    prediction = 1 if spam_probability >= 0.3 else 0
    confidence = round(max(float(probability[0]), float(probability[1])) * 100, 2)

    
    return {
        "message": data.message,
        "prediction": "spam" if prediction == 1 else "ham",
        "confidence": f"{confidence}%",
        "spam_probability": round(spam_probability * 100, 2)
    }

# ── Serve frontend at root ─────────────────────────────────
@app.get("/")
def root():
    return FileResponse(os.path.join(FRONTEND_DIR, 'index.html'))

# ── Health check endpoint ──────────────────────────────────
@app.get("/health")
def health():
    return {"status": "Spam Detector API is running!"}