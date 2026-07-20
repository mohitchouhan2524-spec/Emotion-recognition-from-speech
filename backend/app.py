"""
Emotion Recognition inference API.

Receives a short audio recording from the browser, runs it through the
exact same preprocessing pipeline used at training time (trim -> fix
length -> mel-spectrogram + delta features), and returns predicted
emotion probabilities from the trained EmotionCNN checkpoint.

No auth, no database — a single stateless endpoint by design.

Run:
    uvicorn app:app --reload --port 8000

Requires ffmpeg on PATH (used by pydub to decode browser-recorded
webm/opus audio into wav before feature extraction).
"""

import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydub import AudioSegment

# --- Wire up imports to the existing src/ pipeline -------------------------
# backend/app.py -> one level up is the project root (where src/ lives).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.config import BEST_MODEL_PATH, NUM_CLASSES, SAMPLE_RATE, DURATION
from src.model import EmotionCNN
from src.feature_extraction import load_and_trim_audio, fix_length, extract_features_sequence

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# -----------------------------------------------------------------------------

app = FastAPI(title="Speech Emotion Recognition API")

# Frontend and backend are served from the same origin in this setup, so
# CORS restrictions don't apply to normal browser use. Left permissive only
# in case you call /predict from a different origin later (e.g. local dev
# with a separate live-reload server).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
class_labels = []


@app.on_event("startup")
def load_model():
    global model, class_labels
    if not os.path.exists(BEST_MODEL_PATH):
        raise RuntimeError(
            f"No checkpoint found at {BEST_MODEL_PATH}. Train a model first (train.py)."
        )
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)

    net = EmotionCNN(num_classes=NUM_CLASSES).to(device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    model = net
    class_labels = checkpoint["label_classes"]
    print(f"Loaded checkpoint (epoch {checkpoint['epoch']}, val_acc={checkpoint['val_acc']:.4f})")
    print(f"Classes: {class_labels}")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "classes": class_labels}


@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    raw_bytes = await audio.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    tmp_in_path = tmp_wav_path = None
    try:
        # Browser MediaRecorder output (webm/opus) needs decoding via ffmpeg
        # before librosa can read it — write to a temp file for pydub/librosa.
        suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            tmp_in.write(raw_bytes)
            tmp_in_path = tmp_in.name

        sound = AudioSegment.from_file(tmp_in_path)
        sound = sound.set_channels(1).set_frame_rate(SAMPLE_RATE)

        tmp_wav_path = tmp_in_path + ".wav"
        sound.export(tmp_wav_path, format="wav")

        # --- Same preprocessing path as training ---
        y = load_and_trim_audio(tmp_wav_path, sr=SAMPLE_RATE)
        y = fix_length(y, sr=SAMPLE_RATE, duration=DURATION)
        sequence = extract_features_sequence(y, sr=SAMPLE_RATE)  # (channels, n_mels, time)

        tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)  # add batch dim

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        result = {label: float(p) for label, p in zip(class_labels, probs)}
        top_label = max(result, key=result.get)

        return {
            "emotion": top_label,
            "confidence": result[top_label],
            "probabilities": result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    finally:
        for path in (tmp_in_path, tmp_wav_path):
            if path and os.path.exists(path):
                os.remove(path)