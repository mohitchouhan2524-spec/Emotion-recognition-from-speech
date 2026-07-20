FROM python:3.11-slim

# ffmpeg: needed by pydub to decode browser-recorded webm/opus audio
# libsndfile1: needed by librosa/soundfile to read wav files
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first (avoids pulling unnecessary CUDA packages,
# which would blow up image size and slow every deploy)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what's needed to run the app — see .dockerignore for what's
# excluded (raw audio dataset, venv, notebooks, feature caches, etc.)
COPY src/ ./src/
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY checkpoints/ ./checkpoints/

# Hugging Face Spaces expects the app on port 7860. Render/other platforms
# typically inject their own $PORT — the shell form below respects either.
ENV PORT=7860
EXPOSE 7860

CMD uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}