---
title: Voice Emotion Reader
emoji: 🎙️
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Voice Emotion Reader

Records 5 seconds of speech in the browser and classifies the speaker's
emotion using a CNN trained on RAVDESS. No auth, no database — a single
stateless inference endpoint.

## Deploying to Hugging Face Spaces

1. Create a new Space at huggingface.co/new-space, choose **Docker** as
   the SDK.
2. Push this repo to the Space's git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
3. If your checkpoint file (`checkpoints/best_model.pt` or wherever
   `BEST_MODEL_PATH` points in `config.py`) is over 10MB, it needs Git LFS:
   ```bash
   git lfs install
   git lfs track "*.pt"
   git add .gitattributes
   ```
4. Spaces will build the Dockerfile automatically. First build takes a
   few minutes (installing torch + ffmpeg); watch the build logs in the
   Space's "Logs" tab.
5. Once built, your app is live at
   `https://<your-username>-<space-name>.hf.space`

## Local run (unchanged)

```bash
docker build -t voice-emotion .
docker run -p 7860:7860 voice-emotion
```

Then open `http://localhost:7860` — frontend and backend are served from
the same origin now, so no separate `http.server` step is needed anymore.

## Project layout expected

```
project-root/
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── README.md              (this file — Spaces config lives in the YAML above)
├── src/                   (config.py, model.py, feature_extraction.py, ...)
├── checkpoints/           (best_model.pt — wherever BEST_MODEL_PATH points)
├── backend/
│   └── app.py
└── frontend/
    └── index.html
```