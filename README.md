# Speech Emotion Recognition

A CNN trained on the RAVDESS dataset that listens to 5 seconds of speech and predicts the speaker's emotion out of eight classes (angry, calm, disgust, fearful, happy, neutral, sad, surprised). Runs as a single FastAPI app that serves both the prediction API and a small browser frontend for recording audio.

## Project layout

```
project-root/
├── src/                 data parsing, feature extraction, model, training, evaluation
├── backend/app.py       inference API, loads the checkpoint and serves the frontend
├── frontend/index.html  records 5s of audio in the browser and shows the result
├── tests/                unit tests for filename parsing
├── models/          trained model weights
├── Dockerfile
└── requirements.txt
```

## Running it locally

Needs Python 3.11 and ffmpeg (used to decode the audio recorded in the browser before it's fed to the model).

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Start the server:

```
cd backend
uvicorn app:app --reload --port 7860
```

Open `http://localhost:7860`, allow microphone access, and record for 5 seconds.

You can also run it with Docker, which matches what actually runs in production:

```
docker build -t voice-emotion .
docker run -p 7860:7860 voice-emotion
```

## Training your own model

The repo ships with a trained checkpoint, but if you want to retrain from scratch:

```
python src/data_parser.py
python src/feature_extraction.py
python src/train.py
python src/evaluate.py
```

`data_parser.py` expects the RAVDESS dataset extracted into an `Audiodata/` folder at the project root, with subfolders `Actor_01`, `Actor_02`, and so on.

## Deployment
To deploy your own copy:

1. Push the repo to GitHub.
2. Create a Render account — no card required for the free tier.
3. New → Web Service, connect the repo. Render detects the Dockerfile automatically.
4. Pick the free instance type and deploy.

The free tier sleeps after 15 minutes with no traffic. The first request after that takes 30-60 seconds while the instance wakes up and reloads the model — that's expected, not a bug.

## Notes

`Audiodata/` (the raw dataset) and `features.npy` (cached feature arrays) are not committed to git. Both are regenerated locally by `feature_extraction.py`, and together they'd add several GB to the repo. The trained checkpoint is small enough to commit directly, so no Git LFS is needed.

CORS is left open in `app.py` since the frontend and backend are served from the same origin and nothing external calls the API.
