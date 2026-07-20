"""
config.py
---------
Central configuration for the RAVDESS Speech Emotion Recognition project.
All paths, constants, and label mappings live here so every other script
can import from a single source of truth.
"""

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "Audiodata")

# Processed data outputs
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
METADATA_CSV = os.path.join(DATA_DIR, "metadata.csv")
FEATURES_CSV = os.path.join(PROCESSED_DIR, "features.csv")
FEATURES_NPY = os.path.join(PROCESSED_DIR, "features.npy")

# Model + results directories
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pt")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

# Ensure key directories exist (safe no-ops if they already do)
for _dir in [DATA_DIR, PROCESSED_DIR, MODELS_DIR, CHECKPOINTS_DIR,
             RESULTS_DIR, LOGS_DIR, PLOTS_DIR, REPORTS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# -------------------------------------------------------------------
# RAVDESS FILENAME IDENTIFIER MAPPINGS
# Filename format: modality-vocalChannel-emotion-intensity-statement-repetition-actor.wav
# Example: 03-01-06-01-02-01-12.wav
# -------------------------------------------------------------------

MODALITY_MAP = {
    "01": "full_AV",
    "02": "video_only",
    "03": "audio_only",
}

VOCAL_CHANNEL_MAP = {
    "01": "speech",
    "02": "song",
}

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

INTENSITY_MAP = {
    "01": "normal",
    "02": "strong",
}

STATEMENT_MAP = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}

REPETITION_MAP = {
    "01": "1st_repetition",
    "02": "2nd_repetition",
}

EXPECTED_MODALITY = "03"       # audio_only
EXPECTED_VOCAL_CHANNEL = "01"  # speech

N_ACTORS = 24
TRIALS_PER_ACTOR = 60
TOTAL_EXPECTED_FILES = N_ACTORS * TRIALS_PER_ACTOR  # 1440

# -------------------------------------------------------------------
# AUDIO / FEATURE EXTRACTION SETTINGS
# -------------------------------------------------------------------
SAMPLE_RATE = 22050        # librosa default; RAVDESS source is 48kHz, we resample
DURATION = 4.0             # seconds to pad to 4.0 s
TRIM_SILENCE = True        # trim leading/trailing silence before padding/truncating
TRIM_TOP_DB = 25           # silence threshold in dB below peak; lower = more aggressive trim
N_MFCC = 40

F0_MIN = 50    # Hz, roughly the lowest pitch a human voice reaches
F0_MAX = 500   # Hz, roughly the highest pitch a human voice reaches
INCLUDE_DELTA_FEATURES = True  # add delta + delta-delta mel channels (3-channel input)
N_MELS = 128
FRAME_LENGTH = 2048
HOP_LENGTH = 512

# -------------------------------------------------------------------
# TRAIN/TEST SPLIT & MODEL SETTINGS
# -------------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.167
VAL_SIZE = 0.167

EMOTION_LABELS = list(EMOTION_MAP.values())  # ordered class list for model output
NUM_CLASSES = len(EMOTION_LABELS)


def get_actor_splits(unique_actors):
    """
    Single source of truth for the actor-independent train/val/test split.
    Used by BOTH feature_extraction.py (to decide which files are safe to
    augment) and dataset.py (to build the actual DataLoaders), so the two
    scripts can never disagree about which actors belong to which split.

    Stratifies by gender (RAVDESS alternates male/female by actor ID) so
    both splits stay balanced.

    Parameters
    ----------
    unique_actors : list[int]
        Sorted list of actor IDs present in the dataset (e.g. 1..24).

    Returns
    -------
    train_actors, val_actors, test_actors : list[int]
    """
    from sklearn.model_selection import train_test_split

    unique_actors = sorted(unique_actors)
    actor_gender = ["male" if a % 2 != 0 else "female" for a in unique_actors]

    train_actors, temp_actors = train_test_split(
        unique_actors,
        test_size=(TEST_SIZE + VAL_SIZE),
        stratify=actor_gender,
        random_state=RANDOM_SEED,
    )

    temp_gender = ["male" if a % 2 != 0 else "female" for a in temp_actors]
    test_relative_size = TEST_SIZE / (TEST_SIZE + VAL_SIZE)
    val_actors, test_actors = train_test_split(
        temp_actors,
        test_size=test_relative_size,
        stratify=temp_gender,
        random_state=RANDOM_SEED,
    )

    return train_actors, val_actors, test_actors


# -------------------------------------------------------------------
# AUGMENTATION SETTINGS
# Applied ONLY to training-split files (see get_actor_splits above) —
# augmenting val/test would leak artificially "easier" variants into
# evaluation and inflate accuracy dishonestly.
# -------------------------------------------------------------------
AUGMENTATION_ENABLED = True
PITCH_SHIFT_STEPS = [-2, 2]      # semitones; one variant per value
TIME_STRETCH_RATES = [0.9, 1.1]  # <1.0 = slower, >1.0 = faster; one variant per value
NOISE_FACTOR = 0.005             # relative amplitude of injected white noise
# With 2 pitch + 2 stretch + 1 noise variant, each train file yields
# 1 (original) + 5 (augmented) = 6x the training samples.