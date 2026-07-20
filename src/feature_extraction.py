"""
feature_extraction.py
----------------------
Extracts audio features (MFCCs, mel-spectrogram, chroma, zero-crossing rate,
spectral contrast) from each RAVDESS file listed in metadata.csv, and saves
the results as:
    - data/processed/features.csv   (flat, tabular — good for classical ML)
    - data/processed/features.npy   (stacked array — good for deep learning)

Usage:
    python src/feature_extraction.py
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    METADATA_CSV,
    FEATURES_CSV,
    FEATURES_NPY,
    SAMPLE_RATE,
    DURATION,
    N_MFCC,
    N_MELS,
    FRAME_LENGTH,
    HOP_LENGTH,
    TRIM_SILENCE,
    TRIM_TOP_DB,
    AUGMENTATION_ENABLED,
    F0_MIN,
    F0_MAX,
    INCLUDE_DELTA_FEATURES,
    get_actor_splits,
)
from augmentation import generate_augmented_variants


def load_and_trim_audio(filepath: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load an audio file, resample to `sr`, and trim leading/trailing silence.
    Does NOT pad/truncate to a fixed length — that happens separately via
    fix_length(), since augmentation (e.g. time-stretch) changes clip length
    and needs to be re-fixed AFTER the transform is applied, not before.
    """
    y, _ = librosa.load(filepath, sr=sr)
    if TRIM_SILENCE:
        y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    return y


def fix_length(y: np.ndarray, sr: int = SAMPLE_RATE, duration: float = DURATION) -> np.ndarray:
    """Pads or truncates an audio array to a fixed length in seconds."""
    target_len = int(sr * duration)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    return y


def load_audio(filepath: str, sr: int = SAMPLE_RATE, duration: float = DURATION) -> np.ndarray:
    """
    Convenience wrapper: load, trim silence, and fix to a target length in
    one call. Used for the non-augmented path.
    """
    y = load_and_trim_audio(filepath, sr=sr)
    return fix_length(y, sr=sr, duration=duration)


def extract_pitch_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Extracts F0 (pitch) contour statistics using librosa's pyin algorithm.

    This directly targets prosody — pitch trajectory and variability — which
    plain spectral features (MFCC/mel/chroma) underrepresent. This is what
    PCA/t-SNE analysis showed was missing: valence-confused pairs like
    happy/angry and calm/sad share similar spectral energy but differ in
    pitch dynamics.
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=F0_MIN, fmax=F0_MAX, sr=sr
    )

    voiced_f0 = f0[~np.isnan(f0)]

    if len(voiced_f0) == 0:
        # Fully unvoiced/silent clip — return zeros rather than crashing
        return {
            "f0_mean": 0.0, "f0_std": 0.0, "f0_min": 0.0, "f0_max": 0.0,
            "f0_range": 0.0, "voiced_fraction": 0.0,
        }

    return {
        "f0_mean": float(np.mean(voiced_f0)),
        "f0_std": float(np.std(voiced_f0)),
        "f0_min": float(np.min(voiced_f0)),
        "f0_max": float(np.max(voiced_f0)),
        "f0_range": float(np.max(voiced_f0) - np.min(voiced_f0)),
        "voiced_fraction": float(np.mean(~np.isnan(f0))),  # fraction of frames with detected pitch
    }


def extract_features_flat(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """
    Extract scalar (mean/std) summary features for classical ML models
    (e.g. SVM, Random Forest, XGBoost) from a single audio clip.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                 n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                          n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                          n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr,
                                                            n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)

    feature_dict = {}

    def add_stats(name, arr):
        feature_dict[f"{name}_mean"] = np.mean(arr)
        feature_dict[f"{name}_std"] = np.std(arr)

    # Per-coefficient MFCC stats (40 coefficients -> 80 columns)
    for i in range(mfcc.shape[0]):
        add_stats(f"mfcc{i+1}", mfcc[i])

    for i in range(chroma.shape[0]):
        add_stats(f"chroma{i+1}", chroma[i])

    add_stats("mel", mel)
    add_stats("zcr", zcr)
    add_stats("spectral_contrast", spectral_contrast)
    add_stats("rms", rms)

    # Delta + delta-delta MFCC (rate of change) — captures how quickly
    # spectral content shifts, another prosody-adjacent signal missing
    # from static MFCC alone.
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    for i in range(mfcc_delta.shape[0]):
        add_stats(f"mfcc{i+1}_delta", mfcc_delta[i])
    for i in range(mfcc_delta2.shape[0]):
        add_stats(f"mfcc{i+1}_delta2", mfcc_delta2[i])

    # Pitch/F0 contour features
    feature_dict.update(extract_pitch_features(y, sr))

    return feature_dict


def extract_features_sequence(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract time-series mel-spectrogram channels for deep learning models
    (CNN / CNN-LSTM).

    If INCLUDE_DELTA_FEATURES is True, stacks 3 channels: static mel-spectrogram,
    its delta (first derivative — how fast energy is changing per mel band),
    and delta-delta (second derivative — acceleration of that change). This
    gives the CNN explicit access to spectral dynamics, not just static
    snapshots — directly addressing the prosody gap identified via PCA/t-SNE.

    Shape: (3, n_mels, time_steps) if deltas enabled, else (1, n_mels, time_steps).
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                          n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if INCLUDE_DELTA_FEATURES:
        mel_delta = librosa.feature.delta(mel_db)
        mel_delta2 = librosa.feature.delta(mel_db, order=2)
        stacked = np.stack([mel_db, mel_delta, mel_delta2], axis=0)  # (3, n_mels, time)
        return stacked

    return mel_db[np.newaxis, :, :]  # (1, n_mels, time) — keep channel dim consistent


def build_features(metadata_path: str = METADATA_CSV):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"metadata.csv not found at {metadata_path}. "
            f"Run data_parser.py first to generate it."
        )

    df = pd.read_csv(metadata_path)

    # Determine which actors belong to the TRAINING split. Augmentation is
    # only applied to these files — augmenting val/test would leak "easier"
    # synthetic variants into evaluation and inflate accuracy dishonestly.
    train_actors, val_actors, test_actors = get_actor_splits(df["actor_id"].unique())
    train_actors_set = set(train_actors)
    print(f"Train actors (eligible for augmentation): {sorted(train_actors_set)}")
    print(f"Val/test actors (original audio only, no augmentation): "
          f"{sorted(set(val_actors) | set(test_actors))}")

    flat_records = []
    sequence_features = []
    n_augmented = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:
            y_trimmed = load_and_trim_audio(row["filepath"])
            is_train_file = row["actor_id"] in train_actors_set

            # --- Always process the original (unaugmented) clip ---
            y_fixed = fix_length(y_trimmed)
            flat = extract_features_flat(y_fixed)
            flat.update({
                "filename": row["filename"],
                "emotion": row["emotion"],
                "actor_id": row["actor_id"],
                "gender": row["gender"],
                "is_augmented": False,
                "augmentation_type": "none",
            })
            flat_records.append(flat)
            sequence_features.append(extract_features_sequence(y_fixed))

            # --- Augmented variants: TRAIN actors only ---
            if AUGMENTATION_ENABLED and is_train_file:
                for variant_name, y_aug in generate_augmented_variants(y_trimmed):
                    y_aug_fixed = fix_length(y_aug)
                    flat_aug = extract_features_flat(y_aug_fixed)
                    flat_aug.update({
                        "filename": f"{row['filename'].replace('.wav', '')}_{variant_name}.wav",
                        "emotion": row["emotion"],
                        "actor_id": row["actor_id"],
                        "gender": row["gender"],
                        "is_augmented": True,
                        "augmentation_type": variant_name,
                    })
                    flat_records.append(flat_aug)
                    sequence_features.append(extract_features_sequence(y_aug_fixed))
                    n_augmented += 1

        except Exception as e:
            print(f"[ERROR] Failed to process {row['filename']}: {e}")

    flat_df = pd.DataFrame(flat_records)
    sequence_array = np.stack(sequence_features)  # shape: (n_samples, n_mels, time_steps)

    print(f"\nOriginal files: {len(df)} | Augmented variants added: {n_augmented} | "
          f"Total feature rows: {len(flat_df)}")

    return flat_df, sequence_array


def save_features(flat_df: pd.DataFrame, sequence_array: np.ndarray):
    flat_df.to_csv(FEATURES_CSV, index=False)
    np.save(FEATURES_NPY, sequence_array)
    print(f"\nSaved flat features (classical ML) to: {FEATURES_CSV}  shape={flat_df.shape}")
    print(f"Saved sequence features (deep learning) to: {FEATURES_NPY}  shape={sequence_array.shape}")


if __name__ == "__main__":
    flat_df, sequence_array = build_features()
    save_features(flat_df, sequence_array)