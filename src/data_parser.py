"""
data_parser.py
---------------
Walks the RAVDESS Audiodata/Actor_XX/ folders, parses each 7-part filename,
validates it against expected dataset constraints, and builds a metadata
DataFrame / CSV that maps every audio file to its labels.

Filename format:
    modality-vocalChannel-emotion-intensity-statement-repetition-actor.wav
    e.g. 03-01-06-01-02-01-12.wav

Usage:
    python src/data_parser.py
"""

import os
import glob
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    METADATA_CSV,
    MODALITY_MAP,
    VOCAL_CHANNEL_MAP,
    EMOTION_MAP,
    INTENSITY_MAP,
    STATEMENT_MAP,
    REPETITION_MAP,
    EXPECTED_MODALITY,
    EXPECTED_VOCAL_CHANNEL,
    TOTAL_EXPECTED_FILES,
)


def parse_filename(filepath: str) -> dict:
    """
    Parse a single RAVDESS filename into its labeled components.

    Parameters
    ----------
    filepath : str
        Full path to the .wav file.

    Returns
    -------
    dict
        Parsed identifiers + human-readable labels for this file.
        Returns None if the filename doesn't match the expected 7-part format.
    """
    filename = os.path.basename(filepath)
    name_no_ext = os.path.splitext(filename)[0]
    parts = name_no_ext.split("-")

    if len(parts) != 7:
        print(f"[WARN] Skipping malformed filename: {filename}")
        return None

    modality, vocal_channel, emotion, intensity, statement, repetition, actor = parts

    # --- Validate against this dataset's known constraints ---
    if modality != EXPECTED_MODALITY:
        print(f"[WARN] Unexpected modality '{modality}' in {filename} — expected audio-only ('03')")
    if vocal_channel != EXPECTED_VOCAL_CHANNEL:
        print(f"[WARN] Unexpected vocal channel '{vocal_channel}' in {filename} — expected speech ('01')")

    # Neutral emotion has no "strong" intensity in RAVDESS — flag if seen
    if emotion == "01" and intensity == "02":
        print(f"[WARN] Neutral with 'strong' intensity is not expected: {filename}")

    record = {
        "filename": filename,
        "filepath": filepath,
        "modality_code": modality,
        "modality": MODALITY_MAP.get(modality, "unknown"),
        "vocal_channel_code": vocal_channel,
        "vocal_channel": VOCAL_CHANNEL_MAP.get(vocal_channel, "unknown"),
        "emotion_code": emotion,
        "emotion": EMOTION_MAP.get(emotion, "unknown"),
        "intensity_code": intensity,
        "intensity": INTENSITY_MAP.get(intensity, "unknown"),
        "statement_code": statement,
        "statement": STATEMENT_MAP.get(statement, "unknown"),
        "repetition_code": repetition,
        "repetition": REPETITION_MAP.get(repetition, "unknown"),
        "actor_code": actor,
        "actor_id": int(actor),
        "gender": "male" if int(actor) % 2 != 0 else "female",
    }
    return record


def build_metadata(raw_data_dir: str = RAW_DATA_DIR) -> pd.DataFrame:
    """
    Scan all Actor_XX folders under raw_data_dir, parse every .wav file,
    and return a metadata DataFrame.
    """
    if not os.path.isdir(raw_data_dir):
        raise FileNotFoundError(
            f"Could not find raw data directory: {raw_data_dir}\n"
            f"Make sure your 'Audiodata' folder is placed at the project root."
        )

    actor_folders = sorted(glob.glob(os.path.join(raw_data_dir, "Actor_*")))
    if not actor_folders:
        raise FileNotFoundError(
            f"No 'Actor_*' folders found inside {raw_data_dir}. "
            f"Check that the dataset was extracted correctly."
        )

    records = []
    for actor_folder in actor_folders:
        wav_files = sorted(glob.glob(os.path.join(actor_folder, "*.wav")))
        for wav_path in wav_files:
            parsed = parse_filename(wav_path)
            if parsed is not None:
                records.append(parsed)

    df = pd.DataFrame(records)

    # --- Sanity checks ---
    print(f"\nParsed {len(df)} files from {len(actor_folders)} actor folders.")
    if len(df) != TOTAL_EXPECTED_FILES:
        print(
            f"[WARN] Expected {TOTAL_EXPECTED_FILES} files total, "
            f"but found {len(df)}. Check for missing/extra files."
        )

    return df


def save_metadata(df: pd.DataFrame, output_path: str = METADATA_CSV) -> None:
    df.to_csv(output_path, index=False)
    print(f"Saved metadata CSV to: {output_path}")


if __name__ == "__main__":
    metadata_df = build_metadata()
    save_metadata(metadata_df)

    # Quick summary printout
    print("\n--- Emotion class distribution ---")
    print(metadata_df["emotion"].value_counts())

    print("\n--- Gender distribution ---")
    print(metadata_df["gender"].value_counts())

    print("\n--- Sample rows ---")
    print(metadata_df.head())