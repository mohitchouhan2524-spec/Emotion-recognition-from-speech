"""
dataset.py
----------
Wraps the extracted sequence features (mel-spectrograms, from
feature_extraction.py) and their emotion labels into a PyTorch Dataset,
plus a helper to build train/val/test DataLoaders.

Usage:
    python src/dataset.py     # quick sanity check / shape printout
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    FEATURES_CSV,
    FEATURES_NPY,
    EMOTION_LABELS,
    RANDOM_SEED,
    get_actor_splits,
)


class RAVDESSDataset(Dataset):
    """
    PyTorch Dataset wrapping mel-spectrogram sequences and emotion labels.

    Each item returns:
        spectrogram : FloatTensor, shape (1, n_mels, time_steps)  [1 channel for CNN input]
        label       : LongTensor, encoded emotion class index
    """

    def __init__(self, spectrograms: np.ndarray, labels: np.ndarray):
        # spectrograms already have shape (N, channels, n_mels, time) — channel
        # dim (mel + delta + delta2) is added by feature_extraction.py now,
        # so no manual unsqueeze needed here (previously: unsqueeze(1) for a
        # single-channel mel-only input).
        self.spectrograms = torch.tensor(spectrograms, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]


def load_features_and_labels(features_csv: str = FEATURES_CSV, features_npy: str = FEATURES_NPY):
    """
    Loads the flat features CSV (for labels/metadata alignment) and the
    sequence feature array (for model input), and returns them aligned.
    """
    if not os.path.exists(features_csv) or not os.path.exists(features_npy):
        raise FileNotFoundError(
            "Feature files not found. Run feature_extraction.py first to "
            f"generate {features_csv} and {features_npy}."
        )

    flat_df = pd.read_csv(features_csv)
    sequences = np.load(features_npy)

    if len(flat_df) != len(sequences):
        raise ValueError(
            f"Mismatch: features.csv has {len(flat_df)} rows but "
            f"features.npy has {len(sequences)} entries. Re-run feature_extraction.py."
        )

    label_encoder = LabelEncoder()
    label_encoder.fit(EMOTION_LABELS)  # fixed, known class order from config
    labels = label_encoder.transform(flat_df["emotion"].values)

    return sequences, labels, label_encoder, flat_df


def get_dataloaders(batch_size: int = 32, num_workers: int = 0):
    """
    Builds train/val/test DataLoaders with an ACTOR-INDEPENDENT split.

    Why actor-independent: RAVDESS has 24 actors, each recording all 8
    emotions. A sample-level split (splitting individual clips randomly)
    lets the same actor's voice appear in train, val, AND test — the model
    can partly learn to recognize *voices* rather than *emotions*, which
    inflates accuracy and doesn't reflect real-world use (predicting on a
    speaker the model has never heard). Here we split whole actors instead,
    stratified by gender so both splits stay balanced.

    Returns
    -------
    train_loader, val_loader, test_loader, label_encoder
    """
    sequences, labels, label_encoder, flat_df = load_features_and_labels()

    unique_actors = flat_df["actor_id"].unique()
    train_actors, val_actors, test_actors = get_actor_splits(unique_actors)

    print(f"Train actors ({len(train_actors)}): {sorted(train_actors)}")
    print(f"Val actors   ({len(val_actors)}): {sorted(val_actors)}")
    print(f"Test actors  ({len(test_actors)}): {sorted(test_actors)}")

    train_mask = flat_df["actor_id"].isin(train_actors).values
    val_mask = flat_df["actor_id"].isin(val_actors).values
    test_mask = flat_df["actor_id"].isin(test_actors).values

    # Safety check: augmented samples must NEVER appear in val/test — this
    # would leak synthetic "easy" variants into evaluation and inflate
    # accuracy dishonestly. feature_extraction.py should already guarantee
    # this, but we verify it here too since it's a correctness-critical
    # invariant for the whole augmentation feature.
    if "is_augmented" in flat_df.columns:
        val_aug_count = flat_df.loc[val_mask, "is_augmented"].sum()
        test_aug_count = flat_df.loc[test_mask, "is_augmented"].sum()
        assert val_aug_count == 0, f"{val_aug_count} augmented samples leaked into val set!"
        assert test_aug_count == 0, f"{test_aug_count} augmented samples leaked into test set!"
        train_aug_count = flat_df.loc[train_mask, "is_augmented"].sum()
        print(f"Augmented samples — train: {train_aug_count}, val: {val_aug_count} (must be 0), "
              f"test: {test_aug_count} (must be 0)")

    X_train, y_train = sequences[train_mask], labels[train_mask]
    X_val, y_val = sequences[val_mask], labels[val_mask]
    X_test, y_test = sequences[test_mask], labels[test_mask]

    train_dataset = RAVDESSDataset(X_train, y_train)
    val_dataset = RAVDESSDataset(X_val, y_val)
    test_dataset = RAVDESSDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}")
    print(f"Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

    return train_loader, val_loader, test_loader, label_encoder


if __name__ == "__main__":
    train_loader, val_loader, test_loader, label_encoder = get_dataloaders(batch_size=32)

    # Sanity check: pull one batch and print shapes
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"\nSample batch shape: {sample_batch.shape}")   # (batch_size, 1, n_mels, time_steps)
    print(f"Sample labels shape: {sample_labels.shape}")
    print(f"Sample label values: {sample_labels[:8].tolist()}")