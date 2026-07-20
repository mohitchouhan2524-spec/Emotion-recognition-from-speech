"""
augmentation.py
----------------
Audio augmentation functions for training-set expansion.

IMPORTANT: These functions must only ever be applied to TRAINING split
audio. Applying them to validation/test data would let "easier" synthetic
variants leak into evaluation and produce dishonestly inflated accuracy —
the exact class of bug we already fixed once with the actor-independent
split. feature_extraction.py enforces this by only calling these functions
for files belonging to train actors (see config.get_actor_splits).

Usage (standalone sanity check):
    python src/augmentation.py
"""

import os
import sys
import numpy as np
import librosa

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    SAMPLE_RATE,
    PITCH_SHIFT_STEPS,
    TIME_STRETCH_RATES,
    NOISE_FACTOR,
)


def pitch_shift(y: np.ndarray, sr: int = SAMPLE_RATE, n_steps: float = 2) -> np.ndarray:
    """Shifts pitch up/down by n_steps semitones without changing duration."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def time_stretch(y: np.ndarray, rate: float = 1.1) -> np.ndarray:
    """
    Speeds up (rate > 1.0) or slows down (rate < 1.0) the audio without
    changing pitch. Output length changes — caller must re-pad/truncate.
    """
    return librosa.effects.time_stretch(y, rate=rate)


def add_noise(y: np.ndarray, noise_factor: float = NOISE_FACTOR) -> np.ndarray:
    """Adds low-level white noise, scaled relative to the clip's own amplitude."""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise * np.max(np.abs(y))


def generate_augmented_variants(y: np.ndarray, sr: int = SAMPLE_RATE) -> list:
    """
    Given a clean (already-trimmed) audio array, generates all configured
    augmented variants.

    Returns
    -------
    list of (variant_name, augmented_audio) tuples.
    """
    variants = []

    for steps in PITCH_SHIFT_STEPS:
        try:
            aug = pitch_shift(y, sr=sr, n_steps=steps)
            variants.append((f"pitch{steps:+d}", aug))
        except Exception as e:
            print(f"[WARN] pitch_shift({steps}) failed: {e}")

    for rate in TIME_STRETCH_RATES:
        try:
            aug = time_stretch(y, rate=rate)
            variants.append((f"stretch{rate}", aug))
        except Exception as e:
            print(f"[WARN] time_stretch({rate}) failed: {e}")

    try:
        aug = add_noise(y)
        variants.append(("noise", aug))
    except Exception as e:
        print(f"[WARN] add_noise failed: {e}")

    return variants


if __name__ == "__main__":
    # Quick sanity check with synthetic audio (no dataset file needed)
    dummy_sr = SAMPLE_RATE
    dummy_y = 0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 2, dummy_sr * 2))

    variants = generate_augmented_variants(dummy_y, sr=dummy_sr)
    print(f"Generated {len(variants)} augmented variants from 1 dummy clip:")
    for name, y_aug in variants:
        print(f"  {name:<12} -> length={len(y_aug)} samples ({len(y_aug)/dummy_sr:.2f}s)")