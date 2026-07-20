"""
Tests for data_parser.py — specifically parse_filename(), since it's the
single point where raw RAVDESS filenames become actor_id / emotion / gender
labels. A silent bug here corrupts every downstream label without raising
any error, so these tests check parsing correctness directly rather than
relying on model accuracy to reveal problems.

Run from project root:
    pytest tests/test_data_parser.py -v
"""

import os
import sys

import pytest

# PROJECT_ROOT = the folder that CONTAINS src/ (i.e. one level up from
# this tests/ folder). Derived from __file__, not os.getcwd(), so this
# works no matter which directory pytest is invoked from.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_parser import parse_filename
from src.config import (
    MODALITY_MAP,
    VOCAL_CHANNEL_MAP,
    EMOTION_MAP,
    INTENSITY_MAP,
    STATEMENT_MAP,
    REPETITION_MAP,
    EXPECTED_MODALITY,
    EXPECTED_VOCAL_CHANNEL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_filename(modality="03", vocal_channel="01", emotion="01",
                   intensity="01", statement="01", repetition="01", actor="01"):
    """Builds a syntactically valid RAVDESS-style filename from parts."""
    parts = [modality, vocal_channel, emotion, intensity, statement, repetition, actor]
    return "-".join(parts) + ".wav"


# ---------------------------------------------------------------------------
# Core parsing correctness
# ---------------------------------------------------------------------------

def test_parses_all_expected_fields():
    """A well-formed filename should populate every expected key."""
    filename = make_filename(emotion="06", actor="12")
    filepath = f"/fake/path/Actor_12/{filename}"

    result = parse_filename(filepath)

    assert result is not None
    expected_keys = {
        "filename", "filepath", "modality_code", "modality",
        "vocal_channel_code", "vocal_channel", "emotion_code", "emotion",
        "intensity_code", "intensity", "statement_code", "statement",
        "repetition_code", "repetition", "actor_code", "actor_id", "gender",
    }
    assert expected_keys.issubset(result.keys())


def test_emotion_code_maps_correctly():
    """Emotion label in the output must match config.EMOTION_MAP for that code."""
    for code, label in EMOTION_MAP.items():
        filename = make_filename(emotion=code)
        result = parse_filename(f"/fake/{filename}")
        assert result["emotion_code"] == code
        assert result["emotion"] == label


def test_modality_and_vocal_channel_map_correctly():
    for code, label in MODALITY_MAP.items():
        filename = make_filename(modality=code, vocal_channel=EXPECTED_VOCAL_CHANNEL)
        result = parse_filename(f"/fake/{filename}")
        assert result["modality"] == label

    for code, label in VOCAL_CHANNEL_MAP.items():
        filename = make_filename(vocal_channel=code, modality=EXPECTED_MODALITY)
        result = parse_filename(f"/fake/{filename}")
        assert result["vocal_channel"] == label


def test_intensity_statement_repetition_map_correctly():
    for code, label in INTENSITY_MAP.items():
        result = parse_filename(f"/fake/{make_filename(intensity=code)}")
        assert result["intensity"] == label

    for code, label in STATEMENT_MAP.items():
        result = parse_filename(f"/fake/{make_filename(statement=code)}")
        assert result["statement"] == label

    for code, label in REPETITION_MAP.items():
        result = parse_filename(f"/fake/{make_filename(repetition=code)}")
        assert result["repetition"] == label


# ---------------------------------------------------------------------------
# actor_id / gender derivation — this logic is duplicated implicitly across
# the pipeline (gender stratification in get_actor_splits relies on the same
# odd/even assumption), so it's worth locking down explicitly.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("actor_num,expected_gender", [
    (1, "male"),
    (3, "male"),
    (23, "male"),
    (2, "female"),
    (4, "female"),
    (24, "female"),
])
def test_actor_id_and_gender_derivation(actor_num, expected_gender):
    actor_code = f"{actor_num:02d}"
    filename = make_filename(actor=actor_code)
    result = parse_filename(f"/fake/{filename}")

    assert result["actor_id"] == actor_num
    assert isinstance(result["actor_id"], int)
    assert result["gender"] == expected_gender


def test_actor_id_is_int_not_str():
    result = parse_filename(f"/fake/{make_filename(actor='07')}")
    assert result["actor_id"] == 7
    assert type(result["actor_id"]) is int


# ---------------------------------------------------------------------------
# Malformed input handling
# ---------------------------------------------------------------------------

def test_malformed_filename_too_few_parts_returns_none():
    result = parse_filename("/fake/03-01-06-01.wav")
    assert result is None


def test_malformed_filename_too_many_parts_returns_none():
    result = parse_filename("/fake/03-01-06-01-02-01-12-99.wav")
    assert result is None


def test_unknown_code_falls_back_to_unknown_label():
    """Codes outside the known map should degrade gracefully, not crash."""
    filename = make_filename(emotion="99")
    result = parse_filename(f"/fake/{filename}")
    assert result is not None
    assert result["emotion_code"] == "99"
    assert result["emotion"] == "unknown"


def test_filepath_and_filename_fields_preserved():
    filename = make_filename()
    filepath = f"/some/nested/Actor_01/{filename}"
    result = parse_filename(filepath)

    assert result["filename"] == filename
    assert result["filepath"] == filepath


# ---------------------------------------------------------------------------
# Real RAVDESS filename samples (sanity-check against actual dataset naming)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected_actor,expected_gender", [
    ("03-01-01-01-01-01-01.wav", 1, "male"),      # neutral, actor 1
    ("03-01-05-02-02-01-14.wav", 14, "female"),   # angry-strong, actor 14
    ("03-01-08-01-01-02-24.wav", 24, "female"),   # surprised, actor 24
])
def test_real_ravdess_style_filenames(filename, expected_actor, expected_gender):
    result = parse_filename(f"/data/Audiodata/Actor_{expected_actor:02d}/{filename}")
    assert result is not None
    assert result["actor_id"] == expected_actor
    assert result["gender"] == expected_gender
    assert result["emotion_code"] in EMOTION_MAP