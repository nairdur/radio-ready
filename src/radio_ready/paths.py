"""Shared data locations."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
ARTIFACTS = DATA / "artifacts"
PROCESSED = DATA / "processed"

KIDZ_BOP_CSV = RAW / "kidz-bop" / "KB_censored-lyrics.csv"
GENIUS_DIR = RAW / "genius"
LEXICON_CSV = ARTIFACTS / "explicit_words.csv"
SUBSTITUTIONS_CSV = ARTIFACTS / "kidzbop_substitutions.csv"
