"""File locations shared by all pipeline steps."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
ARTIFACTS = DATA / "artifacts"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"

# Raw inputs
KIDZ_BOP_CSV = RAW / "kidz-bop" / "KB_censored-lyrics.csv"
GENIUS_DIR = RAW / "genius"
SURGE_CSV = RAW / "surge-profanity" / "profanity_en.csv"
CMUDICT_FILE = RAW / "cmudict" / "cmudict.dict"

# Human-edited files: committed, never regenerated
ANNOTATIONS = ROOT / "annotations"
VETTING_CSV = ANNOTATIONS / "explicit_word_vetting.csv"

# Generated
SONGS_PARQUET = PROCESSED / "labels" / "songs.parquet"
LEARNED_LEXICON_CSV = ARTIFACTS / "learned_explicit_words.csv"
LEXICON_METRICS_JSON = ARTIFACTS / "lexicon_metrics.json"
KIDZBOP_DIR = PROCESSED / "kidzbop"
KIDZBOP_WORDS_CSV = ARTIFACTS / "kidzbop_words.csv"
KIDZBOP_SUBSTITUTIONS_CSV = ARTIFACTS / "kidzbop_substitutions.csv"
GENIUS_SECTIONS_JSONL = PROCESSED / "genius_sections.jsonl"
CLASSIFIER_DIR = OUTPUTS / "explicit_classifier"
GENERATIONS_DIR = OUTPUTS / "generations"
COMPARISON_DIR = OUTPUTS / "comparison"

# Commercial-LLM rewrites: never overwritten by the pipeline
COMMERCIAL_DIR = DATA / "commercial"

# Training data
PARADEHATE_PARQUET = RAW / "paradehate" / "data" / "train-00000-of-00001.parquet"
PARADEHATE_PAIRS = PROCESSED / "paradehate" / "pairs.jsonl"
SFT_DIR = PROCESSED / "sft"
