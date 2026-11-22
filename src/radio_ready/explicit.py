"""Explicit-word detection used by the pipeline (Genius sections, Kidz Bop pairs).

Word lists:
  combined learned + Surge words (severity >= SURGE_MIN_SEVERITY) that are also among the
           learned candidates (learned_explicit_words.csv, after vetting) + every Surge word
           rated Strong or Severe (severity >= SURGE_ALWAYS_SEVERITY: slurs such as the
           hard-r n-word, which are too rare in the labelled songs to be learned), except
           HOMOGRAPHS; minus words vetted not_explicit; default
  learned  lookup words from `learned_lexicon` (Fell et al. 2019) after human vetting:
           rows of learned_explicit_words.csv with in_lookup = True, plus every word with
           verdict = explicit in annotations/explicit_word_vetting.csv
  surge    Surge AI Obscenity List, single-word entries with severity >= SURGE_MIN_SEVERITY
           (1.0 = every entry; the ratings run 1-3)

A text is explicit if any token (text.tokenize) is in the list. Already-censored tokens such
as "f*ck" are also reported.
"""

import re

import pandas as pd

from radio_ready import paths
from radio_ready.text import tokenize

# f*ck, b*tches, f*** ; not stage directions such as "*horn honks*"
CENSORED = re.compile(r"(?<![\w*])[A-Za-z]+(?:\*+[A-Za-z]+|\*{2,})[A-Za-z*]*")
SURGE_MIN_SEVERITY = 1.0
SURGE_ALWAYS_SEVERITY = 1.6  # Surge "Strong" (1.6-2.4) and "Severe" (2.6-3.0)
# Strong/Severe Surge entries that are mostly ordinary words in lyrics ("wop wop", "nip it")
HOMOGRAPHS = frozenset({"wop", "nip", "tramp", "bender"})


def learned_words() -> set[str]:
    lex = pd.read_csv(paths.LEARNED_LEXICON_CSV, keep_default_na=False)
    found = set(lex.loc[lex["in_lookup"].astype(str) == "True", "word"])
    if paths.VETTING_CSV.exists():
        vet = pd.read_csv(paths.VETTING_CSV, dtype=str, keep_default_na=False)
        found |= set(vet.loc[vet["verdict"].str.strip().str.lower() == "explicit", "word"].str.lower())
        found -= set(vet.loc[vet["verdict"].str.strip().str.lower() == "not_explicit", "word"].str.lower())
    return found


def surge_words(min_severity: float = SURGE_MIN_SEVERITY) -> set[str]:
    s = pd.read_csv(paths.SURGE_CSV)
    s["text"] = s["text"].str.lower().str.strip()
    s = s[(s["severity_rating"] >= min_severity) & ~s["text"].str.contains(" ")]
    return set(s["text"])


def learned_candidates() -> set[str]:
    return set(pd.read_csv(paths.LEARNED_LEXICON_CSV, keep_default_na=False)["word"])


def not_explicit_words() -> set[str]:
    if not paths.VETTING_CSV.exists():
        return set()
    vet = pd.read_csv(paths.VETTING_CSV, dtype=str, keep_default_na=False)
    return set(vet.loc[vet["verdict"].str.strip().str.lower() == "not_explicit", "word"]
               .str.lower())


def combined_words() -> set[str]:
    always = surge_words(SURGE_ALWAYS_SEVERITY) - HOMOGRAPHS
    return (learned_words() | (surge_words() & learned_candidates()) | always)         - not_explicit_words()


WORD_LISTS = {"combined": combined_words, "learned": learned_words, "surge": surge_words}


class ExplicitDetector:
    def __init__(self, source: str = "combined"):
        self.source = source
        self.words = frozenset(WORD_LISTS[source]())

    def find(self, text: str) -> list[str]:
        hits = [t for t in tokenize(text) if t in self.words]
        return hits + [m.lower() for m in CENSORED.findall(text)]

    def is_explicit(self, text: str) -> bool:
        return bool(self.find(text))
