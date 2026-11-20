"""Explicit-word tiers picked by Claude. LLM judgment; not used by the data pipeline.

Tiers:
  strong   Kidz Bop "profanity" words not in MILD, plus the STRONG set below
  mild     the MILD set below
  kb_only  other Kidz Bop censored words

STRONG and MILD were written by Claude, not taken from a dataset. Racial slurs appear only
through the Kidz Bop data, never in these sets.

Run:  python -m radio_ready.llm_judgment.claude_word_tiers
Needs data/artifacts/kidzbop_words.csv (python -m radio_ready.kidzbop).
Writes data/artifacts/llm_judgment/claude_word_tiers.csv.
"""

import re

import pandas as pd

from radio_ready import paths

STRONG = {
    "fuck", "motherfucker", "shit", "bullshit", "bitch", "ass", "asshole", "cunt", "pussy",
    "dick", "cock", "whore", "hoe", "slut", "faggot", "fag", "dyke", "twat", "tits", "titties",
    "cum", "jizz", "coochie",
}
MILD = {"damn", "dammit", "goddamn", "hell", "hellish", "hella", "crap", "piss", "bastard"}

# Stems matched inside longer words; other words need a word boundary plus an inflection.
ANYWHERE = {"fuck", "shit", "bitch", "cunt", "pussy"}
SUFFIX = r"(?:s|es|ed|d|ing|in'?|in|er|ers|y|ies|ey|z)?"

OUT_CSV = paths.ARTIFACTS / "llm_judgment" / "claude_word_tiers.csv"


def tier(word: str, category: str) -> str:
    if word in MILD:
        return "mild"
    if word in STRONG or category == "profanity":
        return "strong"
    return "kb_only"


def build() -> pd.DataFrame:
    kb = pd.read_csv(paths.KIDZBOP_WORDS_CSV, keep_default_na=False)
    words = kb[["word", "category"]].assign(source="kidzbop")
    extra = sorted((STRONG | MILD) - set(words["word"]))
    words = pd.concat([words, pd.DataFrame({"word": extra, "category": "", "source": "claude"})])
    words["tier"] = [tier(w, c) for w, c in zip(words["word"], words["category"])]
    return words.sort_values(["tier", "word"]).reset_index(drop=True)


class TierMatcher:
    """Regex matcher over the words of the given tiers (inflections, embedded stems)."""

    def __init__(self, tiers=("strong",)):
        table = pd.read_csv(OUT_CSV, keep_default_na=False)
        vocab = sorted(set(table.loc[table["tier"].isin(tiers), "word"]), key=len, reverse=True)
        parts = []
        for w in vocab:
            stem = next((s for s in ANYWHERE if w.startswith(s)), None)
            parts.append(rf"[a-z]*{re.escape(stem)}[a-z']*" if stem else rf"{re.escape(w)}{SUFFIX}")
        self.pattern = re.compile(
            r"(?<![a-z])(?:" + "|".join(sorted(set(parts), key=len, reverse=True)) + r")(?![a-z])",
            re.IGNORECASE,
        )

    def find(self, text: str) -> list[str]:
        return [m.lower() for m in self.pattern.findall(text)]


def main() -> None:
    table = build()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    print(table["tier"].value_counts().to_string())
    print(f"output: {OUT_CSV}")


if __name__ == "__main__":
    main()
