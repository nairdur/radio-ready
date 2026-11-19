"""Explicit-word lexicon built from Kidz Bop censorship data, plus a matcher.

Kidz Bop censors far more than radio does (kiss, body, god, ...), so every word gets a tier:
  strong  - always bleeped on radio (profanity, slurs, explicit sexual terms)
  mild    - often left in radio edits (damn, hell, ...); tracked for metrics only
  kb_only - censored by Kidz Bop but not radio-explicit

Run:  python -m radio_ready.lexicon
Writes data/artifacts/explicit_words.csv and data/artifacts/kidzbop_substitutions.csv.
"""

import difflib
import re
from collections import Counter, defaultdict
from functools import lru_cache

import pandas as pd

from radio_ready import paths
from radio_ready.text import normalize, words

CUT_MARKER = "cuts verse"

# Tiering: Kidz Bop "profanity" words are strong unless listed in MILD. The sets below are
# curated additions/overrides for sexual terms and slurs Kidz Bop does not label as profanity;
# racial slurs come only from the dataset, never from this list.
STRONG = {
    "fuck", "motherfucker", "shit", "bullshit", "bitch", "ass", "asshole", "cunt", "pussy",
    "dick", "cock", "whore", "hoe", "slut", "faggot", "fag", "dyke", "twat", "tits", "titties",
    "cum", "jizz", "coochie",
}
MILD = {"damn", "dammit", "goddamn", "hell", "hellish", "hella", "crap", "piss", "bastard"}

# Matched inside longer words (motherfuckin', bullshit); everything else needs a word
# boundary plus an optional inflection so "ass" never matches "class" or "assume".
ANYWHERE = {"fuck", "shit", "bitch", "cunt", "pussy"}
SUFFIX = r"(?:s|es|ed|d|ing|in'?|in|er|ers|y|ies|ey|z)?"
CENSORED = re.compile(r"(?<![\w*])[A-Za-z]+\*+[A-Za-z*]*")


def _diff_replacements(og: str, kb: str, badword: str) -> list[str]:
    """Word-level diff: what replaced the token(s) containing the bad word."""
    a = [w.lower() for w in words(og)]
    b = [w.lower() for w in words(kb)]
    found = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ("replace", "delete") and any(badword in w for w in a[i1:i2]):
            found.append(" ".join(b[j1:j2]) or "<deleted>")
    return found


def build_lexicon() -> tuple[pd.DataFrame, pd.DataFrame]:
    kb = pd.read_csv(paths.KIDZ_BOP_CSV)
    kb["ogLyric"] = kb["ogLyric"].astype(str).map(normalize)
    kb["kbLyric"] = kb["kbLyric"].astype(str).map(normalize)
    kb["badword"] = kb["badword"].str.lower().str.strip()
    kb["cut"] = kb["kbLyric"].str.lower() == CUT_MARKER

    subs: Counter = Counter()
    per_word: dict[str, Counter] = defaultdict(Counter)
    unique_rows = kb[~kb["cut"]].drop_duplicates(["songName", "ogLyric", "kbLyric", "badword"])
    for row in unique_rows.itertuples():
        for rep in _diff_replacements(row.ogLyric, row.kbLyric, row.badword):
            subs[(row.badword, rep)] += 1
            per_word[row.badword][rep] += 1

    stats = kb.groupby("badword").agg(
        category=("category", "first"),
        kb_rows=("badword", "size"),
        kb_occurrences=("count", "sum"),
        kb_songs=("songName", "nunique"),
        kb_artists=("ogArtist", "nunique"),
        kb_cut_rate=("cut", "mean"),
    )
    stats["source"] = "kidzbop"

    extra = sorted((STRONG | MILD) - set(stats.index))
    stats = pd.concat([stats, pd.DataFrame(index=extra, data={"source": "curated"})])
    stats.index.name = "word"
    stats = stats.reset_index()

    def tier(row) -> str:
        if row.word in MILD:
            return "mild"
        if row.word in STRONG or row.category == "profanity":
            return "strong"
        return "kb_only"

    stats["tier"] = stats.apply(tier, axis=1)
    stats["top_replacements"] = stats["word"].map(
        lambda w: "; ".join(f"{r} ({n})" for r, n in per_word[w].most_common(5))
    )
    order = {"strong": 0, "mild": 1, "kb_only": 2}
    stats = stats.sort_values(
        ["tier", "kb_rows", "word"], key=lambda s: s.map(order) if s.name == "tier" else s,
        ascending=[True, False, True],
    )
    stats["kb_cut_rate"] = stats["kb_cut_rate"].round(3)
    for col in ("kb_rows", "kb_occurrences", "kb_songs", "kb_artists"):
        stats[col] = stats[col].astype("Int64")
    cols = ["word", "tier", "category", "source", "kb_rows", "kb_occurrences", "kb_songs",
            "kb_artists", "kb_cut_rate", "top_replacements"]

    sub_df = pd.DataFrame(
        [(w, r, n) for (w, r), n in subs.items()], columns=["badword", "replacement", "count"]
    ).sort_values(["count", "badword"], ascending=[False, True])
    return stats[cols], sub_df


class ExplicitMatcher:
    """Finds explicit words of the given tiers; also flags already-censored tokens (f*ck)."""

    def __init__(self, tiers=("strong",)):
        lex = pd.read_csv(paths.LEXICON_CSV)
        vocab = sorted(set(lex.loc[lex["tier"].isin(tiers), "word"]), key=len, reverse=True)
        parts = []
        for w in vocab:
            stem = next((s for s in ANYWHERE if w.startswith(s)), None)
            if stem:
                parts.append(rf"[a-z]*{re.escape(stem)}[a-z']*")
            else:
                parts.append(rf"{re.escape(w)}{SUFFIX}")
        self.pattern = re.compile(
            r"(?<![a-z])(?:" + "|".join(sorted(set(parts), key=len, reverse=True)) + r")(?![a-z])",
            re.IGNORECASE,
        )

    @lru_cache(maxsize=100_000)
    def find(self, text: str) -> tuple[str, ...]:
        hits = [m.lower() for m in self.pattern.findall(text)]
        hits += [m.lower() for m in CENSORED.findall(text)]
        return tuple(hits)

    def is_explicit(self, text: str) -> bool:
        return bool(self.find(text))


def main() -> None:
    lexicon, subs = build_lexicon()
    paths.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    lexicon.to_csv(paths.LEXICON_CSV, index=False)
    subs.to_csv(paths.SUBSTITUTIONS_CSV, index=False)
    print(lexicon["tier"].value_counts().to_string())
    print(f"lexicon: {paths.LEXICON_CSV}\nsubstitutions: {paths.SUBSTITUTIONS_CSV}")


if __name__ == "__main__":
    main()
