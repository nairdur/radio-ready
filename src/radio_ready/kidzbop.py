"""Score, rank and split the human-written Kidz Bop line pairs.

Each pair is (original line, Kidz Bop line). Pairs where Kidz Bop simply cut the verse are
dropped, duplicates (same song + same lines) are merged. Metrics per pair:

  similarity    cosine of all-distilroberta-v1 embeddings (meaning kept)
  tox_og/tox_kb unbiased-toxic-roberta probability for each side
  lex_kb        strong explicit words still present in the rewrite (lexicon)
  clean         combined classifier: 0 if lex_kb is non-empty, else 1 - tox_kb
  syl_og/syl_kb syllables per side; syl_score = 1 - |diff| / syl_og (floored at 0)
  rhyme_kept    end word unchanged or rhyming
  edit_ratio    share of words changed (diagnostic: too high = rewritten, not edited)

quality = similarity * clean * syl_score * (0.75 + 0.25 * rhyme_kept)
Ranking pool: radio-relevant pairs (original has a strong explicit word or tox_og >= 0.5),
minus near-duplicate lines within a song. The top fraction of the pool by quality is split by
song into train/val/test.

Run:  python -m radio_ready.kidzbop [--top 0.3]
"""

import argparse
import difflib
import json
import random

import numpy as np
import pandas as pd

from radio_ready import paths
from radio_ready.lexicon import CUT_MARKER, ExplicitMatcher
from radio_ready.scoring import Similarity, Toxicity
from radio_ready.text import end_rhyme_kept, normalize, syllables, words

OUT_DIR = paths.PROCESSED / "kidzbop"


def load_pairs() -> pd.DataFrame:
    kb = pd.read_csv(paths.KIDZ_BOP_CSV)
    for col in ("ogLyric", "kbLyric", "ogArtist", "songName"):
        kb[col] = kb[col].astype(str).map(normalize)
    kb = kb[kb["kbLyric"].str.lower() != CUT_MARKER]
    kb = kb[kb["ogLyric"].str.lower() != kb["kbLyric"].str.lower()]
    pairs = (
        kb.groupby(["ogArtist", "songName", "ogLyric", "kbLyric"], sort=False)
        .agg(badwords=("badword", lambda s: sorted(set(s))),
             categories=("category", lambda s: sorted(set(s))),
             year=("year", "first"))
        .reset_index()
        .rename(columns={"ogArtist": "artist", "songName": "song",
                         "ogLyric": "original", "kbLyric": "rewrite"})
    )
    return pairs


def edit_ratio(a: str, b: str) -> float:
    wa, wb = [w.lower() for w in words(a)], [w.lower() for w in words(b)]
    return 1 - difflib.SequenceMatcher(a=wa, b=wb, autojunk=False).ratio()


def score(pairs: pd.DataFrame) -> pd.DataFrame:
    og, kb = pairs["original"].tolist(), pairs["rewrite"].tolist()
    matcher = ExplicitMatcher()
    tox = Toxicity()
    pairs["similarity"] = Similarity().pairwise(og, kb).round(4)
    pairs["tox_og"] = tox.score(og).round(4)
    pairs["tox_kb"] = tox.score(kb).round(4)
    pairs["lex_og"] = [list(matcher.find(t)) for t in og]
    pairs["lex_kb"] = [list(matcher.find(t)) for t in kb]
    pairs["clean"] = np.where(pairs["lex_kb"].str.len() > 0, 0.0, 1 - pairs["tox_kb"]).round(4)
    pairs["syl_og"] = [syllables(t) for t in og]
    pairs["syl_kb"] = [syllables(t) for t in kb]
    diff = (pairs["syl_og"] - pairs["syl_kb"]).abs()
    pairs["syl_score"] = (1 - diff / pairs["syl_og"].clip(lower=1)).clip(lower=0).round(4)
    pairs["rhyme_kept"] = [end_rhyme_kept(a, b) for a, b in zip(og, kb)]
    pairs["edit_ratio"] = [round(edit_ratio(a, b), 4) for a, b in zip(og, kb)]
    pairs["quality"] = (
        pairs["similarity"].clip(lower=0) * pairs["clean"] * pairs["syl_score"]
        * (0.75 + 0.25 * pairs["rhyme_kept"])
    ).round(4)
    pairs = pairs.sort_values("quality", ascending=False).reset_index(drop=True)
    pairs.insert(0, "rank", pairs.index + 1)
    return pairs


def mark_near_duplicates(pairs: pd.DataFrame, threshold: float = 0.85) -> pd.Series:
    """Within a song, a line >= threshold similar (word level) to a better-ranked one is a dup."""
    dup = pd.Series(False, index=pairs.index)
    kept: dict[tuple, list[list[str]]] = {}
    for idx, row in pairs.iterrows():  # pairs are already sorted best-first
        key = (row["artist"], row["song"])
        toks = [w.lower() for w in words(row["original"])]
        if any(difflib.SequenceMatcher(a=toks, b=k).ratio() >= threshold for k in kept.get(key, [])):
            dup[idx] = True
        else:
            kept.setdefault(key, []).append(toks)
    return dup


def split_by_song(df: pd.DataFrame, ratios: tuple[float, float, float], seed: int) -> pd.Series:
    songs = sorted(set(zip(df["artist"], df["song"])))
    random.Random(seed).shuffle(songs)
    n_train = round(len(songs) * ratios[0])
    n_val = round(len(songs) * ratios[1])
    assign = {s: "train" for s in songs[:n_train]}
    assign |= {s: "val" for s in songs[n_train : n_train + n_val]}
    assign |= {s: "test" for s in songs[n_train + n_val :]}
    return pd.Series([assign[(a, s)] for a, s in zip(df["artist"], df["song"])], index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score, rank and split Kidz Bop pairs.")
    parser.add_argument("--top", type=float, default=0.3, help="fraction of pairs to keep")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.6, 0.2, 0.2))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = score(load_pairs())
    # Radio-relevant: the original has a strong explicit word or is clearly toxic. Kidz Bop also
    # censors words radio keeps (kiss, drink, damn); those near-copies would dominate the ranking.
    pairs["radio_relevant"] = (pairs["lex_og"].str.len() > 0) | (pairs["tox_og"] >= 0.5)
    pairs["near_duplicate"] = mark_near_duplicates(pairs)
    pool = pairs[pairs["radio_relevant"] & ~pairs["near_duplicate"]]
    n_keep = max(1, round(len(pool) * args.top))
    pairs["selected"] = pairs.index.isin(pool.index[:n_keep])
    pairs["split"] = ""
    top = pairs[pairs["selected"]]
    pairs.loc[top.index, "split"] = split_by_song(top, tuple(args.ratios), args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT_DIR / "pairs_scored.csv", index=False)
    keep = ["artist", "song", "original", "rewrite", "badwords", "similarity", "clean",
            "syl_og", "syl_kb", "rhyme_kept", "quality"]
    for split in ("train", "val", "test"):
        part = pairs[pairs["split"] == split][keep]
        with open(OUT_DIR / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for rec in part.to_dict("records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"pairs after cleaning: {len(pairs)}, radio-relevant: {pairs['radio_relevant'].sum()}, "
          f"near-duplicates: {pairs['near_duplicate'].sum()}, pool: {len(pool)}")
    print(f"selected top {args.top:.0%} of pool: {n_keep}, "
          f"quality cutoff: {pool['quality'].iloc[n_keep - 1]:.3f}")
    print(pairs[pairs["selected"]]["split"].value_counts().to_string())
    cols = ["similarity", "clean", "syl_score", "rhyme_kept", "edit_ratio", "quality"]
    print(pairs.groupby("selected")[cols].mean().round(3).to_string())
    print(f"outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
