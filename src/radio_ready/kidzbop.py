"""Compile Kidz Bop rewrites into DPO negatives, with Genius section context.

Kidz Bop edits are treated as bad radio edits to train against, not as targets. Each
(original line, Kidz Bop line) gets rule-based failure tags (no LLM judging):

  cut             Kidz Bop deleted the line ("cuts verse"); rejected = empty line
  over_censored   original has no strong explicit word and tox_og < 0.5 (kiss, drink, god),
                  so the right radio edit is no change; chosen = original
  still_explicit  rewrite still has a strong explicit word or tox_kb >= 0.9
  meaning_drift   similarity < 0.70 (ParaDeHate threshold)
  syllable_break  syllable count changed by >= 2
  rhyme_break     end word neither kept nor rhyming

Rows with at least one tag become negatives. over_censored rows are complete pairs now; the
rest need a chosen rewrite from synthetic generation (`needs_chosen`). Rows with no tag are
acceptable edits and are not used.

Context: each Kidz Bop song is looked up in Genius (fuzzy title + artist match), and the
original line is located in the cleaned sections, so DPO prompts can use the same
section-level format as synthetic generation. Splits are by song.

Run:  python -m radio_ready.kidzbop
Writes data/processed/kidzbop/{pairs_scored.csv, dpo_negatives.jsonl}
"""

import argparse
import difflib
import json
import random
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from radio_ready import paths
from radio_ready.genius import KidzBopSongs, split_sections
from radio_ready.lexicon import CUT_MARKER, ExplicitMatcher
from radio_ready.scoring import (
    SIMILARITY_THRESHOLD, TOXICITY_THRESHOLD, Similarity, Toxicity,
)
from radio_ready.text import end_rhyme_kept, normalize, split_adlibs, syllables, words

OUT_DIR = paths.PROCESSED / "kidzbop"
LINE_MATCH_THRESHOLD = 0.75


def load_pairs() -> pd.DataFrame:
    kb = pd.read_csv(paths.KIDZ_BOP_CSV)
    for col in ("ogLyric", "kbLyric", "ogArtist", "songName"):
        kb[col] = kb[col].astype(str).map(normalize)
    kb["cut"] = kb["kbLyric"].str.lower() == CUT_MARKER
    kb.loc[kb["cut"], "kbLyric"] = ""
    kb = kb[kb["ogLyric"].str.lower() != kb["kbLyric"].str.lower()]
    return (
        kb.groupby(["ogArtist", "songName", "ogLyric", "kbLyric", "cut"], sort=False)
        .agg(badwords=("badword", lambda s: sorted(set(s))),
             categories=("category", lambda s: sorted(set(s))),
             year=("year", "first"))
        .reset_index()
        .rename(columns={"ogArtist": "artist", "songName": "song",
                         "ogLyric": "original", "kbLyric": "rejected"})
    )


def edit_ratio(a: str, b: str) -> float:
    wa, wb = [w.lower() for w in words(a)], [w.lower() for w in words(b)]
    return 1 - difflib.SequenceMatcher(a=wa, b=wb, autojunk=False).ratio()


def score(pairs: pd.DataFrame) -> pd.DataFrame:
    og, kb = pairs["original"].tolist(), pairs["rejected"].tolist()
    matcher = ExplicitMatcher()
    tox = Toxicity()
    cut = pairs["cut"].to_numpy()
    pairs["tox_og"] = tox.score(og).round(4)
    pairs["lex_og"] = [list(matcher.find(t)) for t in og]
    pairs["syl_og"] = [syllables(t) for t in og]

    edited = ~cut
    sim = np.full(len(pairs), np.nan)
    tox_kb = np.full(len(pairs), np.nan)
    idx = np.flatnonzero(edited)
    sim[idx] = Similarity().pairwise([og[i] for i in idx], [kb[i] for i in idx])
    tox_kb[idx] = tox.score([kb[i] for i in idx])
    pairs["similarity"] = sim.round(4)
    pairs["tox_kb"] = tox_kb.round(4)
    pairs["lex_kb"] = [list(matcher.find(t)) if t else [] for t in kb]
    pairs["syl_kb"] = [syllables(t) if t else 0 for t in kb]
    pairs["rhyme_kept"] = [end_rhyme_kept(a, b) if b else False for a, b in zip(og, kb)]
    pairs["edit_ratio"] = [round(edit_ratio(a, b), 4) if b else 1.0 for a, b in zip(og, kb)]
    return pairs


def failure_tags(row) -> list[str]:
    tags = []
    radio_explicit = bool(row.lex_og) or row.tox_og >= 0.5
    if row.cut:
        tags.append("cut")
    if not radio_explicit:
        tags.append("over_censored")
    if row.cut:
        return tags
    if row.lex_kb or row.tox_kb >= TOXICITY_THRESHOLD:
        tags.append("still_explicit")
    if row.similarity < SIMILARITY_THRESHOLD:
        tags.append("meaning_drift")
    if abs(row.syl_og - row.syl_kb) >= 2:
        tags.append("syllable_break")
    if not row.rhyme_kept:
        tags.append("rhyme_break")
    return tags


# ---- Genius context ---------------------------------------------------------------------

def match_key(line: str) -> str:
    main, _ = split_adlibs(normalize(line))
    return re.sub(r"[^a-z0-9 ]", "", main.lower()).strip()


def fetch_genius_songs(kb_songs: KidzBopSongs) -> dict[tuple[str, str], list[dict]]:
    """All Genius rows whose title + artist match a Kidz Bop song (any genre tag)."""
    found: dict[tuple[str, str], list[dict]] = {}
    for shard in sorted(paths.GENIUS_DIR.rglob("*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(
            batch_size=50_000, columns=["title", "artist", "lyrics"]
        ):
            df = batch.to_pandas()
            for row in df.itertuples(index=False):
                for key in kb_songs.match(str(row.artist), str(row.title)):
                    found.setdefault(key, []).append(
                        {"artist": normalize(str(row.artist)), "title": normalize(str(row.title)),
                         "lyrics": row.lyrics}
                    )
    return found


def locate(line: str, sections) -> dict | None:
    """Best section/line position for a Kidz Bop original line (single or two joined lines)."""
    target = match_key(line)
    best = None
    for si, s in enumerate(sections):
        keys = [match_key(l) for l in s.lines]
        for li in range(len(keys)):
            for span in (1, 2):
                if li + span > len(keys):
                    continue
                cand = " ".join(keys[li : li + span])
                ratio = difflib.SequenceMatcher(a=target, b=cand, autojunk=False).ratio()
                if best is None or ratio > best[0]:
                    best = (ratio, si, li, span)
    if best is None or best[0] < LINE_MATCH_THRESHOLD:
        return None
    ratio, si, li, span = best
    s = sections[si]
    return {
        "match_ratio": round(ratio, 3),
        "section_index": si,
        "section_header": s.header,
        "section_type": s.type,
        "section_lines": s.lines,
        "line_index": li,
        "line_span": span,
        "context_before": sections[si - 1].lines[-8:] if si > 0 else [],
        "context_after": sections[si + 1].lines[:8] if si + 1 < len(sections) else [],
    }


def attach_context(pairs: pd.DataFrame) -> pd.DataFrame:
    genius = fetch_genius_songs(KidzBopSongs())
    contexts, sources = [], []
    for row in pairs.itertuples():
        best, source = None, None
        for cand in genius.get((row.artist, row.song), []):
            hit = locate(row.original, split_sections(cand["lyrics"]))
            if hit and (best is None or hit["match_ratio"] > best["match_ratio"]):
                best, source = hit, f"{cand['artist']} - {cand['title']}"
        contexts.append(best)
        sources.append(source)
    pairs["genius_song_found"] = [bool(genius.get((a, s))) for a, s in zip(pairs["artist"], pairs["song"])]
    pairs["genius_source"] = sources
    pairs["context"] = contexts
    return pairs


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
    parser = argparse.ArgumentParser(description="Compile Kidz Bop rewrites into DPO negatives.")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.6, 0.2, 0.2))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = score(load_pairs())
    pairs["failure_tags"] = [failure_tags(r) for r in pairs.itertuples()]
    pairs["negative"] = pairs["failure_tags"].str.len() > 0
    pairs["chosen"] = [r.original if "over_censored" in r.failure_tags else None
                       for r in pairs.itertuples()]
    pairs["needs_chosen"] = pairs["negative"] & pairs["chosen"].isna()
    pairs["split"] = split_by_song(pairs, tuple(args.ratios), args.seed)
    pairs = attach_context(pairs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in ("train.jsonl", "val.jsonl", "test.jsonl"):
        (OUT_DIR / stale).unlink(missing_ok=True)
    flat = pairs.drop(columns=["context"]).assign(
        context_found=pairs["context"].notna(),
        match_ratio=[c["match_ratio"] if c else None for c in pairs["context"]],
    )
    flat.to_csv(OUT_DIR / "pairs_scored.csv", index=False)

    keep = ["artist", "song", "split", "original", "rejected", "chosen", "needs_chosen",
            "failure_tags", "badwords", "categories", "similarity", "tox_og", "tox_kb",
            "syl_og", "syl_kb", "rhyme_kept", "edit_ratio", "genius_source", "context"]
    with open(OUT_DIR / "dpo_negatives.jsonl", "w", encoding="utf-8") as f:
        for rec in pairs[pairs["negative"]][keep].to_dict("records"):
            rec = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()}
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    neg = pairs[pairs["negative"]]
    print(f"rows: {len(pairs)}, negatives: {len(neg)}, acceptable (unused): {(~pairs['negative']).sum()}")
    print("failure tags:", pd.Series([t for ts in neg["failure_tags"] for t in ts]).value_counts().to_dict())
    print(f"complete pairs (chosen = original): {neg['chosen'].notna().sum()}, "
          f"need a synthetic chosen: {neg['needs_chosen'].sum()}")
    songs = pairs.drop_duplicates(["artist", "song"])
    print(f"Kidz Bop songs found in Genius: {songs['genius_song_found'].sum()} / {len(songs)}")
    print(f"negatives with section context: {neg['context'].notna().sum()} / {len(neg)}")
    print(neg.groupby("split").agg(n=("original", "size"), with_context=("context", lambda c: c.notna().sum()),
                                   needs_chosen=("needs_chosen", "sum")).to_string())
    print(f"outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
