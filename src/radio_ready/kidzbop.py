"""Kidz Bop song sections: side-by-side and evaluation dataset, plus Kidz Bop word statistics.

Separate from the Genius sample (genius.py excludes these songs). For each Kidz Bop song found
in Genius, every eligible explicit section (same rules as genius.py) is kept. Sections with a
line Kidz Bop edited also carry the Kidz Bop version next to the original. Uses: 7B rewrites
for side-by-side human ranking, evaluation of the trained model, and a seeded pilot subset for
testing 7B generation.

Inputs:  data/raw/kidz-bop/KB_censored-lyrics.csv, data/raw/genius,
         explicit-word list (radio_ready.explicit)
Outputs: data/artifacts/kidzbop_words.csv          per censored word: category, rows,
                                                     occurrences, songs, artists, cut rate,
                                                     top replacements
         data/artifacts/kidzbop_substitutions.csv  word-level diff: bad word -> replacement
         data/processed/kidzbop/sections.jsonl      side-by-side sections
         data/processed/kidzbop/pilot.jsonl         pilot subset (--pilot-size)

Section record: the genius.py fields (lines, syllables, stress, rhyme_scheme, adlibs,
explicit_words, context) plus
  has_kidz_bop_edit  whether Kidz Bop edited a line in this section
  kidz_bop_lines  the section with Kidz Bop lines substituted at their positions; a cut line
                  is ""; when one Kidz Bop line covers two original lines, the second is "";
                  None when the section has no edit
  kidz_bop_edits  [{line_index, line_span, original, kidz_bop, cut, badwords, match_ratio}]
  split           train/val/test by song
  pilot           True for the pilot subset (sections with a Kidz Bop edit)

Run:  python -m radio_ready.kidzbop
"""

import argparse
import difflib
import json
import random
import re
from collections import Counter, defaultdict

import pandas as pd
import pyarrow.parquet as pq

from radio_ready import paths
from radio_ready.explicit import ExplicitDetector
from radio_ready.genius import LYRIC_TYPES, KidzBopSongs, build_record, split_sections
from radio_ready.text import HOMOGLYPHS, normalize, split_adlibs, words

CUT_MARKER = "cuts verse"


def load_rows() -> pd.DataFrame:
    kb = pd.read_csv(paths.KIDZ_BOP_CSV)
    for col in ("ogLyric", "kbLyric", "ogArtist", "songName"):
        kb[col] = kb[col].astype(str).map(normalize)
    kb["badword"] = kb["badword"].str.lower().str.strip()
    kb["cut"] = kb["kbLyric"].str.lower() == CUT_MARKER
    kb.loc[kb["cut"], "kbLyric"] = ""
    return kb


# ---- word statistics ------------------------------------------------------------------------

def diff_replacements(og: str, kb: str, badword: str) -> list[str]:
    """What replaced the token(s) containing the bad word, by word-level diff."""
    a = [w.lower() for w in words(og)]
    b = [w.lower() for w in words(kb)]
    found = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ("replace", "delete") and any(badword in w for w in a[i1:i2]):
            found.append(" ".join(b[j1:j2]) or "<deleted>")
    return found


def word_stats(kb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subs: Counter = Counter()
    per_word: dict[str, Counter] = defaultdict(Counter)
    edited = kb[~kb["cut"]].drop_duplicates(["songName", "ogLyric", "kbLyric", "badword"])
    for row in edited.itertuples():
        for rep in diff_replacements(row.ogLyric, row.kbLyric, row.badword):
            subs[(row.badword, rep)] += 1
            per_word[row.badword][rep] += 1

    stats = (
        kb.groupby("badword")
        .agg(category=("category", "first"), kb_rows=("badword", "size"),
             kb_occurrences=("count", "sum"), kb_songs=("songName", "nunique"),
             kb_artists=("ogArtist", "nunique"), kb_cut_rate=("cut", "mean"))
        .rename_axis("word").reset_index()
        .sort_values(["kb_rows", "word"], ascending=[False, True])
    )
    stats["kb_cut_rate"] = stats["kb_cut_rate"].round(3)
    stats["top_replacements"] = stats["word"].map(
        lambda w: "; ".join(f"{r} ({n})" for r, n in per_word[w].most_common(5)))
    sub_df = pd.DataFrame([(w, r, n) for (w, r), n in subs.items()],
                          columns=["badword", "replacement", "count"])
    return stats, sub_df.sort_values(["count", "badword"], ascending=[False, True])


# ---- line pairs -----------------------------------------------------------------------------

def load_pairs(kb: pd.DataFrame) -> pd.DataFrame:
    """One row per (song, original line); the first Kidz Bop version is kept."""
    kb = kb[kb["ogLyric"].str.lower() != kb["kbLyric"].str.lower()]
    pairs = (
        kb.groupby(["ogArtist", "songName", "ogLyric"], sort=False)
        .agg(kidz_bop=("kbLyric", "first"), cut=("cut", "first"),
             badwords=("badword", lambda s: sorted(set(s))))
        .reset_index()
        .rename(columns={"ogArtist": "artist", "songName": "song", "ogLyric": "original"})
    )
    return pairs


# ---- locating lines in Genius ---------------------------------------------------------------

def match_key(line: str) -> str:
    main, _ = split_adlibs(normalize(line))
    return re.sub(r"[^a-z0-9 ]", "", main.lower()).strip()


def fetch_genius_songs(kb_songs: KidzBopSongs) -> dict[tuple[str, str], list[dict]]:
    """Genius rows whose title + artist match a Kidz Bop song (all genre tags)."""
    found: dict[tuple[str, str], list[dict]] = {}
    for shard in sorted(paths.GENIUS_DIR.rglob("*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(
            batch_size=50_000, columns=["title", "artist", "tag", "year", "lyrics"]
        ):
            for row in batch.to_pandas().to_dict("records"):
                for key in kb_songs.match(str(row["artist"]), str(row["title"])):
                    row["artist"] = normalize(str(row["artist"]))
                    row["title"] = normalize(str(row["title"]))
                    row["lyrics"] = (row["lyrics"] or "").translate(HOMOGLYPHS)
                    found.setdefault(key, []).append(row)
    return found


def locate(line: str, sections, min_ratio: float) -> tuple[float, int, int, int] | None:
    """(ratio, section_index, line_index, span) of the best match, one or two joined lines."""
    target = match_key(line)
    best = None
    for si, s in enumerate(sections):
        keys = [match_key(l) for l in s.lines]
        for li in range(len(keys)):
            for span in (1, 2):
                if li + span > len(keys):
                    continue
                ratio = difflib.SequenceMatcher(
                    a=target, b=" ".join(keys[li : li + span]), autojunk=False).ratio()
                if best is None or ratio > best[0]:
                    best = (ratio, si, li, span)
    return best if best and best[0] >= min_ratio else None


def song_sections(pairs: pd.DataFrame, candidates: list[dict], min_ratio: float):
    """Pick the Genius version where the most Kidz Bop lines are found; return its matches."""
    best = None
    for cand in candidates:
        sections = split_sections(cand["lyrics"])
        hits = []
        for row in pairs.itertuples():
            found = locate(row.original, sections, min_ratio)
            if found:
                hits.append((row, found))
        if best is None or len(hits) > len(best[2]):
            best = (cand, sections, hits)
    return best


def build_section(cand, sections, index, hits, explicit) -> dict:
    record = build_record(cand, sections, index, explicit, eligible=0)
    record.pop("eligible_sections_in_song")
    lines = list(record["lines"])
    edits = []
    for row, (ratio, _, li, span) in sorted(hits, key=lambda h: h[1][2]):
        edits.append({"line_index": li, "line_span": span, "original": row.original,
                      "kidz_bop": row.kidz_bop, "cut": bool(row.cut),
                      "badwords": row.badwords, "match_ratio": round(ratio, 3)})
        lines[li] = row.kidz_bop
        if span == 2:
            lines[li + 1] = ""
    record["has_kidz_bop_edit"] = bool(edits)
    record["kidz_bop_lines"] = lines if edits else None
    record["kidz_bop_edits"] = edits
    return record


def split_songs(keys: list[tuple[str, str]], ratios, seed: int) -> dict:
    keys = sorted(set(keys))
    random.Random(seed).shuffle(keys)
    n_train, n_val = round(len(keys) * ratios[0]), round(len(keys) * ratios[1])
    return {k: ("train" if i < n_train else "val" if i < n_train + n_val else "test")
            for i, k in enumerate(keys)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Kidz Bop side-by-side sections and word stats.")
    parser.add_argument("--detector", choices=["learned", "surge"], default="learned")
    parser.add_argument("--min-lines", type=int, default=4)
    parser.add_argument("--max-lines", type=int, default=16)
    parser.add_argument("--min-line-match", type=float, default=0.75)
    parser.add_argument("--pilot-size", type=int, default=50)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.6, 0.2, 0.2))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    kb = load_rows()
    stats, subs = word_stats(kb)
    paths.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stats.to_csv(paths.KIDZBOP_WORDS_CSV, index=False)
    subs.to_csv(paths.KIDZBOP_SUBSTITUTIONS_CSV, index=False)

    detector = ExplicitDetector(args.detector)
    pairs = load_pairs(kb)
    genius = fetch_genius_songs(KidzBopSongs())
    counter = Counter()
    records = []
    for (artist, song), song_pairs in pairs.groupby(["artist", "song"], sort=True):
        counter["songs"] += 1
        if (artist, song) not in genius:
            continue
        counter["songs_in_genius"] += 1
        cand, sections, hits = song_sections(song_pairs, genius[(artist, song)], args.min_line_match)
        counter["lines"] += len(song_pairs)
        counter["lines_located"] += len(hits)
        by_section = defaultdict(list)
        for hit in hits:
            by_section[hit[1][1]].append(hit)
        counter["sections_with_edits"] += len(by_section)
        for index, s in enumerate(sections):
            if s.type not in LYRIC_TYPES or not args.min_lines <= len(s.lines) <= args.max_lines:
                continue
            explicit = detector.find("\n".join(s.lines))
            if not explicit:
                continue
            record = build_section(cand, sections, index, by_section.get(index, []), explicit)
            record.update({"kidz_bop_artist": artist, "kidz_bop_song": song})
            records.append(record)

    splits = split_songs([(r["kidz_bop_artist"], r["kidz_bop_song"]) for r in records],
                         args.ratios, args.seed)
    edited = [i for i, r in enumerate(records) if r["has_kidz_bop_edit"]]
    pilot = set(random.Random(args.seed).sample(edited, min(args.pilot_size, len(edited))))
    for i, r in enumerate(records):
        r["split"] = splits[(r["kidz_bop_artist"], r["kidz_bop_song"])]
        r["pilot"] = i in pilot

    out = paths.KIDZBOP_DIR
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("pairs_scored.csv", "dpo_negatives.jsonl"):
        (out / stale).unlink(missing_ok=True)
    with open(out / "sections.jsonl", "w", encoding="utf-8") as f, \
            open(out / "pilot.jsonl", "w", encoding="utf-8") as p:
        for r in records:
            line = json.dumps(r, ensure_ascii=False) + "\n"
            f.write(line)
            if r["pilot"]:
                p.write(line)

    print(f"words: {len(stats)}, substitutions: {len(subs)}")
    for k, v in counter.items():
        print(f"{k}: {v}")
    print(f"explicit sections kept: {len(records)} from "
          f"{len({(r['kidz_bop_artist'], r['kidz_bop_song']) for r in records})} songs")
    print(f"sections with a Kidz Bop edit: {len(edited)}, "
          f"edits in them: {sum(len(r['kidz_bop_edits']) for r in records)}")
    print(f"splits: {dict(Counter(r['split'] for r in records))}, pilot: {len(pilot)}")
    print(f"outputs: {paths.KIDZBOP_WORDS_CSV}, {paths.KIDZBOP_SUBSTITUTIONS_CSV}, {out}")


if __name__ == "__main__":
    main()
