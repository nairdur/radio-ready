"""Clean Genius lyrics, split songs into sections, and sample explicit sections.

Cleaning per song:
  - full-line headers like [Verse 1: Eminem] become section metadata (type + performers);
    credit headers ([Produced by ...]) and inline tags ([?]) are dropped
  - scrape junk (title line "... Lyrics", "123Embed", "You might also like") is removed
  - parenthetical ad-libs are moved out of the line into an `adlibs` field
  - repeated sections (same chorus twice) are kept once, with a repeat count

A section is eligible when its type is a sung/rapped part (verse, chorus, hook, ...), it has
--min-lines..--max-lines lines, and its main lines contain a strong explicit word.

Sampling mirrors the earlier branch: one random eligible song per artist (reservoir sampling,
seeded), then one random eligible section from that song. Songs in the Kidz Bop data are
excluded so human-written eval pairs never leak into synthetic data.

Run:  python -m radio_ready.genius [--tags rap trap pop rb] [--limit 50000]
Writes data/processed/genius_sections.jsonl
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd
import pyarrow.parquet as pq

from radio_ready import paths
from radio_ready.lexicon import ExplicitMatcher
from radio_ready.text import (
    HOMOGLYPHS, assonance_key, end_word, normalize, split_adlibs, stress, syllables,
)

HEADER = re.compile(r"^\[([^\]]*)\]$")
INLINE_TAG = re.compile(r"\[[^\]]*\]")
JUNK_LINE = re.compile(
    r"^(?:you might also like|\d*embed|see .+ live|get tickets as low as .*)$", re.IGNORECASE
)
EMBED_SUFFIX = re.compile(r"\d*Embed$")
GLUED_JUNK = re.compile(r"^You might also like", re.IGNORECASE)
CREDIT_HEADER = re.compile(r"^(?:produced|written|prod\.|lyrics|instrumental|tracklist)", re.I)

SECTION_TYPES = [
    ("pre-chorus", ("pre-chorus", "pre chorus", "prechorus")),
    ("post-chorus", ("post-chorus", "post chorus", "postchorus")),
    ("chorus", ("chorus",)),
    ("hook", ("hook",)),
    ("verse", ("verse",)),
    ("bridge", ("bridge",)),
    ("refrain", ("refrain",)),
    ("intro", ("intro",)),
    ("outro", ("outro",)),
    ("interlude", ("interlude", "skit")),
]
LYRIC_TYPES = {"verse", "chorus", "hook", "pre-chorus", "post-chorus", "bridge", "refrain"}
PERFORMER_SPLIT = re.compile(r"\s*(?:,|&|\+|\band\b|\bx\b|\bwith\b)\s*", re.IGNORECASE)


@dataclass
class Section:
    header: str
    type: str
    performers: list[str]
    lines: list[str] = field(default_factory=list)
    adlibs: list[str] = field(default_factory=list)
    repeats: int = 1


def parse_header(header: str) -> tuple[str, list[str]]:
    label, _, who = header.partition(":")
    low = label.lower()
    kind = next((name for name, keys in SECTION_TYPES if any(k in low for k in keys)), "other")
    performers = [p.strip() for p in PERFORMER_SPLIT.split(who) if p.strip()] if who else []
    return kind, performers


def split_sections(lyrics: str) -> list[Section]:
    raw = [normalize(line) for line in str(lyrics).splitlines()]
    if raw and raw[0].endswith("Lyrics") and not HEADER.match(raw[0]):
        raw = raw[1:]

    sections: list[Section] = []
    header, kind, performers = "", "unlabeled", []
    current: Section | None = None

    def close() -> None:
        nonlocal current
        if current is not None and current.lines:
            sections.append(current)
        current = None

    for line in raw:
        line = GLUED_JUNK.sub("", EMBED_SUFFIX.sub("", line)).strip()
        match = HEADER.match(line)
        if match:
            close()
            text = match.group(1).strip()
            if CREDIT_HEADER.match(text):
                header, kind, performers = text, "credit", []
            else:
                header = text
                kind, performers = parse_header(text)
            continue
        if not line:
            close()
            continue
        if JUNK_LINE.match(line) or kind == "credit":
            continue
        line, adlibs = split_adlibs(INLINE_TAG.sub(" ", line))
        if current is None:
            current = Section(header, kind, performers)
        current.adlibs.extend(adlibs)
        if re.search(r"[A-Za-z]", line) and syllables(line) > 0:  # drops "Mm"-style fillers
            current.lines.append(line)
    close()
    return dedupe(sections)


def section_key(section: Section) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", "", l.lower()) for l in section.lines)


def dedupe(sections: list[Section]) -> list[Section]:
    seen: dict[str, Section] = {}
    out = []
    for s in sections:
        key = section_key(s)
        if key in seen:
            seen[key].repeats += 1
            continue
        seen[key] = s
        out.append(s)
    return out


def rhyme_scheme(lines: list[str]) -> str:
    letters: dict[str, str] = {}
    scheme = []
    for line in lines:
        key = assonance_key(end_word(line)) if end_word(line) else ""
        if key not in letters:
            letters[key] = chr(ord("A") + len(letters) % 26)
        scheme.append(letters[key])
    return "".join(scheme)


def song_key(artist: str, title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", f"{artist}{title}".lower())


def kidz_bop_songs() -> set[str]:
    kb = pd.read_csv(paths.KIDZ_BOP_CSV, usecols=["ogArtist", "songName"])
    return {song_key(a, s) for a, s in zip(kb["ogArtist"], kb["songName"])}


def build_record(row: dict, sections: list[Section], index: int, explicit: tuple, eligible: int):
    s = sections[index]
    prev_lines = sections[index - 1].lines if index > 0 else []
    next_lines = sections[index + 1].lines if index + 1 < len(sections) else []
    uid = hashlib.md5(f"{row['artist']}|{row['title']}|{index}".encode()).hexdigest()[:12]
    return {
        "id": uid,
        "artist": row["artist"],
        "title": row["title"],
        "tag": row["tag"],
        "year": None if pd.isna(row["year"]) else int(row["year"]),
        "section_index": index,
        "n_sections": len(sections),
        "section_header": s.header,
        "section_type": s.type,
        "performers": s.performers,
        "repeats": s.repeats,
        "lines": s.lines,
        "syllables": [syllables(l) for l in s.lines],
        "stress": [stress(l) for l in s.lines],
        "rhyme_scheme": rhyme_scheme(s.lines),
        "adlibs": s.adlibs,
        "explicit_words": list(explicit),
        "eligible_sections_in_song": eligible,
        "context_before": prev_lines[-8:],
        "context_after": next_lines[:8],
    }


def iter_rows(tags: set[str], limit: int | None):
    seen = 0
    for shard in sorted(paths.GENIUS_DIR.rglob("*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(
            batch_size=20_000, columns=["title", "artist", "tag", "year", "lyrics"]
        ):
            df = batch.to_pandas()
            df = df[df["tag"].isin(tags)]
            for row in df.to_dict("records"):
                row["artist"] = normalize(str(row["artist"]))
                row["title"] = normalize(str(row["title"]))
                yield row
                seen += 1
                if limit and seen >= limit:
                    return


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample explicit Genius sections per artist.")
    parser.add_argument("--tags", nargs="+", default=["rap", "trap", "pop", "rb"])
    parser.add_argument("--min-lines", type=int, default=4)
    parser.add_argument("--max-lines", type=int, default=16)
    parser.add_argument("--min-artist-songs", type=int, default=3,
                        help="drop artists with fewer eligible songs (filters one-off uploads)")
    parser.add_argument("--limit", type=int, default=None, help="max songs to scan (testing)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    matcher = ExplicitMatcher(tiers=("strong",))
    excluded = kidz_bop_songs()
    rng = random.Random(args.seed)
    counts: Counter = Counter()
    chosen: dict[str, dict] = {}
    stats = Counter()

    for row in iter_rows(set(args.tags), args.limit):
        stats["scanned"] += 1
        row["lyrics"] = (row["lyrics"] or "").translate(HOMOGLYPHS)
        if not matcher.pattern.search(row["lyrics"]):
            continue
        stats["explicit_songs"] += 1
        if song_key(row["artist"], row["title"]) in excluded:
            stats["kidz_bop_excluded"] += 1
            continue
        sections = split_sections(row["lyrics"])
        eligible = []
        for i, s in enumerate(sections):
            if s.type in LYRIC_TYPES and args.min_lines <= len(s.lines) <= args.max_lines:
                hits = matcher.find("\n".join(s.lines))
                if hits:
                    eligible.append((i, hits))
        if not eligible:
            continue
        stats["eligible_songs"] += 1
        artist = row["artist"]
        counts[artist] += 1
        if rng.random() < 1 / counts[artist]:
            index, hits = rng.choice(eligible)
            chosen[artist] = build_record(row, sections, index, hits, len(eligible))

    records = [r for a, r in chosen.items() if counts[a] >= args.min_artist_songs]
    for r in records:
        r["artist_eligible_songs"] = counts[r["artist"]]
    records.sort(key=lambda r: (r["artist"].lower(), r["title"].lower()))

    paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    out = paths.PROCESSED / "genius_sections.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"artists with an eligible song: {len(chosen)}")
    print(f"artists kept (>= {args.min_artist_songs} eligible songs): {len(records)}")
    print(Counter(r["section_type"] for r in records).most_common())
    print(f"output: {out}")


if __name__ == "__main__":
    main()
