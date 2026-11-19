import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("data/genius-expertise/lyrics.jl")
DEFAULT_WORDS_CSV = Path("reference/explicit_words.csv")
DEFAULT_OUTPUT = Path("output/genius_explicit_stanzas_by_artist.json")
DEFAULT_STANZA_CSV = Path("output/genius_selected_stanzas.csv")

BRACKET_SECTION = re.compile(r"\[([^\]]+)\]")
SONG_SUFFIX = re.compile(r"(?:-lyrics)?$", re.IGNORECASE)


def load_explicit_words(words_csv: Path) -> list[str]:
    words: list[str] = []
    known_headers = {
        "word",
        "words",
        "explicit_word",
        "explicit_words",
        "term",
        "terms",
    }

    with words_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        for row_number, row in enumerate(reader):
            if not row:
                continue

            value = row[0].strip()
            if not value:
                continue
            if row_number == 0 and value.lower().replace(" ", "_") in known_headers:
                continue
            if value not in words:
                words.append(value)

    if not words:
        raise ValueError(f"No explicit words found in {words_csv}")
    return words


def compile_word_patterns(words: Iterable[str]) -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for word in words:
        escaped = re.escape(word)
        pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
        patterns.append((word, pattern))
    return patterns


def find_matches(
    text: str,
    patterns: Iterable[tuple[str, re.Pattern[str]]],
) -> list[str]:
    return [word for word, pattern in patterns if pattern.search(text)]


def slug_title(text: str) -> str:
    value = text.strip()
    value = SONG_SUFFIX.sub("", value).strip("-")
    value = re.sub(r"[-]+", " ", value)
    return value.title()


def longest_common_prefix(left: str, right: str) -> list[str]:
    left_parts = left.removesuffix("-lyrics").split("-")
    right_parts = right.removesuffix("-lyrics").split("-")
    common: list[str] = []

    for left_part, right_part in zip(left_parts, right_parts):
        if left_part.lower() != right_part.lower():
            break
        common.append(left_part)
    return common


def infer_artist(
    slug: str,
    previous_slug: str | None,
    next_slug: str | None,
) -> str:
    candidates: list[list[str]] = []
    if previous_slug:
        candidates.append(longest_common_prefix(slug, previous_slug))
    if next_slug:
        candidates.append(longest_common_prefix(slug, next_slug))

    candidate = max(
        (parts for parts in candidates if len(parts) >= 2),
        key=len,
        default=[],
    )

    if candidate:
        return " ".join(candidate).title()

    parts = slug.removesuffix("-lyrics").split("-")
    fallback_length = min(2, len(parts))
    return " ".join(parts[:fallback_length]).title()


def infer_song_title(slug: str, artist: str) -> str:
    value = slug.removesuffix("-lyrics")
    artist_parts = artist.lower().replace(" ", "-").split("-")
    parts = value.split("-")

    if len(parts) >= len(artist_parts):
        prefix = parts[: len(artist_parts)]
        if [part.lower() for part in prefix] == artist_parts:
            value = "-".join(parts[len(artist_parts) :])

    if not value:
        value = slug.removesuffix("-lyrics")
    return slug_title(value)


def extract_explicit_stanzas(
    lyrics: str,
    patterns: Iterable[tuple[str, re.Pattern[str]]],
    stanza_size: int,
) -> list[dict[str, object]]:
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    stanzas: list[dict[str, object]] = []
    section = "Unlabeled"
    section_lines: list[str] = []

    def append_chunks(chunk_lines: list[str], section_name: str) -> None:
        for offset in range(0, len(chunk_lines), stanza_size):
            chunk = chunk_lines[offset : offset + stanza_size]
            text = "\n".join(chunk)
            if BRACKET_SECTION.search(text):
                continue
            matched_words = find_matches(text, patterns)
            if not matched_words:
                continue

            stanzas.append(
                {
                    "section": section_name,
                    "chunk_start": offset + 1,
                    "chunk_end": offset + len(chunk),
                    "matched_words": matched_words,
                    "text": text,
                }
            )

    for line in lines:
        full_line_section = BRACKET_SECTION.fullmatch(line)
        if full_line_section:
            if section_lines:
                append_chunks(section_lines, section)
                section_lines = []
            section = full_line_section.group(1).strip()
            continue
        section_lines.append(line)

    if section_lines:
        append_chunks(section_lines, section)

    return stanzas


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPOSITORY_ROOT / path


def build_artist_sample(
    input_jsonl: Path,
    words: list[str],
    stanza_size: int,
    seed: int,
) -> dict[str, object]:
    patterns = compile_word_patterns(words)
    records: list[dict[str, str]] = []

    with input_jsonl.open("r", encoding="utf-8-sig") as input_file:
        for line_number, json_line in enumerate(input_file, start=1):
            if not json_line.strip():
                continue
            try:
                record = json.loads(json_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {input_jsonl}: {error}"
                ) from error

            slug = str(record.get("song", "")).strip()
            lyrics = str(record.get("lyrics", ""))
            if slug and lyrics:
                records.append({"slug": slug, "lyrics": lyrics})

    artists: dict[str, list[dict[str, str]]] = {}
    for index, record in enumerate(records):
        previous_slug = records[index - 1]["slug"] if index > 0 else None
        next_slug = records[index + 1]["slug"] if index + 1 < len(records) else None
        artist = infer_artist(record["slug"], previous_slug, next_slug)
        artists.setdefault(artist, []).append(record)

    rng = random.Random(seed)
    result: dict[str, object] = {
        "seed": seed,
        "stanza_size": stanza_size,
        "artists": {},
    }
    artist_result: dict[str, object] = {}

    for artist in sorted(artists):
        artist_songs = artists[artist]
        chosen = rng.choice(artist_songs)
        stanzas = extract_explicit_stanzas(
            chosen["lyrics"],
            patterns,
            stanza_size,
        )
        selected_stanza = rng.choice(stanzas) if stanzas else None
        if selected_stanza is None:
            continue

        artist_result[artist] = {
            "song_slug": chosen["slug"],
            "song_title": infer_song_title(chosen["slug"], artist),
            "full_song_title": slug_title(chosen["slug"]),
            "selected_stanza": selected_stanza,
        }

    result["artists"] = artist_result
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Group Genius lyrics by artist, select one random song per artist, "
            "and export explicit-word stanza chunks as JSON."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--words-csv", type=Path, default=DEFAULT_WORDS_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stanza-csv", type=Path, default=DEFAULT_STANZA_CSV)
    parser.add_argument("--stanza-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.stanza_size < 1:
        parser.error("--stanza-size must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    words_path = resolve_path(args.words_csv)
    output_path = resolve_path(args.output)
    stanza_csv_path = resolve_path(args.stanza_csv)

    words = load_explicit_words(words_path)
    result = build_artist_sample(
        input_path,
        words,
        args.stanza_size,
        args.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    artists = result["artists"]
    artist_count = len(artists)
    stanza_count = 0

    stanza_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with stanza_csv_path.open("w", encoding="utf-8", newline="") as stanza_file:
        writer = csv.writer(stanza_file)
        writer.writerow(["artist|full song title", "stanza"])

        for artist, artist_data in artists.items():
            selected_stanza = artist_data["selected_stanza"]
            if selected_stanza is None:
                continue
            writer.writerow(
                [
                    f"[{artist}|{artist_data['full_song_title']}]",
                    selected_stanza["text"],
                ]
            )
            stanza_count += 1

    print(
        f"Wrote one random song for {artist_count} artist(s) and "
        f"{stanza_count} selected stanza(s) from {len(words)} explicit word(s)"
    )
    print(f"JSON: {output_path}")
    print(f"CSV:  {stanza_csv_path}")


if __name__ == "__main__":
    main()
