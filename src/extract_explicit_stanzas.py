import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("data/genius-expertise/lyrics.jl")
DEFAULT_WORDS_CSV = Path("reference/explicit_words.csv")
DEFAULT_OUTPUT = Path("output/genius_explicit_stanzas.csv")


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
        pattern = re.compile(
            rf"(?<!\w){escaped}(?!\w)",
            flags=re.IGNORECASE,
        )
        patterns.append((word, pattern))
    return patterns


def find_matches(
    line: str,
    patterns: Iterable[tuple[str, re.Pattern[str]]],
) -> list[str]:
    return [word for word, pattern in patterns if pattern.search(line)]


def extract_stanzas(
    input_jsonl: Path,
    words: list[str],
    output_csv: Path,
    context_lines: int,
) -> int:
    patterns = compile_word_patterns(words)
    matches_written = 0

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "song",
                "matched_line_number",
                "matched_words",
                "stanza_start",
                "stanza_end",
                "stanza",
            ],
        )
        writer.writeheader()

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

                song = str(record.get("song", ""))
                lyrics = str(record.get("lyrics", ""))
                lyrics_lines = lyrics.splitlines()

                for matched_index, lyric_line in enumerate(lyrics_lines):
                    matched_words = find_matches(lyric_line, patterns)
                    if not matched_words:
                        continue

                    start_index = max(0, matched_index - context_lines)
                    end_index = min(len(lyrics_lines), matched_index + context_lines + 1)
                    stanza_lines = lyrics_lines[start_index:end_index]

                    writer.writerow(
                        {
                            "song": song,
                            "matched_line_number": matched_index + 1,
                            "matched_words": "|".join(matched_words),
                            "stanza_start": start_index + 1,
                            "stanza_end": end_index,
                            "stanza": "\n".join(stanza_lines),
                        }
                    )
                    matches_written += 1

    return matches_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search JSONL lyrics by explicit-word CSV and export matching lines "
            "with surrounding context."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSONL file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--words-csv",
        type=Path,
        default=DEFAULT_WORDS_CSV,
        help=f"CSV containing explicit words (default: {DEFAULT_WORDS_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=4,
        help="Number of lyrics lines to include before and after each match",
    )
    args = parser.parse_args()

    if args.context < 0:
        parser.error("--context must be zero or greater")
    return args


def main() -> None:
    args = parse_args()
    input_path = (
        args.input if args.input.is_absolute()
        else REPOSITORY_ROOT / args.input
    )
    words_path = (
        args.words_csv if args.words_csv.is_absolute()
        else REPOSITORY_ROOT / args.words_csv
    )
    output_path = (
        args.output if args.output.is_absolute()
        else REPOSITORY_ROOT / args.output
    )
    words = load_explicit_words(words_path)
    count = extract_stanzas(input_path, words, output_path, args.context)
    print(
        f"Wrote {count} matching stanza row(s) from "
        f"{len(words)} word(s) to {output_path}"
    )


if __name__ == "__main__":
    main()
