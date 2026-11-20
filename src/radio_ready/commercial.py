"""Commercial-LLM rewrites (ChatGPT, Claude, Gemini, ...) by manual copy-paste.

data/commercial/prompt.txt   one generic prompt for every section: paste it at the start of a
                             chat (then send sections one by one) or in front of each section
data/commercial/commercial_table.csv, one row per section:
  id, source        identify the section
  section           "Title" by Artist, then the section lines: copy this cell into the LLM
  output_<model>    paste that LLM's answer here (one column per model)
The prompt asks for the whole section on one line, lines separated by " / ", so an answer
pastes into a single cell. A numbered multi-line answer ("1. ...") is accepted too.

Rows: every section of the Kidz Bop pilot (a comparison set the local model also rewrites)
plus --genius-sections Genius sections chosen by a hash of the section id, all with at least
one detected explicit word. The Genius rows are reserved for commercial models and skipped by
radio_ready.generate. The prompt never marks lines: the model decides what radio would censor.

Running this module (also the `commercial` pipeline step) only adds what is missing: new rows,
and an output column for each model in --models. Existing rows and pasted answers are never
changed, so it is safe to re-run while filling the table (close it in Excel first, which locks
the file). prompt.txt is rewritten every run.

load_rewrites() reads the filled cells for radio_ready.compare. An answer is valid when it has
the same number of lines as the section; compare scores what changed and what is left explicit.

Run:  python -m radio_ready.commercial [--genius-sections 50] [--models chatgpt claude gemini]
"""

import argparse
import csv
import hashlib
import json
import re

from radio_ready import paths
from radio_ready.explicit import ExplicitDetector
from radio_ready.prompts import clean_response_line, merge, target_lines

TABLE = paths.COMMERCIAL_DIR / "commercial_table.csv"
PROMPT_TXT = paths.COMMERCIAL_DIR / "prompt.txt"
SEP = " / "
OUTPUT_PREFIX = "output_"
BASE_COLUMNS = ["id", "source", "section"]
DEFAULT_MODELS = ["chatgpt", "claude", "gemini"]
SOURCES = {
    "kidzbop_pilot": paths.KIDZBOP_DIR / "pilot.jsonl",
    "genius": paths.GENIUS_SECTIONS_JSONL,
}
NUMBERED = re.compile(r"^\s*(\d+)[.)]\s?(.*)$")

PROMPT = f"""\
You are a professional lyricist who writes radio edits of songs. I will send you sections of \
songs, each starting with the title and artist, then one lyric line per row.

For each section, rewrite only the lines with words that would get censored on the radio. Leave \
every other line exactly as it is. In the lines you rewrite, change as few words as possible and \
keep the rhyme, the syllable count, the end rhyme, the lyricism and the flow, so it sounds like \
the same song. Do not clean up slang, dialect, grammar or spelling.

Answer with the full section on a single line: every lyric line in order (not the title line), \
separated by "{SEP}", with unchanged lines copied exactly. Do not add line numbers, quotes or \
anything else."""


def section_cell(record: dict) -> str:
    return f'"{record["title"]}" by {record["artist"]}\n' + "\n".join(record["lines"])


def hash_rank(section_id: str) -> int:
    return int(hashlib.sha256(section_id.encode()).hexdigest()[:12], 16)


def read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def read_table() -> tuple[list[str], list[dict]]:
    if not TABLE.exists():
        return [], []
    with open(TABLE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def reserved_ids() -> set[str]:
    """Genius sections reserved for commercial models (skipped by the local generator)."""
    return {r["id"] for r in read_table()[1] if r.get("source") == "genius"}


def build(genius_sections: int, models: list[str]) -> None:
    columns, rows = read_table()
    have = {r["id"] for r in rows}
    detector = ExplicitDetector()

    candidates = [("kidzbop_pilot", r) for r in read_jsonl(SOURCES["kidzbop_pilot"])]
    genius = sorted(read_jsonl(SOURCES["genius"]), key=lambda r: hash_rank(r["id"]))
    candidates += [("genius", r) for r in genius[:genius_sections]]

    added = 0
    for source, record in candidates:
        if record["id"] in have or not target_lines(record, detector):
            continue
        rows.append({"id": record["id"], "source": source, "section": section_cell(record)})
        have.add(record["id"])
        added += 1

    outputs = [c for c in columns if c.startswith(OUTPUT_PREFIX)]
    outputs += [OUTPUT_PREFIX + m for m in models if OUTPUT_PREFIX + m not in outputs]
    extra = [c for c in columns if c not in BASE_COLUMNS and c not in outputs]
    fieldnames = BASE_COLUMNS + extra + outputs

    paths.COMMERCIAL_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_TXT.write_text(PROMPT + "\n", encoding="utf-8")
    tmp = TABLE.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in fieldnames} for r in rows)
    tmp.replace(TABLE)

    filled = {c: sum(bool(r.get(c, "").strip()) for r in rows) for c in outputs}
    print(f"table: {len(rows)} sections ({added} added), filled per model: {filled}")
    print(f"files: {TABLE}\n       {PROMPT_TXT}")


def parse_answer(answer: str, record: dict) -> dict[int, str] | None:
    """{line index: rewrite} for the changed lines of a one-line " / " answer or a numbered
    multi-line answer; None unless it has exactly the section's number of lines."""
    text = answer.strip().strip("`").strip()
    rows = [r for r in text.splitlines() if r.strip()]
    if rows and all(NUMBERED.match(r) for r in rows):
        lines = [clean_response_line(NUMBERED.match(r).group(2)) for r in rows]
    else:
        lines = [l.strip() for l in " ".join(rows).split(SEP.strip())]
    if len(lines) == len(record["lines"]) + 1 and lines[0].strip('"').startswith(record["title"]):
        lines = lines[1:]  # title line echoed back
    if len(lines) != len(record["lines"]) or not all(lines):
        return None
    return {i: t for i, (t, o) in enumerate(zip(lines, record["lines"])) if t != o.strip()}


def load_rewrites(ids: set[str] | None = None) -> dict[str, dict[str, list[str]]]:
    """{model: {section id: rewritten lines}} for every valid pasted answer."""
    records = {r["id"]: r for p in SOURCES.values() for r in read_jsonl(p)}
    columns, rows = read_table()
    systems: dict[str, dict[str, list[str]]] = {}
    invalid: dict[str, list[str]] = {}
    for row in rows:
        record = records.get(row["id"])
        if record is None or (ids is not None and row["id"] not in ids):
            continue
        for col in columns:
            if not col.startswith(OUTPUT_PREFIX) or not row.get(col, "").strip():
                continue
            model = col[len(OUTPUT_PREFIX):]
            rewrites = parse_answer(row[col], record)
            if rewrites is None:
                invalid.setdefault(model, []).append(row["id"])
            else:
                systems.setdefault(model, {})[row["id"]] = merge(record, rewrites)
    for model, bad in invalid.items():
        print(f"commercial {model}: {len(bad)} invalid answers (wrong line count): "
              f"{', '.join(bad)}")
    return systems


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or extend the commercial-LLM table.")
    parser.add_argument("--genius-sections", type=int, default=50)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="one output column per model; existing columns are kept")
    args = parser.parse_args()
    build(args.genius_sections, args.models)


if __name__ == "__main__":
    main()
