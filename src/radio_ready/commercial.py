"""Commercial-LLM rewrites (ChatGPT, Claude, Gemini, ...) by manual copy-paste, via one table.

data/commercial/commercial_table.csv has one row per section:
  id, source, artist, title, prompt_version   identify the section
  prompt                                      the complete prompt: copy this cell into the LLM
  output_<model>                              paste that LLM's answer here (one column per model)
The prompt asks for the whole section on one line, lines separated by " / ", so an answer
pastes into a single cell. A numbered multi-line answer ("4. ...") is accepted too.

Rows: every section of the Kidz Bop pilot (a comparison set the local model also rewrites)
plus --genius-sections Genius sections chosen by a hash of the section id. The Genius rows are
reserved for commercial models and skipped by radio_ready.generate.

Running this module (also the `commercial` pipeline step) only adds what is missing: new rows,
and an output column for each model in --models. Existing rows, prompts and pasted answers are
never changed, so it is safe to re-run while filling the table (close it in Excel first, which
locks the file).

load_rewrites() reads the filled cells for radio_ready.compare. An answer is valid when it has
the same number of lines as the section and every target line (explicit word) was changed;
changes outside the target lines and their neighbors are ignored (original kept).

Run:  python -m radio_ready.commercial [--genius-sections 50] [--models chatgpt claude gemini]
"""

import argparse
import csv
import hashlib
import json
import re

from radio_ready import paths
from radio_ready.explicit import ExplicitDetector
from radio_ready.prompts import (
    INSTRUCTIONS, PROMPT_VERSION, SYSTEM_PROMPT, merge, neighbor_lines, parse_rewrites,
    section_text, target_lines,
)

TABLE = paths.COMMERCIAL_DIR / "commercial_table.csv"
SEP = " / "
OUTPUT_PREFIX = "output_"
BASE_COLUMNS = ["id", "source", "artist", "title", "prompt_version", "prompt"]
DEFAULT_MODELS = ["chatgpt", "claude", "gemini"]
SOURCES = {
    "kidzbop_pilot": paths.KIDZBOP_DIR / "pilot.jsonl",
    "genius": paths.GENIUS_SECTIONS_JSONL,
}

TABLE_PROMPT = """\
{system}

Write the radio edit of this section of "{title}" by {artist}.

{instructions}

{section}

Answer with the full section on a single line: all {n} lines in order, separated by " / ", \
with every line you did not change copied exactly. Do not add line numbers, brackets, quotes \
or anything else."""


def prompt_for(record: dict, targets: dict[int, list[str]]) -> str:
    return TABLE_PROMPT.format(system=SYSTEM_PROMPT, title=record["title"],
                               artist=record["artist"], instructions=INSTRUCTIONS,
                               section=section_text(record, targets), n=len(record["lines"]))


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
        if record["id"] in have:
            continue
        targets = target_lines(record, detector)
        if not targets:
            continue
        rows.append({"id": record["id"], "source": source, "artist": record["artist"],
                     "title": record["title"], "prompt_version": PROMPT_VERSION,
                     "prompt": prompt_for(record, targets)})
        have.add(record["id"])
        added += 1

    outputs = [c for c in columns if c.startswith(OUTPUT_PREFIX)]
    outputs += [OUTPUT_PREFIX + m for m in models if OUTPUT_PREFIX + m not in outputs]
    extra = [c for c in columns if c not in BASE_COLUMNS and c not in outputs]
    fieldnames = BASE_COLUMNS + extra + outputs

    paths.COMMERCIAL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = TABLE.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in fieldnames} for r in rows)
    tmp.replace(TABLE)

    filled = {c: sum(bool(r.get(c, "").strip()) for r in rows) for c in outputs}
    print(f"table: {len(rows)} sections ({added} added), filled per model: {filled}")
    print(f"file: {TABLE}")


def parse_answer(answer: str, record: dict, targets) -> dict[int, str] | None:
    """{line index: rewrite} from a one-line " / " answer or a numbered multi-line answer."""
    text = answer.strip().strip("`").strip()
    if re.match(r"^\d+[.)]\s", text):  # numbered lines ("4. ..."), one or several
        numbered = parse_rewrites(text, record, targets)
        if numbered is not None:
            return numbered
    lines = [l.strip() for l in " ".join(text.splitlines()).split(SEP.strip())]
    if len(lines) != len(record["lines"]):
        return None
    allowed = set(targets) | set(neighbor_lines(record, targets))
    changed = {i: t for i, (t, o) in enumerate(zip(lines, record["lines"]))
               if t and t != o.strip() and i in allowed}
    return changed if set(targets) <= set(changed) else None


def load_rewrites(ids: set[str] | None = None) -> dict[str, dict[str, list[str]]]:
    """{model: {section id: rewritten lines}} for every valid pasted answer."""
    records = {r["id"]: r for p in SOURCES.values() for r in read_jsonl(p)}
    columns, rows = read_table()
    detector = ExplicitDetector()
    systems: dict[str, dict[str, list[str]]] = {}
    invalid: dict[str, list[str]] = {}
    for row in rows:
        record = records.get(row["id"])
        if record is None or (ids is not None and row["id"] not in ids):
            continue
        targets = target_lines(record, detector)
        for col in columns:
            if not col.startswith(OUTPUT_PREFIX) or not row.get(col, "").strip():
                continue
            model = col[len(OUTPUT_PREFIX):]
            rewrites = parse_answer(row[col], record, targets)
            if rewrites is None:
                invalid.setdefault(model, []).append(row["id"])
            else:
                systems.setdefault(model, {})[row["id"]] = merge(record, rewrites)
    for model, bad in invalid.items():
        print(f"commercial {model}: {len(bad)} invalid answers "
              f"(wrong line count or a target line unchanged): {', '.join(bad)}")
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
