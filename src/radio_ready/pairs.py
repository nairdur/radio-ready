"""Export synthetic pairs (explicit section -> radio edit) from generation files.

Inputs:  outputs/generations/<name>.jsonl (radio_ready.generate / radio_ready.rewrite), and
         the section records they were generated from (Genius sections, Kidz Bop sections)
Output:  data/processed/pairs/<name>.jsonl, one pair per section:
  id, source (genius / kidzbop), artist, title, section_header, split
  original           the explicit section lines
  rewrite            the radio edit lines (same number of lines)
  changed            indexes of the lines that differ
  model, prompt_version, style   how the rewrite was made
No training features are added: how pairs are used for SFT or DPO is up to the training code.

A pair is kept when the generation is valid, every chosen candidate passed the ParaDeHate
filters (radio_ready.generate), at least one line changed, and the detector finds no explicit
word in the rewrite. Genius sections reserved for commercial LLMs are never exported (they
are an evaluation set). Splits: Genius pairs by artist (60/20/20, the same hash as
radio_ready.labels), Kidz Bop pairs keep their song split.

Human edits: annotations/pair_edits.csv (committed, edited by hand, never overwritten). Every
run adds a row for each rewritten line not yet in it (kept and dropped sections alike):
  id, line (1-based), artist, title, status (kept / dropped: reason), original, rewrite
  edit     your line; replaces the rewrite (blank: keep the rewrite)
  verdict  like / superlike / dislike (written by radio_ready.vet, or typed)
  drop     y to leave the whole section out
  note     free text
Rows can also be added by hand for other lines of a section (id and line). Rules:
  - a dislike or drop on any line leaves the section out;
  - a section the filters dropped comes back when every one of its rows is liked, superliked
    or edited (human approval overrides the filters), and nothing explicit is left;
  - edits replace the rewrite in every case.
Pairs list edited line indexes in `human_edited` and line verdicts in `human_verdicts`.
Close the file in Excel before running.

Run:  python -m radio_ready.pairs GENERATION [GENERATION ...] [--name NAME]
      python -m radio_ready.pairs solid_pilot_solid-v1
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from radio_ready import paths
from radio_ready.commercial import reserved_ids
from radio_ready.explicit import ExplicitDetector
from radio_ready.labels import artist_split

PAIRS_DIR = paths.PROCESSED / "pairs"
SECTION_FILES = {"genius": paths.GENIUS_SECTIONS_JSONL,
                 "kidzbop": paths.KIDZBOP_DIR / "sections.jsonl"}
EDITS_CSV = paths.ANNOTATIONS / "pair_edits.csv"
EDIT_COLUMNS = ["id", "line", "artist", "title", "status", "original", "rewrite", "edit",
                "verdict", "drop", "note"]
DROP_VALUES = {"y", "yes", "x", "1", "drop"}
APPROVE = {"like", "superlike"}


def read_edits() -> list[dict]:
    if not EDITS_CSV.exists():
        return []
    with open(EDITS_CSV, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_edits(rows: list[dict]) -> None:
    EDITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = EDITS_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EDIT_COLUMNS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in EDIT_COLUMNS} for r in rows)
    tmp.replace(EDITS_CSV)


def load_sections() -> dict[str, tuple[str, dict]]:
    found = {}
    for source, path in SECTION_FILES.items():
        if path.exists():
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    rec = json.loads(line)
                    found[rec["id"]] = (source, rec)
    return found


def generation_path(name: str) -> Path:
    path = Path(name)
    return path if path.suffix == ".jsonl" and path.exists() else \
        paths.GENERATIONS_DIR / f"{Path(name).stem}.jsonl"


def write_side_by_side(path: Path, pairs: list[dict]) -> None:
    """Readable markdown: one table per pair, changed lines in bold, unchanged lines "="."""
    esc = lambda t: str(t).replace("|", r"\|")  # noqa: E731
    out = [f"# {path.stem}: original vs rewrite", "",
           f"{len(pairs)} pairs. Changed lines in bold; (edited) marks your edits.", ""]
    for p in pairs:
        out += [f"## {esc(p['artist'])} - {esc(p['title'])} "
                f"({esc(p['section_header'] or '')}, {p['split']}, `{p['id']}`)", "",
                "| # | original | rewrite |", "|---|---|---|"]
        for i, (o, r) in enumerate(zip(p["original"], p["rewrite"])):
            mark = " (edited)" if i in p["human_edited"] else ""
            out.append(f"| {i + 1} | **{esc(o)}** | **{esc(r)}**{mark} |" if o != r
                       else f"| {i + 1} | {esc(o)} | = |")
        out.append("")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def to_pair(gen: dict, source: str, rec: dict) -> dict:
    original, rewrite = rec["lines"], gen["lines"]
    return {
        "id": rec["id"], "source": source, "artist": rec["artist"], "title": rec["title"],
        "section_header": rec.get("section_header"),
        "split": rec.get("split") if source == "kidzbop" else artist_split(rec["artist"]),
        "original": original, "rewrite": rewrite,
        "changed": [i for i, (a, b) in enumerate(zip(original, rewrite)) if a != b],
        "model": gen["model"], "prompt_version": gen["prompt_version"],
        "style": gen.get("style"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export synthetic pairs from generation files.")
    parser.add_argument("generations", nargs="+", help="names in outputs/generations or paths")
    parser.add_argument("--name", default=None, help="output name (default: first input)")
    args = parser.parse_args()

    sections, reserved, detector = load_sections(), reserved_ids(), ExplicitDetector()
    edits = read_edits()
    by_line = {(r["id"], int(r["line"])): r for r in edits if r.get("line", "").isdigit()}
    drop_ids = {r["id"] for r in edits if r.get("drop", "").strip().lower() in DROP_VALUES
                or r.get("verdict", "").strip().lower() == "dislike"}
    pairs, dropped, seen, new_rows = [], Counter(), set(), []
    for name in args.generations:
        for line in open(generation_path(name), encoding="utf-8"):
            gen = json.loads(line)
            if gen["id"] in seen or gen["id"] in reserved or gen["id"] not in sections:
                dropped["duplicate, reserved or no section record"] += 1
                continue
            source, rec = sections[gen["id"]]
            lines = list(gen["lines"]) if gen.get("valid") and gen.get("lines") \
                else list(rec["lines"])
            reason = (None if gen.get("valid") and gen.get("passed") is not False
                      else "no usable answer" if not gen.get("valid") else "failed filters")
            # sheet rows for every rewritten line (and every target line without an answer)
            rows = sorted({i for i, (a, b) in enumerate(zip(rec["lines"], lines)) if a != b}
                          | set(gen.get("targets", [])))
            for i in rows:
                if (gen["id"], i + 1) not in by_line:
                    row = {"id": gen["id"], "line": str(i + 1), "artist": rec["artist"],
                           "title": rec["title"], "original": rec["lines"][i],
                           "rewrite": lines[i] if lines[i] != rec["lines"][i] else "",
                           "status": "kept" if reason is None else f"dropped: {reason}"}
                    by_line[(gen["id"], i + 1)] = row
                    new_rows.append(row)
            edited, verdicts, approved_all = [], {}, True
            for (sid, n), row in by_line.items():
                if sid != gen["id"] or not 0 < n <= len(lines):
                    continue
                verdict = row.get("verdict", "").strip().lower()
                if verdict:
                    verdicts[str(n - 1)] = verdict
                if row.get("edit", "").strip():
                    lines[n - 1] = row["edit"].strip()
                    edited.append(n - 1)
                elif verdict not in APPROVE:
                    approved_all = False
            if gen["id"] in drop_ids:
                dropped["dropped by human (dislike or drop)"] += 1
            elif reason and not approved_all:
                dropped[reason] += 1
            elif lines == rec["lines"]:
                dropped["nothing changed"] += 1
            elif detector.find("\n".join(lines)):
                dropped["explicit word left" + (" after edits" if edited else "")] += 1
            else:
                pair = to_pair(gen | {"lines": lines}, source, rec)
                pair["human_edited"] = sorted(edited)
                pair["human_verdicts"] = verdicts
                pairs.append(pair)
                seen.add(gen["id"])
    write_edits(edits + new_rows)

    out = PAIRS_DIR / f"{args.name or Path(args.generations[0]).stem}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    write_side_by_side(out.with_suffix(".md"), pairs)
    print(f"pairs: {len(pairs)} {dict(Counter(p['split'] for p in pairs))}, "
          f"with human edits: {sum(bool(p['human_edited']) for p in pairs)}")
    print(f"dropped: {dict(dropped)}")
    print(f"output: {out} (side by side: {out.with_suffix('.md').name})")
    print(f"edit sheet: {EDITS_CSV} ({len(new_rows)} new rows)")


if __name__ == "__main__":
    main()
