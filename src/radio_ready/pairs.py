"""Export synthetic pairs (explicit section -> radio edit) from generation files.

Inputs:  outputs/generations/<name>.jsonl (radio_ready.generate / radio_ready.rewrite), and
         the section records they were generated from (Genius sections, Kidz Bop sections)
Outputs, in pairs/:
  <name>.patch.jsonl  committed, no lyrics: per pair the ids, split, how it was made and a
                      word-level diff of each changed line (radio_ready.linediff)
  <name>.jsonl        full pairs, gitignored (lyrics); rebuilt anywhere with --rebuild from the
  <name>.md           patches and the section records (download + kidzbop / genius steps)
A full pair:
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

Human edits: annotations/edits/<name>.csv, one sheet per export (committed, never
overwritten; written by radio_ready.vet <name>). Every run adds a row for each rewritten line not yet in it (kept and dropped
sections alike). No lyrics are stored: the model rewrite and your edit are word-level diffs
against the original line, which is read from the section records when the sheet is loaded.
  id, line (1-based), artist, title, status (kept / dropped: reason)
  rewrite_diff  the model's rewrite of the line
  edit_diff     your line; replaces the rewrite (blank: keep the rewrite)
  verdict  like / superlike / dislike (written by radio_ready.vet, or typed)
  drop     y to leave the whole section out
  note     free text
Rows can also be added by hand for other lines of a section (id and line). Rules:
  - a dislike or drop on any line leaves the section out;
  - a section the filters dropped comes back when every one of its rows is liked, superliked
    or edited (human approval overrides the filters), and nothing explicit is left;
  - edits replace the rewrite in every case.
Pairs list edited line indexes in `human_edited` and line verdicts in `human_verdicts`.
Sheets in the older layout (full original / rewrite / edit text columns) are read and
converted on the next write.

Chosen versions: `--choices` exports the version picked per section in
annotations/noise_choices.csv (radio_ready.vet --choose; a pick counts as human approval, so
the filters do not drop it). Output name: noise_chosen.

Run:  python -m radio_ready.pairs GENERATION [GENERATION ...] [--name NAME]
      python -m radio_ready.pairs solid_pilot_solid-v1
      python -m radio_ready.pairs --choices
      python -m radio_ready.pairs --rebuild         # full pairs from every pairs/*.patch.jsonl
      python -m radio_ready.pairs commercial_claude --include-reserved --require-approval
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
from radio_ready.linediff import apply, diff

PAIRS_DIR = paths.ROOT / "pairs"  # committed: the deliverable
SECTION_FILES = {"genius": paths.GENIUS_SECTIONS_JSONL,
                 "kidzbop": paths.KIDZBOP_DIR / "sections.jsonl"}
EDITS_DIR = paths.ANNOTATIONS / "edits"  # one sheet per export: edits/<name>.csv
EDIT_COLUMNS = ["id", "line", "artist", "title", "status", "rewrite_diff", "edit_diff",
                "verdict", "drop", "note"]
DROP_VALUES = {"y", "yes", "x", "1", "drop"}
APPROVE = {"like", "superlike"}


def sheet_path(name: str) -> Path:
    return EDITS_DIR / f"{name}.csv"


def read_edits(sheet: Path) -> list[dict]:
    """Sheet rows as stored; call fill_text() to add original / rewrite / edit text."""
    if not sheet.exists():
        return []
    with open(sheet, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fill_text(rows: list[dict], lines_of) -> None:
    """Add original, rewrite and edit text to rows; lines_of(section id) -> lines or None."""
    for r in rows:
        if r.get("original"):
            continue  # older layout with full text
        lines = lines_of(r["id"])
        n = int(r["line"]) - 1 if str(r.get("line", "")).isdigit() else -1
        if not lines or not 0 <= n < len(lines):
            r["original"], r["rewrite"], r["edit"] = "", "", ""
            continue
        r["original"] = lines[n]
        for col, out in (("rewrite_diff", "rewrite"), ("edit_diff", "edit")):
            r[out] = apply(lines[n], json.loads(r[col])) if r.get(col) else ""


def write_edits(rows: list[dict], sheet: Path) -> None:
    """Store rows with rewrite and edit as diffs against the original line (no lyrics)."""
    out = []
    for r in rows:
        row = dict(r)
        if r.get("original"):
            for text, col in (("rewrite", "rewrite_diff"), ("edit", "edit_diff")):
                value = (r.get(text) or "").strip()
                if col == "rewrite_diff" and value == r["original"]:
                    value = ""  # an unchanged rewrite is no rewrite
                # an edit equal to the original is kept: it means "keep the original line"
                row[col] = json.dumps(diff(r["original"], value), ensure_ascii=False) \
                    if value else ""
        out.append(row)
    sheet.parent.mkdir(parents=True, exist_ok=True)
    tmp = sheet.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EDIT_COLUMNS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in EDIT_COLUMNS} for r in out)
    tmp.replace(sheet)


CHOICES_CSV = paths.ANNOTATIONS / "noise_choices.csv"


def read_choices() -> dict[str, str]:
    """{section id: chosen generation name or "none"} (radio_ready.vet --choose)."""
    if not CHOICES_CSV.exists():
        return {}
    with open(CHOICES_CSV, encoding="utf-8-sig", newline="") as f:
        return {r["id"]: r["choice"] for r in csv.DictReader(f) if r.get("choice")}


def write_choices(choices: dict[str, str]) -> None:
    CHOICES_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHOICES_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["id", "choice"])
        w.writerows(sorted(choices.items()))
    tmp.replace(CHOICES_CSV)


def generation_records(names: list[str], choices: bool):
    """(generation record, chosen by a human) from the files, or from the human choices."""
    if choices:
        wanted: dict[str, set[str]] = {}
        for sid, name in read_choices().items():
            if name != "none":
                wanted.setdefault(name, set()).add(sid)
        for name, ids in sorted(wanted.items()):
            for line in open(generation_path(name), encoding="utf-8"):
                gen = json.loads(line)
                if gen["id"] in ids:
                    yield gen, True
        return
    for name in names:
        for line in open(generation_path(name), encoding="utf-8"):
            yield json.loads(line), False


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
    if name.endswith(".jsonl"):
        return path if path.exists() else paths.GENERATIONS_DIR / path.name
    return paths.GENERATIONS_DIR / f"{name}.jsonl"  # names contain dots ("qwen2.5-...")


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


PATCH_KEYS = ["id", "source", "artist", "title", "section_header", "split", "model",
              "prompt_version", "style", "human_edited", "human_verdicts"]


def to_patch(pair: dict) -> dict:
    """The pair without lyrics: metadata and a word-level diff of each changed line."""
    return {**{k: pair.get(k) for k in PATCH_KEYS},
            "diff": {str(i): diff(pair["original"][i], pair["rewrite"][i])
                     for i in pair["changed"]}}


PAIR_KEYS = ["id", "source", "artist", "title", "section_header", "split", "original",
             "rewrite", "changed", "model", "prompt_version", "style", "human_edited",
             "human_verdicts"]


def from_patch(patch: dict, rec: dict) -> dict:
    """The full pair, field for field as exported (same key order)."""
    rewrite = list(rec["lines"])
    for i, ops in patch["diff"].items():
        rewrite[int(i)] = apply(rec["lines"][int(i)], ops)
    full = {**patch, "original": rec["lines"], "rewrite": rewrite,
            "changed": sorted(int(i) for i in patch["diff"])}
    return {k: full.get(k) for k in PAIR_KEYS}


def write_jsonl(path: Path, records) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_pairs(name: str, pairs: list[dict]) -> Path:
    """Full pairs and side-by-side (gitignored) plus the committed patch file."""
    out = PAIRS_DIR / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, pairs)
    write_side_by_side(PAIRS_DIR / f"{name}.md", pairs)
    write_jsonl(PAIRS_DIR / f"{name}.patch.jsonl", (to_patch(p) for p in pairs))
    return out


def rebuild() -> None:
    sections = load_sections()
    for patch_file in sorted(PAIRS_DIR.glob("*.patch.jsonl")):
        name = patch_file.name.removesuffix(".patch.jsonl")
        patches = [json.loads(l) for l in open(patch_file, encoding="utf-8") if l.strip()]
        missing = [p["id"] for p in patches if p["id"] not in sections]
        pairs = [from_patch(p, sections[p["id"]][1]) for p in patches if p["id"] in sections]
        write_jsonl(PAIRS_DIR / f"{name}.jsonl", pairs)
        write_side_by_side(PAIRS_DIR / f"{name}.md", pairs)
        note = (f", {len(missing)} sections not found (run the download, kidzbop and genius "
                f"steps)") if missing else ""
        print(f"{name}: {len(pairs)} pairs rebuilt{note}")


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
    parser.add_argument("generations", nargs="*", help="names in outputs/generations or paths")
    parser.add_argument("--name", default=None, help="output name (default: first input)")
    parser.add_argument("--choices", action="store_true",
                        help="export the versions chosen in annotations/noise_choices.csv")
    parser.add_argument("--rebuild", action="store_true",
                        help="full pairs from pairs/*.patch.jsonl and the section records")
    parser.add_argument("--include-reserved", action="store_true",
                        help="keep the Genius sections reserved for commercial evaluation")
    parser.add_argument("--require-approval", action="store_true",
                        help="a section becomes a pair only when all its rows are liked, "
                             "superliked or edited (e.g. commercial answers)")
    args = parser.parse_args()
    if args.rebuild:
        rebuild()
        return
    if not args.generations and not args.choices:
        parser.error("give generation names, --choices or --rebuild")
    if args.choices and not args.name:
        args.name = "noise_chosen"

    name = args.name or Path(args.generations[0]).name.removesuffix(".jsonl")
    sheet = sheet_path(name)
    sections, detector = load_sections(), ExplicitDetector()
    reserved = set() if args.include_reserved else reserved_ids()
    edits = read_edits(sheet)
    fill_text(edits, lambda sid: sections[sid][1]["lines"] if sid in sections else None)
    by_line = {(r["id"], int(r["line"])): r for r in edits if r.get("line", "").isdigit()}
    drop_ids = {r["id"] for r in edits if r.get("drop", "").strip().lower() in DROP_VALUES
                or r.get("verdict", "").strip().lower() == "dislike"}
    pairs, dropped, seen, new_rows = [], Counter(), set(), []
    for gen, chosen in generation_records(args.generations, args.choices):
        if gen["id"] in seen or gen["id"] in reserved or gen["id"] not in sections:
            dropped["duplicate, reserved or no section record"] += 1
            continue
        source, rec = sections[gen["id"]]
        lines = list(gen["lines"]) if gen.get("valid") and gen.get("lines") \
            else list(rec["lines"])
        reason = (None if gen.get("valid") and (chosen or gen.get("passed") is not False)
                  else "no usable answer" if not gen.get("valid") else "failed filters")
        if reason is None and args.require_approval:
            reason = "not approved yet"
        # sheet rows for every rewritten line (and every target line without an answer)
        rows = sorted({i for i, (a, b) in enumerate(zip(rec["lines"], lines)) if a != b}
                      | set(gen.get("targets", [])))
        for i in rows:
            if (gen["id"], i + 1) not in by_line:
                row = {"id": gen["id"], "line": str(i + 1), "artist": rec["artist"],
                       "title": rec["title"], "original": rec["lines"][i],
                       "rewrite": lines[i] if lines[i] != rec["lines"][i] else "",
                       "status": ("kept" if reason is None else "needs approval"
                                  if reason == "not approved yet" else f"dropped: {reason}")}
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
    write_edits(edits + new_rows, sheet)

    out = write_pairs(name, pairs)
    print(f"pairs: {len(pairs)} {dict(Counter(p['split'] for p in pairs))}, "
          f"with human edits: {sum(bool(p['human_edited']) for p in pairs)}")
    print(f"dropped: {dict(dropped)}")
    print(f"output: {out} (side by side: {name}.md, committed patch: {name}.patch.jsonl)")
    print(f"edit sheet: {sheet} ({len(new_rows)} new rows); vet: python -m radio_ready.vet "
          f"{name}")


if __name__ == "__main__":
    main()
