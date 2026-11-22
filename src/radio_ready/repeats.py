"""Exact repeats of a 4-line chunk inside a section (a chorus written out twice) are dropped.

repeat_chunks(lines)  (start of a repeat, start of its first copy) for every exact repeat of
                      REPEAT consecutive lines that does not overlap its first copy
drop_repeats(lines)   the lines without them; radio_ready.genius.split_sections uses it, so new
                      sections never contain them (before eligibility and sampling)

Sections built before that keep their ids: `python -m radio_ready.repeats --migrate` drops the
repeats from them in place and remaps everything that points at a line, with a backup of every
file it changes (data/processed/backup_before_repeats/):
  section files       data/processed/genius_sections.jsonl and kidzbop/*.jsonl: lines,
                      syllables, stress, rhyme_scheme (recomputed), kidz_bop_lines and
                      kidz_bop_edits; the record gets repeat_lines_dropped (old indexes) and
                      lines_before_repeats
  vetting sheets      annotations/vetting.csv and annotations/edits/*.csv of the checkouts given
                      (--sheets); a row on a dropped line moves to the same line of the first
                      copy (same text, so its diff still applies) and merges with a row there
  pairs patches       pairs/*.patch.jsonl of the same checkouts
  generations         outputs/generations/*.jsonl: lines, targets, rewrites, unit spans
  commercial table    data/commercial/commercial_table.csv: the section cell (answers pasted
                      with the old line count are still read: radio_ready.commercial)
A line of a dropped repeat that was rewritten where its first copy was not gives the first copy
its rewrite. Files already migrated are listed in data/processed/repeats_migrated.json and are
skipped on a second run.

Run:  python -m radio_ready.repeats --migrate [--checkouts DIR ...]
"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from radio_ready import paths

REPEAT = 4
MARKER = paths.PROCESSED / "repeats_migrated.json"
BACKUP = paths.PROCESSED / "backup_before_repeats"


def repeat_chunks(lines: list[str]) -> list[tuple[int, int]]:
    first: dict[tuple, int] = {}
    found, i = [], 0
    while i + REPEAT <= len(lines):
        chunk = tuple(lines[i:i + REPEAT])
        j = first.get(chunk)
        if j is not None and j + REPEAT <= i:
            found.append((i, j))
            i += REPEAT
            continue
        first.setdefault(chunk, i)
        i += 1
    return found


def drop_repeats(lines: list[str]) -> list[str]:
    drop = {i + k for i, _ in repeat_chunks(lines) for k in range(REPEAT)}
    return [l for n, l in enumerate(lines) if n not in drop]


def first_copies(lines: list[str]) -> dict[int, int]:
    """{dropped old index: the kept old index it repeats}. A first copy can itself overlap a
    later repeat, so the chain is followed back to a kept line."""
    drop = {i + k: j + k for i, j in repeat_chunks(lines) for k in range(REPEAT)}
    out = {}
    for k in drop:
        t = drop[k]
        while t in drop:
            t = drop[t]
        out[k] = t
    return out


def index_map(lines: list[str]) -> tuple[dict[int, int], list[int]]:
    """({old index: new index} for every old line, dropped old indexes). A dropped line maps
    to the same line of the first copy."""
    drop = first_copies(lines)
    keep = [n for n in range(len(lines)) if n not in drop]
    new = {old: n for n, old in enumerate(keep)}
    return {**new, **{k: new[t] for k, t in drop.items()}}, sorted(drop)


def merged_lines(old_lines: list[str], changed: list[str], mapping: dict[int, int],
                 dropped: list[int], first_of: dict[int, int]) -> list[str]:
    """`changed` (a rewrite of the old lines) on the new lines: a dropped line's rewrite goes to
    its first copy when that copy was left as it was."""
    out = list(changed)
    for k in dropped:
        t = first_of[k]
        if out[t] == old_lines[t] and changed[k] != old_lines[k]:
            out[t] = changed[k]
    return [l for n, l in enumerate(out) if n not in set(dropped)]


# --- migration ------------------------------------------------------------------------------

def backup(path: Path) -> None:
    rel = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:8] + "_" + path.name
    BACKUP.mkdir(parents=True, exist_ok=True)
    if not (BACKUP / rel).exists():
        shutil.copy2(path, BACKUP / rel)


def migrate_sections(path: Path, plans: dict) -> int:
    from radio_ready.text import rhyme_scheme

    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n = 0
    for r in records:
        if "repeat_lines_dropped" in r:
            continue
        chunks = repeat_chunks(r["lines"])
        if not chunks:
            continue
        old = list(r["lines"])
        mapping, dropped = index_map(old)
        first_of = first_copies(old)
        plans[r["id"]] = (old, mapping, dropped, first_of)
        keep = lambda xs: [x for i, x in enumerate(xs) if i not in set(dropped)]  # noqa: E731
        r["lines"] = keep(old)
        for f in ("syllables", "stress"):
            if isinstance(r.get(f), list) and len(r[f]) == len(old):
                r[f] = keep(r[f])
        if r.get("rhyme_scheme"):
            r["rhyme_scheme"] = rhyme_scheme(r["lines"])
        if r.get("kidz_bop_lines"):
            r["kidz_bop_lines"] = merged_lines(old, r["kidz_bop_lines"], mapping, dropped,
                                               first_of)
        if r.get("kidz_bop_edits"):
            seen, edits = set(), []
            for e in r["kidz_bop_edits"]:
                e = {**e, "line_index": mapping[e["line_index"]]}
                if e["line_index"] not in seen:
                    seen.add(e["line_index"])
                    edits.append(e)
            r["kidz_bop_edits"] = edits
        r["repeat_lines_dropped"] = dropped
        r["lines_before_repeats"] = len(old)
        n += 1
    if n:
        backup(path)
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(path)
    return n


def plan_of(lines: list[str]):
    """(old lines, index map, dropped indexes, {dropped: its first copy}) for one section."""
    mapping, dropped = index_map(lines)
    return lines, mapping, dropped, first_copies(lines)


def migrate_generation(path: Path, plans: dict) -> int:
    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n = 0
    for g in records:
        plan = plans.get(g["id"])
        if not plan:
            continue
        old, mapping, dropped, first_of = plan
        if g.get("lines") is not None:
            if len(g["lines"]) != len(old):
                continue  # not the old section: leave it
            g["lines"] = merged_lines(old, g["lines"], mapping, dropped, first_of)
            g["rewrites"] = {str(i): l for i, (l, o) in
                             enumerate(zip(g["lines"], drop_repeats(old))) if l != o}
        g["targets"] = sorted({mapping[t] for t in g.get("targets", []) if t in mapping})
        for u in g.get("units") or []:
            if isinstance(u, dict) and "start" in u and u["start"] in mapping:
                end = u["end"] - 1
                u["start"], u["end"] = mapping[u["start"]], mapping.get(end, end) + 1
        n += 1
    if n:
        backup(path)
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for g in records:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        tmp.replace(path)
    return n


def merge_row(into: dict, row: dict) -> None:
    from radio_ready.pairs import add_source

    if not into.get("verdict") and row.get("verdict"):
        into["verdict"] = row["verdict"]
    if not into.get("edit_diff") and row.get("edit_diff"):
        into["edit_diff"] = row["edit_diff"]
    if row.get("note") and row["note"] not in into.get("note", ""):
        into["note"] = "; ".join(x for x in (into.get("note"), row["note"]) if x)
    for part in [x for x in row.get("sources", "").split("; ") if x]:
        name, _, status = part.partition(": ")
        if f"{name}:" not in into.get("sources", ""):
            add_source(into, name, status)


def migrate_sheet(path: Path, plans: dict) -> int:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns, rows = reader.fieldnames, list(reader)
    if "line" not in (columns or []):
        return 0
    out, by_key, n = [], {}, 0
    for r in rows:
        plan = plans.get(r.get("id"))
        if plan and r["line"].isdigit():
            new = plan[1].get(int(r["line"]) - 1)
            if new is not None and new + 1 != int(r["line"]):
                r = {**r, "line": str(new + 1)}
                n += 1
        key = (r.get("id"), r.get("line"), r.get("rewrite_diff", r.get("rewrite", "")),
               r.get("source", ""))
        if key in by_key:
            merge_row(by_key[key], r)
            n += 1
            continue
        by_key[key] = r
        out.append(r)
    if n:
        backup(path)
        tmp = path.with_suffix(".csv.tmp")
        with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(out)
        tmp.replace(path)
    return n


def migrate_patch(path: Path, plans: dict) -> int:
    patches = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    out, seen, n = [], set(), 0
    for p in patches:
        plan = plans.get(p["id"])
        if plan:
            mapping = plan[1]
            if isinstance(p.get("diff"), dict):  # section pair: a first copy comes before
                diff, hv = {}, {}                   # its repeat, so it wins
                for k in sorted(p["diff"], key=int):
                    diff.setdefault(str(mapping[int(k)]), p["diff"][k])
                for k in sorted(p.get("human_verdicts") or {}, key=int):
                    hv.setdefault(str(mapping[int(k)]), p["human_verdicts"][k])
                p["diff"], p["human_verdicts"] = diff, hv
                p["human_edited"] = sorted({mapping[i] for i in p.get("human_edited") or []})
            elif "line" in p:  # line pair (1-based)
                p["line"] = mapping[p["line"] - 1] + 1
            n += 1
        key = json.dumps({k: v for k, v in p.items() if k not in ("from",)}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    if n:
        backup(path)
        with open(path, "w", encoding="utf-8") as f:
            for p in out:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return n


def migrate_commercial(path: Path, plans: dict, records: dict) -> int:
    from radio_ready.commercial import section_cell

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns, rows = reader.fieldnames, list(reader)
    n = 0
    for r in rows:
        if r.get("id") in plans and r["id"] in records:
            r["section"] = section_cell(records[r["id"]])
            n += 1
    if n:
        backup(path)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(rows)
    return n


def migrate(checkouts: list[Path]) -> None:
    done = json.loads(MARKER.read_text(encoding="utf-8")) if MARKER.exists() else []
    plans: dict = {}
    section_files = [paths.GENIUS_SECTIONS_JSONL, *sorted(paths.KIDZBOP_DIR.glob("*.jsonl"))]
    # plans come from the section files; a file migrated before keeps its plan in the backup
    for path in section_files:
        if str(path.resolve()) in done:
            bak = next(BACKUP.glob(f"*_{path.name}"), None)
            if bak:
                for l in open(bak, encoding="utf-8"):
                    r = json.loads(l)
                    if r["id"] not in plans and repeat_chunks(r["lines"]):
                        plans[r["id"]] = plan_of(r["lines"])
            continue
        print(f"{path}: {migrate_sections(path, plans)} sections without repeats", flush=True)
        done.append(str(path.resolve()))
    records = {}
    for path in section_files:
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in plans:
                records[r["id"]] = r
    todo = [(p, migrate_generation) for p in sorted(paths.GENERATIONS_DIR.glob("*.jsonl"))]
    for c in checkouts:
        todo += [(p, migrate_sheet) for p in [c / "annotations" / "vetting.csv",
                                             *sorted((c / "annotations" / "edits").glob("*.csv"))]
                 if p.exists()]
        todo += [(p, migrate_patch) for p in sorted((c / "pairs").glob("*.patch.jsonl"))]
    for path, fn in todo:
        if str(path.resolve()) in done:
            continue
        print(f"{path}: {fn(path, plans)} changed", flush=True)
        done.append(str(path.resolve()))
    ct = paths.COMMERCIAL_DIR / "commercial_table.csv"
    if ct.exists() and str(ct.resolve()) not in done:
        print(f"{ct}: {migrate_commercial(ct, plans, records)} section cells", flush=True)
        done.append(str(ct.resolve()))
    MARKER.write_text(json.dumps(done, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop exact 4-line repeats from sections.")
    parser.add_argument("--migrate", action="store_true")
    parser.add_argument("--checkouts", nargs="*", default=[str(paths.ROOT)],
                        help="checkouts whose annotations/ and pairs/ are remapped")
    args = parser.parse_args()
    if not args.migrate:
        parser.error("--migrate")
    migrate([Path(c) for c in args.checkouts])


if __name__ == "__main__":
    main()
