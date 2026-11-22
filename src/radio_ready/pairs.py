"""Synthetic pairs (explicit section -> radio edit): one vetting sheet, one pair set per source.

Sources (outputs/generations/):
  solid_*        radio_ready.rewrite runs (7B drafts, 14B for hard lines)
  commercial_*   commercial-LLM answers (radio_ready.commercial --export); need approval and
                 may use the Genius sections reserved for commercial evaluation
  rhyme_*        radio_ready.rhyme_swap runs: each explicit word swapped for a clean word that
                 rhymes with it (the end-rhyme thesaurus), the rest of the line kept; need
                 approval
  noise          the noise-style generations (*noise*; radio_ready.generate --style noise-*),
                 except the full-noise runs (FULL_NOISE: whole lines or stanzas turned into
                 made-up sounds, which the model copies back as gibberish). Only versions whose
                 every changed line keeps the meaning and fluency of its original (sensible())
                 reach the sheet; they need approval

Vetting sheet: annotations/vetting.csv (committed; written by radio_ready.vet). One row per
section line and proposed rewrite, so a rewrite that several sources produce for the same
line is vetted once. No lyrics are stored: rewrites and edits are word-level diffs against
the original line (radio_ready.linediff), filled in from the section records when loaded.
  id, line (1-based), artist, title
  rewrite_diff  the proposed rewrite ("" for an explicit line no source rewrote)
  edit_diff     older sheets only: your line, stored on the proposal's row (it replaced the
                rewrite); the app now writes every line you change as a row of its own
  verdict       like / superlike / dislike; "edited" on a proposal you only rewrote
  note          free text
  sources       the sources that proposed this rewrite, with their status (kept / dropped:
                reason / needs approval / your version: a line you wrote while vetting a
                proposal of that source)
Rows that only a removed source (REMOVED_SOURCES) proposed are dropped from the sheet on export.
Rules for one version of a section (all its lines, as one source proposed them):
  - every accepted version of a line counts: the proposal when liked or superliked (or when it
    passed, the source needs no approval and you have not vetted it) and each of your own versions
    of it that you liked or superliked. Each one is in a pair of its own: the first accepted
    version of every line, plus one pair per further version of a line;
  - a line with no accepted version leaves the section out when it was disliked, or when the
    source needs approval or the filters dropped it;
  - nothing explicit may be left.

Outputs, in pairs/:
  <source>.patch.jsonl  committed, no lyrics: ids, split, how it was made, verdicts and a
                        word-level diff of each changed line
  <source>.jsonl/.md    full pairs and a side-by-side view, gitignored (lyrics)
  final.*               --combine: every source, identical pairs (same section, same rewrite)
                        kept once; `from` lists every source that produced the pair and
                        `edited` says whether a line you wrote is in it (both kept in the
                        patch, so --rebuild restores them)
  lines.*               --combine: line pairs, one per line you liked or superliked (your edit
                        if you made one), even when its section version is out because of a
                        disliked line; disliked lines never appear. Each has the line, its
                        section as context, and your `verdict` (like / superlike) as a plain
                        field; one per (section, line, final text)
--rebuild recreates every .jsonl / .md from the patches and the section records (after the
download, kidzbop and genius steps). Splits: Genius by artist (60/20/20 as radio_ready.labels),
Kidz Bop by song. A full pair: id, source (genius / kidzbop), artist, title, section_header,
split, original, rewrite, changed, model, prompt_version, style, human_edited, human_verdicts.
No training features are added.

Run:  python -m radio_ready.pairs SOURCE [SOURCE ...]     # e.g. solid_pilot_solid-v1
      python -m radio_ready.pairs --all                    # every source
      python -m radio_ready.pairs --combine                # pairs/final.*
      python -m radio_ready.pairs --rebuild
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

PAIRS_DIR = paths.ROOT / "pairs"
SECTION_FILES = {"genius": paths.GENIUS_SECTIONS_JSONL,
                 "kidzbop": paths.KIDZBOP_DIR / "sections.jsonl"}
SHEET = paths.ANNOTATIONS / "vetting.csv"
OLD_SHEETS = [paths.ANNOTATIONS / "pair_edits.csv", *sorted((paths.ANNOTATIONS / "edits")
                                                             .glob("*.csv"))]
SHEET_COLUMNS = ["id", "line", "artist", "title", "rewrite_diff", "edit_diff", "verdict",
                 "note", "sources"]
APPROVE = {"like", "superlike"}
REMOVED_SOURCES: set[str] = set()  # sources whose rows are dropped from the sheet on export
NOISE = "noise"
# full-noise styles (whole lines / stanzas turned into made-up sounds): the model copies the
# gibberish back, so their runs are not part of the noise source; your verdicts on their lines
# stay on the sheet, and the lines you liked stay line pairs
FULL_NOISE = ("-noise-line", "-noise-stanza")
MAX_FLUENCY_DROP = 0.3  # noise sense filter: CoLA acceptability lost by a changed line
NONSENSE = "nonsense"  # source status of rows only a sense-filtered version proposed
YOURS = "your version"  # source status of a line you wrote in the vetting app


def only_nonsense(row: dict) -> bool:
    """Every source that proposed this row marks it nonsense (vet skips these rows)."""
    parts = [s for s in row.get("sources", "").split("; ") if s]
    return bool(parts) and all(s.split(": ", 1)[-1] == NONSENSE for s in parts)
FINAL = "final"
LINES = "lines"


# --- sections and generation files ----------------------------------------------------------

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


def sources() -> dict[str, dict]:
    """{source name: {files, require_approval, include_reserved}}."""
    found = {}
    gens = sorted(paths.GENERATIONS_DIR.glob("*.jsonl"))
    for g in gens:
        stem = g.name.removesuffix(".jsonl")
        if stem.startswith("solid_"):
            found[stem] = {"files": [g], "require_approval": False, "include_reserved": False}
        elif stem.startswith("commercial_"):
            found[stem] = {"files": [g], "require_approval": True, "include_reserved": True}
        elif stem.startswith(("rhyme_", "free_", "guided_")):
            found[stem] = {"files": [g], "require_approval": True, "include_reserved": False}
    noise = [g for g in gens if "noise" in g.name and not any(f in g.name for f in FULL_NOISE)]
    if noise:
        found[NOISE] = {"files": noise, "require_approval": True, "include_reserved": False,
                        "sense_filter": True}
    return found


_SCORERS: dict = {}


def sensible(original: list[str], lines: list[str]) -> bool:
    """Every changed line keeps the meaning (all-distilroberta-v1 cosine to its original >=
    0.70, the ParaDeHate threshold) and reads as well as the original (CoLA acceptability drop
    <= MAX_FLUENCY_DROP). Filters out nonsense lines before anyone has to vet them."""
    from radio_ready.scoring import SIMILARITY_THRESHOLD, Fluency, Similarity

    changed = [i for i, (a, b) in enumerate(zip(original, lines)) if a != b]
    if not changed:
        return False
    if not _SCORERS:
        _SCORERS.update(sim=Similarity(on="cpu"), flu=Fluency(on="cpu"))
    a, b = [original[i] for i in changed], [lines[i] for i in changed]
    sims = _SCORERS["sim"].pairwise(b, a)
    drop = _SCORERS["flu"].score(a) - _SCORERS["flu"].score(b)
    return bool((sims >= SIMILARITY_THRESHOLD).all() and (drop <= MAX_FLUENCY_DROP).all())


# --- vetting sheet --------------------------------------------------------------------------

def row_key(r: dict) -> tuple[str, int, str]:
    return r["id"], int(r["line"]), r.get("rewrite_diff", "")


def read_sheet(path: Path = SHEET) -> list[dict]:
    """Rows as stored; fill_text() adds original / rewrite / edit text."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fill_text(rows: list[dict], lines_of) -> None:
    """Add original, rewrite and edit text; lines_of(section id) -> lines or None."""
    for r in rows:
        lines = lines_of(r["id"])
        n = int(r["line"]) - 1
        if not lines or not 0 <= n < len(lines):
            r["original"], r["rewrite"], r["edit"] = "", "", ""
            continue
        r["original"] = lines[n]
        for col, out in (("rewrite_diff", "rewrite"), ("edit_diff", "edit")):
            r[out] = apply(lines[n], json.loads(r[col])) if r.get(col) else ""


def diff_cell(original: str, text: str) -> str:
    return json.dumps(diff(original, text), ensure_ascii=False)


def write_sheet(rows: list[dict], path: Path = SHEET) -> None:
    """Store rows with the edit as a diff (the rewrite diff is the row's key and never changes).
    An edit equal to the original line is kept: it means "keep the original line"."""
    out = []
    for r in rows:
        row = dict(r)
        if r.get("original") is not None and "edit" in r:
            edit = (r.get("edit") or "").strip()
            row["edit_diff"] = diff_cell(r["original"], edit) if edit else ""
        out.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in SHEET_COLUMNS} for r in out)
    tmp.replace(path)


def source_names(r: dict) -> set[str]:
    return {s.split(":", 1)[0] for s in r.get("sources", "").split("; ") if s}


def add_source(r: dict, name: str, status: str) -> None:
    parts = [s for s in r.get("sources", "").split("; ") if s and s.split(":", 1)[0] != name]
    r["sources"] = "; ".join(sorted(parts + [f"{name}: {status}"]))


def migrate(sections) -> list[dict]:
    """Rows of the older per-export sheets (and the full-text pair_edits.csv), merged into the
    one-sheet layout; verdicts, edits and notes are kept, vetted rows win on a clash."""
    merged: dict[tuple, dict] = {}
    for old in OLD_SHEETS:
        if not old.exists():
            continue
        name = old.stem if old.parent.name == "edits" else "solid_pilot_solid-v1"
        for r in read_sheet(old):
            if not r.get("line", "").isdigit() or r["id"] not in sections:
                continue
            lines = sections[r["id"]][1]["lines"]
            original = lines[int(r["line"]) - 1]
            if "rewrite_diff" in r:
                rewrite = apply(original, json.loads(r["rewrite_diff"])) \
                    if r["rewrite_diff"] else ""
                edit = apply(original, json.loads(r["edit_diff"])) if r.get("edit_diff") else ""
            else:  # full-text layout
                rewrite, edit = r.get("rewrite", ""), r.get("edit", "")
            row = {"id": r["id"], "line": r["line"], "artist": r["artist"], "title": r["title"],
                   "rewrite_diff": diff_cell(original, rewrite) if rewrite else "",
                   "edit_diff": diff_cell(original, edit) if edit else "",
                   "verdict": r.get("verdict", ""), "note": r.get("note", ""), "sources": ""}
            add_source(row, name, r.get("status", "") or "kept")
            key = row_key(row)
            if key in merged:
                add_source(merged[key], name, r.get("status", "") or "kept")
                if row["verdict"] and not merged[key]["verdict"]:
                    merged[key].update(verdict=row["verdict"], edit_diff=row["edit_diff"],
                                       note=row["note"])
            else:
                merged[key] = row
    return list(merged.values())


def add_your_version(rows: list[dict], base: dict, text: str, verdict: str,
                     names: set[str]) -> dict | None:
    """Your own version `text` of the line of `base` (a row with text filled in), as a row of
    its own with status YOURS for each source in `names` (the list you vetted it in). The same
    text written in another list is the same row (one more source); different texts are
    different rows, so every version you liked is a pair of its own. A row with this text and
    a different verdict (given in another list) is never changed: returns None, and the caller
    keeps your version as the edit of `base` instead, so both verdicts stand."""
    cell = diff_cell(base["original"], text)
    row = next((r for r in rows if r["id"] == base["id"] and r["line"] == base["line"]
                and r.get("rewrite_diff", "") == cell), None)
    if row is not None and verdict and row.get("verdict") and row["verdict"] != verdict:
        return None
    if row is None:
        row = {"id": base["id"], "line": base["line"], "artist": base["artist"],
               "title": base["title"], "rewrite_diff": cell, "original": base["original"],
               "rewrite": text, "edit": "", "verdict": "", "note": "", "sources": ""}
        rows.append(row)
    if verdict:
        row["verdict"] = verdict
    for n in names:
        if n not in source_names(row):
            add_source(row, n, YOURS)
    return row


def split_edits(rows: list[dict]) -> int:
    """Older sheets stored your line as the edit of the proposal's row, so a second edit (in
    another list) replaced the first. Each such edit becomes a row of its own (add_your_version,
    for the sources of the row, keeping its verdict; an edit counted as accepted, so a blank
    verdict becomes like); the proposal keeps no edit and is marked "edited" (your verdict on
    it as proposed is unknown). Returns how many edits were split off."""
    n = 0
    for r in list(rows):
        edit = (r.get("edit") or "").strip()
        if not edit:
            continue
        verdict = r.get("verdict", "").strip().lower()
        if edit != r.get("rewrite", ""):
            mine = add_your_version(rows, r, edit,
                                    "" if verdict == "dislike" else verdict or "like",
                                    source_names(r) or {"unknown"})
            if mine is None:  # the same text has another verdict in another list: keep both
                continue
        r["edit"], r["edit_diff"] = "", ""
        if verdict != "dislike":
            r["verdict"] = "edited"
        n += 1
    return n


def load_sheet(sections) -> list[dict]:
    """The vetting sheet with text filled in. Rows of older per-export sheets that still exist
    are merged in on every load (a verdict there fills a row without one here), so vetting
    done in an older version of the app is never lost."""
    rows = read_sheet()
    by_key = {row_key(r): r for r in rows}
    for old in migrate(sections):
        row = by_key.get(row_key(old))
        if row is None:
            rows.append(old)
            by_key[row_key(old)] = old
        elif old.get("verdict") and not row.get("verdict"):
            row.update(verdict=old["verdict"], edit_diff=old.get("edit_diff", ""),
                       note=old.get("note", "") or row.get("note", ""))
    fill_text(rows, lambda sid: sections[sid][1]["lines"] if sid in sections else None)
    split_edits(rows)
    return rows


# --- pairs ----------------------------------------------------------------------------------

PATCH_KEYS = ["id", "source", "from", "edited", "artist", "title", "section_header", "split",
              "model", "prompt_version", "style", "human_edited", "human_verdicts"]
PAIR_KEYS = ["id", "source", "from", "edited", "artist", "title", "section_header", "split",
             "original", "rewrite", "changed", "model", "prompt_version", "style",
             "human_edited", "human_verdicts"]


def to_pair(gen: dict, source: str, rec: dict, lines: list[str]) -> dict:
    return {
        "id": rec["id"], "source": source, "artist": rec["artist"], "title": rec["title"],
        "section_header": rec.get("section_header"),
        "split": rec.get("split") if source == "kidzbop" else artist_split(rec["artist"]),
        "original": rec["lines"], "rewrite": lines,
        "changed": [i for i, (a, b) in enumerate(zip(rec["lines"], lines)) if a != b],
        "model": gen.get("model"), "prompt_version": gen.get("prompt_version"),
        "style": gen.get("style"),
    }


def to_patch(pair: dict) -> dict:
    """The pair without lyrics: metadata and a word-level diff of each changed line."""
    return {**{k: pair.get(k) for k in PATCH_KEYS},
            "diff": {str(i): diff(pair["original"][i], pair["rewrite"][i])
                     for i in pair["changed"]}}


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


def write_side_by_side(path: Path, pairs: list[dict]) -> None:
    """Readable markdown: one table per pair, changed lines in bold, unchanged lines "="."""
    esc = lambda t: str(t).replace("|", r"\|")  # noqa: E731
    out = [f"# {path.stem}: original vs rewrite", "",
           f"{len(pairs)} pairs. Changed lines in bold; (edited) marks your edits.", ""]
    for p in pairs:
        out += [f"## {esc(p['artist'])} - {esc(p['title'])} "
                f"({esc(p['section_header'] or '')}, {p['split']}, `{p['id']}`)", "",
                f"from {', '.join(p.get('from') or [])}"
                f"{'; has lines you wrote' if p.get('edited') else ''}", "",
                "| # | original | rewrite |", "|---|---|---|"]
        for i, (o, r) in enumerate(zip(p["original"], p["rewrite"])):
            mark = " (edited)" if i in (p.get("human_edited") or []) else ""
            out.append(f"| {i + 1} | **{esc(o)}** | **{esc(r)}**{mark} |" if o != r
                       else f"| {i + 1} | {esc(o)} | = |")
        out.append("")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_pairs(name: str, pairs: list[dict]) -> Path:
    """Full pairs and side-by-side (gitignored) plus the committed patch file."""
    out = PAIRS_DIR / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, pairs)
    write_side_by_side(PAIRS_DIR / f"{name}.md", pairs)
    write_jsonl(PAIRS_DIR / f"{name}.patch.jsonl", (to_patch(p) for p in pairs))
    return out


def your_versions(rows: list[dict]) -> dict[tuple[str, int], list[dict]]:
    """{(id, line): rows of your own versions (status YOURS)}."""
    found: dict = {}
    for r in rows:
        if any(x.split(": ", 1)[-1] == YOURS for x in r.get("sources", "").split("; ") if x):
            found.setdefault((r["id"], int(r["line"])), []).append(r)
    return found


def alternatives(row: dict, verdict: str, proposed: str, auto: bool, yours: list[dict],
                 name: str) -> list[tuple[str, bool, str]]:
    """Every accepted version of one line as (text, written by you, verdict), best first: the
    proposal
    when you liked or superliked it (or it passed, needs no approval and you have not vetted
    it: a dislike or your own version of it means it is not accepted as is),
    then each of your own versions of it you liked or superliked (rows with status YOURS for
    this source), then an edit stored on the row itself (older sheets: it replaced the
    proposal)."""
    found = []
    if proposed and (verdict in APPROVE or (auto and not verdict)):  # "edited": not as is
        found.append((proposed, False, verdict))
    for r in sorted(yours, key=lambda r: r.get("verdict") != "superlike"):
        v = r.get("verdict", "").strip().lower()
        if name in source_names(r) and v in APPROVE:
            found.append((r["rewrite"], True, v))
    old = (row.get("edit") or "").strip()
    if old and verdict != "dislike":
        found.append((old, True, verdict))
    seen, out = set(), []
    for alt in found:
        if alt[0] not in seen:
            seen.add(alt[0])
            out.append(alt)
    return out


def versions(lines: list[str], alts: dict[int, list[tuple[str, bool, str]]],
             verdicts: dict[str, str]):
    """(section lines, lines you wrote, verdicts of the lines used) for every accepted version:
    the first alternative of each line, then one more version for each further alternative of
    a line (the other lines as in the first). Every liked line is thus in a pair of its own."""
    base, mine, used = list(lines), set(), dict(verdicts)
    for i, found in alts.items():
        if found:
            base[i] = found[0][0]
            if found[0][1]:
                mine.add(i)
            if found[0][2]:
                used[str(i)] = found[0][2]
    yield base, sorted(mine), used
    for i, found in alts.items():
        for text, by_you, verdict in found[1:]:
            v = list(base)
            v[i] = text
            yield (v, sorted((mine - {i}) | ({i} if by_you else set())),
                   {**used, str(i): verdict} if verdict else used)


def export_source(name: str, spec: dict, sections, rows: list[dict], detector) -> str:
    """Pairs of one source; new rows are added to `rows` (the sheet). Returns a summary."""
    by_key = {row_key(r): r for r in rows}
    yours = your_versions(rows)
    reserved = set() if spec["include_reserved"] else reserved_ids()
    pairs, dropped, seen, new = [], Counter(), set(), 0
    for path in spec["files"]:
        for line in open(path, encoding="utf-8"):
            gen = json.loads(line)
            if gen["id"] in reserved or gen["id"] not in sections:
                dropped["reserved or no section record"] += 1
                continue
            source, rec = sections[gen["id"]]
            lines = list(gen["lines"]) if gen.get("valid") and gen.get("lines") \
                else list(rec["lines"])
            reason = (None if gen.get("valid") and gen.get("passed") is not False
                      else "no usable answer" if not gen.get("valid") else "failed filters")
            if reason is None and spec["require_approval"]:
                reason = "needs approval"
            status = "kept" if reason is None else (
                reason if reason == "needs approval" else f"dropped: {reason}")
            changed_rows = [by_key.get((gen["id"], i + 1, diff_cell(a, b)))
                            for i, (a, b) in enumerate(zip(rec["lines"], lines)) if a != b]
            human_ok = bool(changed_rows) and all(
                r is not None and (r.get("verdict", "").strip().lower() in APPROVE
                                   or (r.get("edit") or "").strip()
                                   or any(y.get("verdict") in APPROVE for y in
                                          yours.get((r["id"], int(r["line"])), [])))
                for r in changed_rows)
            if spec.get("sense_filter") and not human_ok and not sensible(rec["lines"], lines):
                dropped["nonsense (sense filter)"] += 1
                for i, (a, b) in enumerate(zip(rec["lines"], lines)):
                    row = by_key.get((gen["id"], i + 1, diff_cell(a, b))) if a != b else None
                    if row is not None:  # already in the sheet: mark it, so vet skips it
                        add_source(row, name, NONSENSE)
                continue
            todo = sorted({i for i, (a, b) in enumerate(zip(rec["lines"], lines)) if a != b}
                          | set(gen.get("targets", [])))
            verdicts, alts, blocked = {}, {}, None
            for i in todo:
                proposed = lines[i] if lines[i] != rec["lines"][i] else ""
                key = (gen["id"], i + 1, diff_cell(rec["lines"][i], proposed) if proposed
                       else "")
                row = by_key.get(key)
                if row is None:
                    row = {"id": gen["id"], "line": str(i + 1), "artist": rec["artist"],
                           "title": rec["title"], "rewrite_diff": key[2],
                           "original": rec["lines"][i], "rewrite": proposed, "edit": "",
                           "verdict": "", "note": "", "sources": ""}
                    by_key[key] = row
                    rows.append(row)
                    new += 1
                add_source(row, name, status)
                verdict = row.get("verdict", "").strip().lower()
                if verdict:
                    verdicts[str(i)] = verdict
                alts[i] = alternatives(row, verdict, proposed, reason is None,
                                       yours.get((gen["id"], i + 1), []), name)
                if not alts[i] and blocked is None and (verdict == "dislike" or reason):
                    blocked = "disliked" if verdict == "dislike" else reason
            if blocked:
                dropped[blocked] += 1
                continue
            for version, edited, used in versions(lines, alts, verdicts):
                key = (gen["id"], tuple(version))
                if version == rec["lines"]:
                    dropped["nothing changed"] += 1
                elif detector.find("\n".join(version)):
                    dropped["explicit word left" + (" after edits" if edited else "")] += 1
                elif key in seen:
                    dropped["duplicate version"] += 1
                else:
                    seen.add(key)
                    pair = to_pair(gen, source, rec, version)
                    pair["human_edited"], pair["human_verdicts"] = edited, used
                    pair["from"], pair["edited"] = [name], bool(edited)
                    pairs.append(pair)
    write_pairs(name, pairs)
    return (f"{name}: {len(pairs)} pairs {dict(Counter(p['split'] for p in pairs))}, "
            f"{sum(bool(p['human_edited']) for p in pairs)} with your edits; {new} new lines "
            f"to vet; not paired: {dict(dropped)}")


def export(names: list[str] | None = None) -> str:
    """Export the named sources (default all) and update the sheet. Returns the summaries."""
    sections, detector, specs = load_sections(), ExplicitDetector(), sources()
    rows = load_sheet(sections)
    lines = []
    for name in names or list(specs):
        if name not in specs:
            lines.append(f"{name}: unknown source (known: {', '.join(specs)})")
            continue
        lines.append(export_source(name, specs[name], sections, rows, detector))
    for r in rows:  # forget removed sources; rows only they proposed are dropped
        kept = [x for x in r.get("sources", "").split("; ")
                if x and x.split(":", 1)[0] not in REMOVED_SOURCES]
        r["sources"] = "; ".join(kept)
    rows[:] = [r for r in rows if r["sources"]]
    write_sheet(rows)
    return "\n".join(lines)


LINE_PATCH_KEYS = ["id", "line", "source", "artist", "title", "section_header", "split",
                   "verdict", "edited", "proposed_by"]


def line_patches(rows: list[dict], sections, detector) -> list[dict]:
    """Lyric-free line pairs: every liked or superliked row (text filled), deduplicated."""
    out, seen = [], set()
    for r in rows:
        verdict = r.get("verdict", "").strip().lower()
        if verdict not in APPROVE or r["id"] not in sections:
            continue
        final = (r.get("edit") or "").strip() or r.get("rewrite", "")
        if not final or final == r["original"] or detector.find(final):
            continue
        key = (r["id"], int(r["line"]), final)
        if key in seen:
            continue
        seen.add(key)
        source, rec = sections[r["id"]]
        out.append({"id": r["id"], "line": int(r["line"]), "source": source,
                    "artist": rec["artist"], "title": rec["title"],
                    "section_header": rec.get("section_header"),
                    "split": rec.get("split") if source == "kidzbop"
                    else artist_split(rec["artist"]),
                    "verdict": verdict, "edited": bool((r.get("edit") or "").strip())
                    or any(x.split(": ", 1)[-1] == YOURS
                           for x in r.get("sources", "").split("; ") if x),
                    "proposed_by": sorted(source_names(r)),
                    "diff": diff(r["original"], final)})
    return out


def from_line_patch(patch: dict, rec: dict) -> dict:
    """A full line pair: the original line, the rewrite and the section as context."""
    original = rec["lines"][patch["line"] - 1]
    return {**{k: patch.get(k) for k in LINE_PATCH_KEYS}, "original": original,
            "rewrite": apply(original, patch["diff"]), "section": rec["lines"]}


def write_lines(patches: list[dict], sections) -> list[dict]:
    pairs = [from_line_patch(p, sections[p["id"]][1]) for p in patches if p["id"] in sections]
    write_jsonl(PAIRS_DIR / f"{LINES}.patch.jsonl", patches)
    write_jsonl(PAIRS_DIR / f"{LINES}.jsonl", pairs)
    esc = lambda t: str(t).replace("|", r"\|")  # noqa: E731
    md = [f"# {LINES}: line pairs you liked or superliked", "", f"{len(pairs)} line pairs.", "",
          "| song | line | original | rewrite | verdict |", "|---|---|---|---|---|"]
    md += [f"| {esc(p['artist'])} - {esc(p['title'])} | {p['line']} | {esc(p['original'])} | "
           f"{esc(p['rewrite'])}{' (edited)' if p['edited'] else ''} | {p['verdict']} |"
           for p in pairs]
    (PAIRS_DIR / f"{LINES}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return pairs


def combine() -> str:
    """pairs/final.*: every source's pairs, identical pairs (same section, same rewrite) once;
    pairs/lines.*: every liked or superliked line as a pair of its own."""
    sections = load_sections()
    seen, pairs, counts = {}, [], Counter()
    for patch_file in sorted(PAIRS_DIR.glob("*.patch.jsonl")):
        name = patch_file.name.removesuffix(".patch.jsonl")
        if name in (FINAL, LINES):
            continue
        for line in open(patch_file, encoding="utf-8"):
            patch = json.loads(line)
            if patch["id"] not in sections:
                counts["section not found"] += 1
                continue
            pair = from_patch(patch, sections[patch["id"]][1])
            key = (pair["id"], tuple(pair["rewrite"]))
            if key in seen:  # the same pair from another list: note it there
                counts["duplicate"] += 1
                first = seen[key]
                first["from"] = sorted(set(first["from"]) | {name})
                first["edited"] = first["edited"] or bool(pair["human_edited"])
                continue
            pair["from"], pair["edited"] = [name], bool(pair["human_edited"])
            seen[key] = pair
            pairs.append(pair)
            counts[name] += 1
    write_pairs(FINAL, pairs)
    lines = write_lines(line_patches(load_sheet(sections), sections, ExplicitDetector()),
                        sections)
    return (f"final: {len(pairs)} pairs {dict(Counter(p['split'] for p in pairs))}, "
            f"{sum(p['edited'] for p in pairs)} with a line you wrote; from {dict(counts)}\n"
            f"lines: {len(lines)} line pairs {dict(Counter(p['split'] for p in lines))}, "
            f"{dict(Counter(p['verdict'] for p in lines))}")


def rebuild() -> str:
    sections, out = load_sections(), []
    for patch_file in sorted(PAIRS_DIR.glob("*.patch.jsonl")):
        name = patch_file.name.removesuffix(".patch.jsonl")
        patches = [json.loads(l) for l in open(patch_file, encoding="utf-8") if l.strip()]
        missing = [p["id"] for p in patches if p["id"] not in sections]
        if name == LINES:
            out.append(f"{LINES}: {len(write_lines(patches, sections))} line pairs rebuilt")
            continue
        pairs = [from_patch(p, sections[p["id"]][1]) for p in patches if p["id"] in sections]
        write_jsonl(PAIRS_DIR / f"{name}.jsonl", pairs)
        write_side_by_side(PAIRS_DIR / f"{name}.md", pairs)
        note = (f", {len(missing)} sections not found (run the download, kidzbop and genius "
                f"steps)") if missing else ""
        out.append(f"{name}: {len(pairs)} pairs rebuilt{note}")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export, combine or rebuild synthetic pairs.")
    parser.add_argument("sources", nargs="*", help="source names (see the module docstring)")
    parser.add_argument("--all", action="store_true", help="export every source")
    parser.add_argument("--combine", action="store_true", help="write pairs/final.*")
    parser.add_argument("--rebuild", action="store_true",
                        help="full pairs from pairs/*.patch.jsonl and the section records")
    args = parser.parse_args()
    if args.rebuild:
        print(rebuild())
    if args.sources or args.all:
        print(export(None if args.all else args.sources))
    if args.combine:
        print(combine())
    if not (args.rebuild or args.sources or args.all or args.combine):
        parser.error("give sources, --all, --combine or --rebuild")


if __name__ == "__main__":
    main()
