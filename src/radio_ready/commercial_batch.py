"""Batch prompts for Claude / Gemini: many new explicit lines in one chat message, one answer
pasted back as a whole.

Items are Genius lines you do not have yet. A section is left out when it is already in the
vetting sheet, a generation run, the commercial table or an earlier batch (old or new), and
so is a line whose text (lower case, letters and digits only) is already in one of those
sections or already in this batch. Sections are taken in a fixed hash order, so the batches
never overlap; each song gives at most one new section per batch.

Version 1 (default): one item per section
  stanza  a 4-line stanza (the blocks of radio_ready.prompts) whose lines are all explicit,
          rewritten as a whole
  line    otherwise one explicit line on its own (the first new one), with no context

Version 2 (--version 2): windows of the line before, the explicit line and the line after,
every explicit word in the whole window rewritten (the model may reword any of its lines).
The windows of a section cover all its explicit lines (a new window starts at the first
explicit line no earlier window holds). Windows that overlap are never in the same batch:
each goes in a later round, and a later batch shows it with the rewrites of the earlier
rounds already applied (the answers of --model). So a version 2 batch holds the next round
of the sections already started whose earlier windows are answered, then --items new
sections.

  data/commercial/batches/batch_NNN.txt          the prompt with the items: paste it in a chat
  data/commercial/batches/batch_NNN.json         which section and lines each item is
  data/commercial/batches/batch_NNN_<model>.txt  paste that model's whole answer here

--export reads every answer file into outputs/generations/ (the radio_ready.generate format):
commercial_batch_<model>.jsonl for version 1 (one record per item; targets = its lines) and
commercial_window_<model>.jsonl for version 2 (one record per section with every answered
window applied in order; targets = its explicit and changed lines). The vetting app lists
each as its own source (approval required). An item whose answer is missing or has the wrong
number of lines is left out and reported.

Run:  python -m radio_ready.commercial_batch [--items 50]
      python -m radio_ready.commercial_batch --version 2 [--items 30] [--model claude]
      python -m radio_ready.commercial_batch --export
"""

import argparse
import csv
import json
import re

from radio_ready import paths
from radio_ready.commercial import TABLE, hash_rank, read_jsonl
from radio_ready.explicit import ExplicitDetector
from radio_ready.prompts import clean_response_line

VERSION = "commercial-batch-v1"
VERSION_2 = "commercial-batch-v2"
BATCH_DIR = paths.COMMERCIAL_DIR / "batches"
OLD_REQUESTS = paths.COMMERCIAL_DIR / "requests"
SEP = " / "
BLOCK = 4
ANSWER = re.compile(r"^\s*\**\s*\[?(\d+)\s*[\].):]\**\s*(.*)$")

PROMPT = """\
You write radio edits of songs. Below are {n} numbered items. Each is one lyric line, or a \
4-line stanza with its lines separated by " / ", from the song named before it. Every item has \
words that would be bleeped on daytime radio.

Rewrite every item so nothing in it would be bleeped:
- Remove every profanity, slur, and sexual or drug term. Do not respell, abbreviate, star out \
or hint at them ("b*tch", "fuk", "n-word", "sh..."); write a different word or phrase.
- Rewrite as much of the line as it needs to sound natural, not just the censored word, but \
keep what the line already says. Do not add new images, story, jokes or a moral.
- Keep the artist's voice: their slang, attitude, dialect and spelling.
- Keep the rhythm and rhyme: about the same number of syllables, the end rhyme (the same last \
word, or one that rhymes with it) and any internal rhyme you can.
- A stanza keeps its four lines and its rhyme scheme.
- Give only your single best version: no options, notes or explanations.

Answer with one row per item, in order, numbered like the items, and nothing else:
1. clean line
2. clean line one / clean line two / clean line three / clean line four

{items}
"""

PROMPT_2 = """\
You write radio edits of songs. Below are {n} numbered items. Each is two or three \
consecutive lyric lines, separated by " / ", from the song named before it. Some lines may \
already be clean from an earlier edit, but every item still has words that would be bleeped \
on daytime radio.

Rewrite every item so nothing in any of its lines would be bleeped:
- Remove every profanity, slur, and sexual or drug term from every line. Do not respell, \
abbreviate, star out or hint at them ("b*tch", "fuk", "n-word", "sh..."); write a different \
word or phrase.
- You may reword any line of the item as much as it needs to sound natural and lead into the \
next line, but keep what the lines already say. Do not add new images, story, jokes or a \
moral. A line that is already clean can stay as it is.
- Keep the artist's voice: their slang, attitude, dialect and spelling.
- Keep the rhythm and rhyme of every line: about the same number of syllables, the end rhyme \
(the same last word, or one that rhymes with it) and any internal rhyme you can, so the lines \
still rhyme with each other as they did.
- Give only your single best version: no options, notes or explanations.

Answer with one row per item, in order, numbered like the items, with the item's lines \
separated by " / " (the same number of lines as the item), and nothing else:
1. clean line one / clean line two / clean line three

{items}
"""


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", "", text.lower()).split())


def batch_files() -> list:
    return sorted(BATCH_DIR.glob("batch_[0-9][0-9][0-9].json"))


def manifests() -> list[dict]:
    return [{**json.loads(p.read_text(encoding="utf-8")), "stem": p.stem}
            for p in batch_files()]


def done_ids() -> set[str]:
    """Sections already vetted, generated, in the commercial table or in a batch."""
    ids = set()
    sheet = paths.ANNOTATIONS / "vetting.csv"
    for path in [sheet, TABLE, *OLD_REQUESTS.glob("*_table.csv")]:
        if path.exists():
            with open(path, encoding="utf-8-sig", newline="") as f:
                ids |= {r["id"] for r in csv.DictReader(f) if r.get("id")}
    for path in paths.GENERATIONS_DIR.glob("*.jsonl"):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                ids.add(json.loads(line)["id"])
    for meta in manifests():
        ids |= {it["id"] for it in meta["items"]} | set(meta.get("plans", {}))
    return ids


def pick(record: dict, flagged: set[int], seen: set[str]) -> list[int] | None:
    """The item of one section: an all-explicit stanza, else one new explicit line."""
    lines = record["lines"]
    if any(SEP.strip() in l for l in lines):
        return None
    for start in range(0, len(lines), BLOCK):
        block = list(range(start, min(start + BLOCK, len(lines))))
        if len(block) == BLOCK and all(i in flagged for i in block) \
                and not any(norm(lines[i]) in seen for i in block):
            return block
    for i in sorted(flagged):
        if norm(lines[i]) not in seen:
            return [i]
    return None


def windows(flagged: set[int], n: int) -> list[dict]:
    """Windows (line before, explicit line, line after) covering every explicit line, each
    with its round: one after the latest round of the earlier windows it overlaps."""
    out = []
    for f in sorted(flagged):
        if any(f in w["lines"] for w in out):
            continue
        lines = list(range(max(0, f - 1), min(n, f + 2)))
        before = [w["round"] for w in out if set(w["lines"]) & set(lines)]
        out.append({"lines": lines, "round": max(before, default=0) + 1})
    return out


def new_sections(by_id: dict, records: list[dict]):
    """New sections in hash order, as (record, seen line texts, songs taken, its song): the
    caller adds what it takes to `seen` and `songs`."""
    done = done_ids()
    seen = {norm(l) for sid in done if sid in by_id for l in by_id[sid]["lines"]}
    songs = set()
    for record in sorted(records, key=lambda r: hash_rank(r["id"])):
        song = (record["artist"].lower(), record["title"].lower())
        if record["id"] in done or song in songs:
            continue
        yield record, seen, songs, song


def write_batch(prompt: str, meta: dict, shown: list[str]) -> str | None:
    if not shown:
        print("nothing to ask: no batch written")
        return None
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    number = len(batch_files()) + 1
    stem = BATCH_DIR / f"batch_{number:03d}"
    stem.with_suffix(".txt").write_text(
        prompt.format(n=len(shown), items="\n".join(shown)), encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps({"batch": number, **meta}, indent=1),
                                         encoding="utf-8")
    print(f"prompt: {stem.with_suffix('.txt')}\n"
          f"answers: {stem}_<model>.txt (for example {stem.name}_claude.txt)")
    return stem.name


def item_text(record: dict, lines: list[str], idx: list[int], n: int) -> str:
    return f'{n}. "{record["title"]}" by {record["artist"]}: ' + SEP.join(lines[i] for i in idx)


def build(n_items: int) -> None:
    detector = ExplicitDetector()
    records = read_jsonl(paths.GENIUS_SECTIONS_JSONL)
    by_id = {r["id"]: r for r in records}
    items = []
    for record, seen, songs, song in new_sections(by_id, records):
        flagged = {i for i, l in enumerate(record["lines"]) if detector.find(l)}
        lines = pick(record, flagged, seen) if flagged else None
        if not lines:
            continue
        seen |= {norm(record["lines"][i]) for i in lines}
        songs.add(song)
        items.append({"n": len(items) + 1, "id": record["id"], "lines": lines})
        if len(items) == n_items:
            break
    shown = [item_text(by_id[it["id"]], by_id[it["id"]]["lines"], it["lines"], it["n"])
             for it in items]
    write_batch(PROMPT, {"prompt_version": VERSION, "items": items}, shown)
    stanzas = sum(len(it["lines"]) > 1 for it in items)
    print(f"{len(items)} items ({stanzas} stanzas, {len(items) - stanzas} single lines)")


def parse(answer: str) -> dict[int, str]:
    """{item number: answer text} of a pasted numbered answer."""
    got = {}
    for row in answer.splitlines():
        m = ANSWER.match(row)
        if m and m.group(2).strip():
            got.setdefault(int(m.group(1)), m.group(2).strip())
    return got


def answered(meta: dict, model: str) -> tuple[dict[int, list[str]], list[int]] | None:
    """{item number: new lines} of the valid answers of `model` to one batch, and the item
    numbers with a missing or wrong answer; None without an answer file."""
    path = BATCH_DIR / f"{meta['stem']}_{model}.txt"
    if not path.exists():
        return None
    got = parse(path.read_text(encoding="utf-8"))
    good, bad = {}, []
    for it in meta["items"]:
        text = got.get(it["n"], "")
        parts = [text] if len(it["lines"]) == 1 else re.split(r"\s+/\s+", text)
        new = [clean_response_line(t.strip().strip('"')) for t in parts]
        if not text or len(new) != len(it["lines"]) or not all(new):
            bad.append(it["n"])
        else:
            good[it["n"]] = new
    return good, bad


def window_state(model: str, by_id: dict) -> dict:
    """Per version 2 section: its plan, the windows issued and answered, and its lines with
    every answered window of `model` applied in batch order."""
    state = {}
    for meta in manifests():
        if meta.get("prompt_version") != VERSION_2:
            continue
        for sid, plan in meta.get("plans", {}).items():
            state[sid] = {"plan": plan, "issued": set(), "answered": set(),
                          "lines": list(by_id[sid]["lines"]), "changed": set()}
        got = answered(meta, model)
        for it in meta["items"]:
            s = state[it["id"]]
            s["issued"].add(it["window"])
            if got and it["n"] in got[0]:
                s["answered"].add(it["window"])
                for i, t in zip(it["lines"], got[0][it["n"]]):
                    if t != s["lines"][i]:
                        s["lines"][i] = t
                        s["changed"].add(i)
    return state


def build_windows(n_new: int, model: str) -> None:
    detector = ExplicitDetector()
    records = read_jsonl(paths.GENIUS_SECTIONS_JSONL)
    by_id = {r["id"]: r for r in records}
    items, plans, waiting = [], {}, 0

    def add(record, lines, k, w):
        idx = w["lines"]
        items.append({"n": len(items) + 1, "id": record["id"], "window": k, "lines": idx,
                      "round": w["round"]})
        return item_text(record, lines, idx, len(items))

    shown = []
    for sid, s in window_state(model, by_id).items():  # next rounds of started sections
        plan = s["plan"]
        for k, w in enumerate(plan):
            if k in s["issued"]:
                continue
            earlier = [j for j, v in enumerate(plan)
                       if v["round"] < w["round"] and set(v["lines"]) & set(w["lines"])]
            if not all(j in s["answered"] for j in earlier):
                waiting += 1
                continue
            if not any(detector.find(s["lines"][i]) for i in w["lines"]):
                continue  # an earlier window already cleaned it: nothing left to do
            shown.append(add(by_id[sid], s["lines"], k, w))
    later = len(items)
    started = 0
    for record, seen, songs, song in new_sections(by_id, records):
        if started == n_new:
            break
        lines = record["lines"]
        if any(SEP.strip() in l for l in lines):
            continue
        flagged = {i for i, l in enumerate(lines) if detector.find(l)}
        if not flagged or any(norm(lines[i]) in seen for i in flagged):
            continue
        plan = windows(flagged, len(lines))
        plans[record["id"]] = plan
        seen |= {norm(l) for l in lines}
        songs.add(song)
        started += 1
        for k, w in enumerate(plan):
            if w["round"] == 1:
                shown.append(add(record, lines, k, w))
    write_batch(PROMPT_2, {"prompt_version": VERSION_2, "model": model, "items": items,
                           "plans": plans}, shown)
    print(f"{len(items)} windows: {later} later rounds of started sections, "
          f"{len(items) - later} from {started} new sections; {waiting} windows wait for "
          f"answers of {model} to earlier rounds")


def export() -> None:
    by_id = {r["id"]: r for r in read_jsonl(paths.GENIUS_SECTIONS_JSONL)}
    detector = ExplicitDetector()
    runs: dict[str, list[dict]] = {}
    window_models = set()
    for meta in manifests():
        for answer in sorted(BATCH_DIR.glob(f"{meta['stem']}_*.txt")):
            model = answer.stem[len(meta["stem"]) + 1:]
            good, bad = answered(meta, model)
            print(f"{answer.name}: {len(good)}/{len(meta['items'])} items"
                  + (f", missing or wrong line count: {bad}" if bad else ""))
            if meta.get("prompt_version") == VERSION_2:
                window_models.add(model)
                continue
            for it in meta["items"]:
                if it["n"] not in good:
                    continue
                lines = list(by_id[it["id"]]["lines"])
                for i, t in zip(it["lines"], good[it["n"]]):
                    lines[i] = t
                runs.setdefault(f"commercial_batch_{model}", []).append({
                    "id": it["id"], "model": model, "prompt_version": meta["prompt_version"],
                    "style": "commercial-batch", "batch": meta["batch"], "item": it["n"],
                    "targets": it["lines"], "lines": lines, "valid": True, "passed": True,
                    "units": []})
    for model in sorted(window_models):
        for sid, s in window_state(model, by_id).items():
            if not s["answered"]:
                continue
            original = by_id[sid]["lines"]
            targets = s["changed"] | {i for i, l in enumerate(original) if detector.find(l)}
            runs.setdefault(f"commercial_window_{model}", []).append({
                "id": sid, "model": model, "prompt_version": VERSION_2,
                "style": "commercial-window", "windows": len(s["plan"]),
                "windows_answered": len(s["answered"]), "targets": sorted(targets),
                "lines": s["lines"], "valid": True, "passed": True, "units": []})
    paths.GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    for name, gens in runs.items():
        out = paths.GENERATIONS_DIR / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for g in gens:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"{name}: {len(gens)} records -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch prompts of new explicit lines.")
    parser.add_argument("--items", type=int, default=50,
                        help="items (version 1) or new sections (version 2)")
    parser.add_argument("--version", type=int, choices=(1, 2), default=1)
    parser.add_argument("--model", default="claude",
                        help="version 2: whose answers earlier rounds are applied from")
    parser.add_argument("--export", action="store_true",
                        help="read the pasted answers into generation files for pairs / vet")
    args = parser.parse_args()
    if args.export:
        export()
    elif args.version == 2:
        build_windows(args.items, args.model)
    else:
        build(args.items)


if __name__ == "__main__":
    main()
