"""Batch prompts for Claude / Gemini: many new explicit lines in one chat message, one answer
pasted back as a whole.

Items are Genius lines you do not have yet. A section is left out when it is already in the
vetting sheet, a generation run, the commercial table or an earlier batch (old or new), and
so is a line whose text (lower case, letters and digits only) is already in one of those
sections or already in this batch. Each section gives at most one item, and each song at most
one per batch:
  stanza  a 4-line stanza (the blocks of radio_ready.prompts) whose lines are all explicit,
          rewritten as a whole
  line    otherwise one explicit line on its own (the first new one), with no context
Sections are taken in a fixed hash order, so the batches never overlap.

  data/commercial/batches/batch_NNN.txt          the prompt with the items: paste it in a chat
  data/commercial/batches/batch_NNN.json         which section and lines each item is
  data/commercial/batches/batch_NNN_<model>.txt  paste that model's whole answer here

--export reads every answer file into outputs/generations/commercial_batch_<model>.jsonl (the
radio_ready.generate format; targets = the item's lines), listed by the vetting app as its own
source (approval required). An item whose answer is missing or has the wrong number of lines is
left out and reported.

Run:  python -m radio_ready.commercial_batch [--items 50]
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


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", "", text.lower()).split())


def batch_files() -> list:
    return sorted(BATCH_DIR.glob("batch_[0-9][0-9][0-9].json"))


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
    for path in batch_files():
        ids |= {it["id"] for it in json.loads(path.read_text(encoding="utf-8"))["items"]}
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


def build(n_items: int) -> None:
    detector = ExplicitDetector()
    records = read_jsonl(paths.GENIUS_SECTIONS_JSONL)
    by_id = {r["id"]: r for r in records}
    done = done_ids()
    seen = {norm(l) for sid in done if sid in by_id for l in by_id[sid]["lines"]}
    songs, items = set(), []
    for record in sorted(records, key=lambda r: hash_rank(r["id"])):
        song = (record["artist"].lower(), record["title"].lower())
        if record["id"] in done or song in songs:
            continue
        flagged = {i for i, l in enumerate(record["lines"]) if detector.find(l)}
        lines = pick(record, flagged, seen) if flagged else None
        if not lines:
            continue
        seen |= {norm(record["lines"][i]) for i in lines}
        songs.add(song)
        items.append({"n": len(items) + 1, "id": record["id"], "lines": lines})
        if len(items) == n_items:
            break

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    number = len(batch_files()) + 1
    stem = BATCH_DIR / f"batch_{number:03d}"
    shown = []
    for it in items:
        r = by_id[it["id"]]
        text = SEP.join(r["lines"][i] for i in it["lines"])
        shown.append(f'{it["n"]}. "{r["title"]}" by {r["artist"]}: {text}')
    stem.with_suffix(".txt").write_text(
        PROMPT.format(n=len(items), items="\n".join(shown)), encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(
        {"batch": number, "prompt_version": VERSION, "items": items}, indent=1),
        encoding="utf-8")
    stanzas = sum(len(it["lines"]) > 1 for it in items)
    print(f"batch {number}: {len(items)} items ({stanzas} stanzas, "
          f"{len(items) - stanzas} single lines)")
    print(f"prompt: {stem.with_suffix('.txt')}\n"
          f"answers: {stem}_<model>.txt (for example {stem.name}_claude.txt)")


def parse(answer: str) -> dict[int, str]:
    """{item number: answer text} of a pasted numbered answer."""
    got = {}
    for row in answer.splitlines():
        m = ANSWER.match(row)
        if m and m.group(2).strip():
            got.setdefault(int(m.group(1)), m.group(2).strip())
    return got


def export() -> None:
    by_id = {r["id"]: r for r in read_jsonl(paths.GENIUS_SECTIONS_JSONL)}
    runs: dict[str, list[dict]] = {}
    for manifest in batch_files():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        for answer in sorted(BATCH_DIR.glob(f"{manifest.stem}_*.txt")):
            model = answer.stem[len(manifest.stem) + 1:]
            got = parse(answer.read_text(encoding="utf-8"))
            bad = []
            for it in meta["items"]:
                record = by_id[it["id"]]
                text = got.get(it["n"], "")
                parts = [text] if len(it["lines"]) == 1 else re.split(r"\s+/\s+", text)
                new = [clean_response_line(t.strip().strip('"')) for t in parts]
                if not text or len(new) != len(it["lines"]) or not all(new):
                    bad.append(it["n"])
                    continue
                lines = list(record["lines"])
                for i, t in zip(it["lines"], new):
                    lines[i] = t
                runs.setdefault(model, []).append({
                    "id": it["id"], "model": model, "prompt_version": meta["prompt_version"],
                    "style": "commercial-batch", "batch": meta["batch"], "item": it["n"],
                    "targets": it["lines"], "lines": lines, "valid": True, "passed": True,
                    "units": []})
            print(f"{answer.name}: {len(meta['items']) - len(bad)}/{len(meta['items'])} items"
                  + (f", missing or wrong line count: {bad}" if bad else ""))
    paths.GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    for model, gens in runs.items():
        out = paths.GENERATIONS_DIR / f"commercial_batch_{model}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for g in gens:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"{model}: {len(gens)} items -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch prompts of new explicit lines.")
    parser.add_argument("--items", type=int, default=50)
    parser.add_argument("--export", action="store_true",
                        help="read the pasted answers into generation files for pairs / vet")
    args = parser.parse_args()
    if args.export:
        export()
    else:
        build(args.items)


if __name__ == "__main__":
    main()
