"""Freer radio edits with Qwen2.5-7B: the model reads the 4-line section and rewrites the
flagged lines as it needs to, keeping rhyme and meter.

One call per 4-line block that has a flagged line (a line with a detected explicit word).
The model sees the whole block, the flagged lines marked with their censored words, each
flagged line's syllable count and the words it rhymes with, plus a few edits you liked
(from annotations/vetting.csv, never from the song being rewritten). It may reword a flagged
line freely; it must drop the censored words, keep what the line says in the verse, its
syllables and its end rhyme, and keep the voice, slang and dialect. Lines without a flagged
word are always kept exactly as written (any change the model makes to them is undone).

Candidates: one greedy answer and --samples sampled ones (temperature 1.0, top-p 0.9), scored
per flagged line. A rewrite of a line is out if it still has a detected explicit word, is
toxic (>= 0.9), is unchanged, or brings in a word that is not in the original line, the CMU
dictionary or clean lyrics (seen 20+ times; slang such as "homies" is fine, "gud" is not).
The rest are ranked: within one syllable, end rhyme kept, then meaning (similarity to the
original line) x fluency (CoLA, relative to the original). Lines with no clean rewrite get
--retries new calls, the block shown with the rewrites already accepted. A line with no
clean rewrite after that gets no proposal: an explicit line is never proposed.

Output: outputs/generations/free_<input>_<VERSION>.jsonl in the radio_ready.generate format;
the vetting app lists it as its own source.

Run:  python -m radio_ready.free [--input FILE] [--limit 10] [--samples 4] [--retries 1]
"""

import argparse
import json
import re
import time
from functools import lru_cache
from pathlib import Path

from radio_ready import cmu, paths
from radio_ready.commercial import reserved_ids
from radio_ready.explicit import ExplicitDetector
from radio_ready.prompts import clean_response_line, target_lines
from radio_ready.rewrite import load, rhyme_instruction, unload
from radio_ready.scoring import TOXICITY_THRESHOLD, Fluency, Similarity, Toxicity
from radio_ready.text import end_rhyme_kept, syllables

VERSION = "free-v1"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_EXAMPLES = 6

SYSTEM = ("You are a professional lyricist who writes radio edits of rap and pop songs. You "
          "rewrite the lines radio would censor so they sound like the artist wrote them clean, "
          "and you leave every other line exactly as it is.")

PROMPT = """\
Four lines from "{title}" by {artist}:

{block}

The lines marked >>> have words radio would censor:
{flagged}

Rewrite the marked lines for the radio edit. Read all four lines first to understand what the \
verse says. You can reword a marked line as much as you need, but:
- every censored word must be gone, with no other censored word, slur or sound-alike in its \
place;
- keep what the line means in this verse, in the artist's voice, slang and dialect (do not \
fix grammar or spelling);
- keep its syllable count and its end rhyme as given above, so it flows like the original;
- use real words only.
Copy the unmarked lines exactly as they are.

Radio edits of other songs, for the style:
{examples}

Answer with the four lines, numbered 1. to 4., and nothing else."""

NUMBERED = re.compile(r"^\s*(?:>>>\s*)?([1-4])[.):]\s*(.*)$")
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def examples(exclude_ids: set[str], k: int = N_EXAMPLES) -> list[tuple[str, str]]:
    """Up to k (original, edit) line pairs you liked, your own edits first, one per song."""
    from radio_ready.pairs import load_sections, load_sheet

    rows = [r for r in load_sheet(load_sections())
            if r.get("verdict") in ("like", "superlike") and r["id"] not in exclude_ids
            and r.get("original")]
    edited = [r for r in rows if (r.get("edit") or "").strip()]
    plain = [r for r in rows if not (r.get("edit") or "").strip() and r.get("rewrite")
             and r.get("verdict") == "superlike"]
    out, seen = [], set()
    for r in edited + plain:
        new = (r.get("edit") or "").strip() or r["rewrite"]
        if r["id"] in seen or new == r["original"]:
            continue
        seen.add(r["id"])
        out.append((r["original"], new))
        if len(out) == k:
            break
    return out


def block_text(lines: list[str], start: int, targets: dict[int, list[str]], record,
               detector) -> tuple[str, str]:
    rows, flagged = [], []
    for j in range(start, min(len(lines), start + 4)):
        n = j - start + 1
        rows.append(f"{'>>> ' if j in targets else '    '}{n}. {lines[j]}")
        if j in targets:
            words = ", ".join(f'"{w}"' for w in targets[j])
            flagged.append(f"- line {n}: {words}; {syllables(record['lines'][j])} syllables; "
                           f"{rhyme_instruction(record, j, detector).rstrip(';')}")
    return "\n".join(rows), "\n".join(flagged)


def parse(text: str, block: list[str]) -> list[str] | None:
    """The four numbered lines of an answer; None if any is missing."""
    got = {}
    for row in text.strip().splitlines():
        m = NUMBERED.match(row)
        if m and int(m.group(1)) not in got:
            got[int(m.group(1))] = clean_response_line(m.group(2).strip().strip("`\"'")).strip()
    if any(not got.get(n) for n in range(1, len(block) + 1)):
        return None
    return [got[n] for n in range(1, len(block) + 1)]


@lru_cache(maxsize=1)
def lyric_words(min_count: int = 20) -> frozenset[str]:
    """Words seen at least min_count times in clean lyrics: slang counts as real words
    (homies, hunnids, winnin), typos and made-up words do not."""
    from radio_ready.rhymes import lyric_counts

    return frozenset(w for w, n in lyric_counts().items() if n >= min_count)


def non_words(new: str, original: str) -> list[str]:
    """Words of the rewrite that are not in the original, the CMU dictionary or clean lyrics."""
    have = {w.lower() for w in WORD.findall(original)}
    return [w for w in WORD.findall(new) if w.lower() not in have
            and not cmu.phones_for_word(w.lower()) and w.lower() not in lyric_words()]


def stem(word: str) -> str:
    return re.sub(r"(?:es|s|e)$", "", word.lower()) or word.lower()


def phones(word: str) -> tuple[str, ...]:
    p = cmu.phones_for_word(word.lower())
    return tuple(re.sub(r"\d", "", x) for x in p[0].split()) if p else ()


def disguised(new: str, original: str, censored: list[str]) -> list[str]:
    """New words that are the censored word in another spelling or a homophone of it, which
    the detector can miss (hoe -> ho, hos)."""
    have = {w.lower() for w in WORD.findall(original)}
    out = []
    for w in WORD.findall(new):
        if w.lower() in have:
            continue
        for e in censored:
            same_sound = phones(w) and phones(w) in (phones(e), phones(stem(e)))
            if stem(w) == stem(e) or same_sound:
                out.append(w)
                break
    return out


class Scorer:
    def __init__(self, detector):
        self.detector = detector
        self.similarity = Similarity(on="cpu")
        self.toxicity = Toxicity(on="cpu")
        self.fluency = Fluency(on="cpu")

    def score(self, new: list[str], old: list[str]) -> list[dict]:
        """One score dict per (rewrite, original line)."""
        sims = self.similarity.pairwise(new, old)
        toxs = self.toxicity.score(new)
        flu = self.fluency.score(old + new)
        out = []
        for n in range(len(new)):
            censored = self.detector.find(old[n])
            c = {"line": new[n], "explicit": sorted(set(self.detector.find(new[n]))
                                                    | set(disguised(new[n], old[n], censored))),
                 "toxicity": round(float(toxs[n]), 3), "similarity": round(float(sims[n]), 3),
                 "syllable_diff": abs(syllables(new[n]) - syllables(old[n])),
                 "rhymes": end_rhyme_kept(old[n], new[n]),
                 "fluency_drop": round(max(float(flu[n] - flu[len(old) + n]), 0.0), 3),
                 "non_words": non_words(new[n], old[n]), "unchanged": new[n] == old[n]}
            c["ok"] = bool(not c["explicit"] and c["toxicity"] < TOXICITY_THRESHOLD
                           and not c["non_words"] and not c["unchanged"])
            out.append(c)
        return out


def rank(c: dict) -> tuple:
    return (not c["ok"], c["syllable_diff"] > 1, not c["rhymes"],
            -c["similarity"] * (1 - c["fluency_drop"]))


def ask(tok, model, scorer, record, lines, start, todo, shot_text, samples, attempt):
    """Scored rewrites {line index: [candidates]} of the lines `todo` of one 4-line block,
    shown with the current `lines` (rewrites already accepted are filled in)."""
    from radio_ready.generate import complete

    detector = scorer.detector
    targets = {i: sorted(set(detector.find(record["lines"][i]))) for i in todo}
    shown, notes = block_text(lines, start, targets, record, detector)
    user = PROMPT.format(title=record["title"], artist=record["artist"], block=shown,
                         flagged=notes, examples=shot_text)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    raws = complete(tok, model, msgs, 1, 256, 0.0, 1.0, False)
    if samples:
        raws += complete(tok, model, msgs, samples, 256, 1.0, 0.9, False)
    block = lines[start:start + 4]
    found: dict[int, list[dict]] = {i: [] for i in todo}
    for k, raw in enumerate(raws):
        got = parse(raw, block)
        if not got:
            continue
        scored = scorer.score([got[i - start] for i in todo], [record["lines"][i] for i in todo])
        for i, c in zip(todo, scored):
            c.update(sample="greedy" if k == 0 else f"t1.0#{k}", attempt=attempt)
            found[i].append(c)
    return found, raws


def main() -> None:
    parser = argparse.ArgumentParser(description="Freer 7B radio edits of flagged lines.")
    parser.add_argument("--input", default=str(paths.GENIUS_SECTIONS_JSONL))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--name", default=None)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--retries", type=int, default=1,
                        help="new calls for lines with no clean rewrite yet")
    args = parser.parse_args()

    detector = ExplicitDetector()
    skip = reserved_ids()
    records = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    records = [r for r in records if r["id"] not in skip and target_lines(r, detector)]
    records = records[: args.limit]
    shots = examples({r["id"] for r in records})
    shot_text = "\n".join(f'- "{a}" -> "{b}"' for a, b in shots) or "- (none yet)"
    scorer = Scorer(detector)
    lyric_words()
    tok, model = load(args.model, False)
    started = time.time()

    name = args.name or f"free_{Path(args.input).stem}_{VERSION}"
    out = paths.GENERATIONS_DIR / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for n, record in enumerate(records, start=1):
            targets = target_lines(record, detector)
            lines = list(record["lines"])
            cands: dict[int, list[dict]] = {i: [] for i in targets}
            raws: list[str] = []
            for start in sorted({i // 4 * 4 for i in targets}):
                todo = [i for i in sorted(targets) if start <= i < start + 4]
                for attempt in range(args.retries + 1):
                    found, raw = ask(tok, model, scorer, record, lines, start, todo,
                                     shot_text, args.samples, attempt)
                    raws += raw
                    for i, cs in found.items():
                        cands[i] += cs
                        best = min(cands[i], key=rank, default=None)
                        if best and best["ok"]:
                            lines[i] = best["line"]
                    todo = [i for i in todo if not any(c["ok"] for c in cands[i])]
                    if not todo:
                        break
            units = []
            for i in sorted(targets):
                order = sorted(range(len(cands[i])), key=lambda k: rank(cands[i][k]))
                chosen = order[0] if order and cands[i][order[0]]["ok"] else None
                units.append({"start": i, "end": i + 1, "chosen": chosen,
                              "candidates": cands[i]})
            done = [u for u in units if u["chosen"] is not None]
            passed = len(done) == len(units) and all(
                u["candidates"][u["chosen"]]["syllable_diff"] <= 1
                and u["candidates"][u["chosen"]]["rhymes"] for u in done)
            f.write(json.dumps({
                "id": record["id"], "model": args.model, "prompt_version": VERSION,
                "style": "free", "samples": args.samples, "temperature": 1.0,
                "targets": sorted(targets), "raw": raws,
                "rewrites": {str(i): l for i, (l, o) in enumerate(zip(lines, record["lines"]))
                             if l != o},
                "lines": lines, "valid": bool(done), "passed": passed, "units": units},
                ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(records)}] {record['id']} {len(done)}/{len(units)} lines rewritten",
                  flush=True)
    unload(model, tok)
    print(f"{len(records)} sections in {(time.time() - started) / 60:.1f} min, output: {out}")


if __name__ == "__main__":
    main()
