"""Synthetic radio edits: a 7B meaning draft for every explicit line, 14B for the hard ones.

Every line with a detected explicit word is classified:
  easy  the explicit words are inside the line (not the end word) and there is one distinct
        explicit word; the 7B draft is kept if it passes the checks below
  hard  the end (rhyme) word is explicit, or the line has two or more distinct explicit words,
        or the 7B draft of an easy line fails a check

Phase 1, Qwen2.5-7B-Instruct, greedy, every explicit line: the base instruction (replace only
the explicit words with clean words that mean the same thing in this line and stanza; no rhyme
or syllable hints). The draft is the meaning anchor. Checks: no detected explicit word,
toxicity < 0.9, similarity to the original >= 0.70, within one syllable of the original.
One-for-one word swaps away from the explicit words (more than one word from them) are put
back, so dialect and grammar stay as written ("this happen" is not corrected to "happens").

Phase 2, Qwen3-14B, hard lines only: the model sees the stanza with every explicit line
replaced by its draft (it never sees explicit text), the draft line marked, the original's
syllable count and the end words of the line's rhyme partners, and rewrites the line so it
works in the verse. --samples candidates (temperature 1.0, top-p 0.9) plus the draft are
scored: meaning against the draft (similarity >= 0.70 to pass), sound against the explicit
original (syllables within one, an end-rhyme key shared with the original line or its rhyme
partners), no explicit word, toxicity < 0.9, fluency relative to the draft. Ranked: passed,
rhymes, syllables within one, then similarity x fluency. If no sample rhymes, the draft stays.

The two models do not fit on an 8 GB GPU together, so phase 1 runs for all sections first
(drafts are saved), then the 7B model is unloaded and phase 2 runs.

Output: outputs/generations/<name>.jsonl in the radio_ready.generate format (lines, valid,
passed, units); every unit records the line class, the reason, the draft and all candidates
with their scores. radio_ready.pairs and radio_ready.compare read it unchanged.

Run:  python -m radio_ready.rewrite [--input FILE] [--limit N] [--samples 5]
"""

import argparse
import difflib
import gc
import json
import re
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from radio_ready import paths
from radio_ready.commercial import reserved_ids
from radio_ready.explicit import ExplicitDetector
from radio_ready.prompts import PLAIN_SYSTEM, clean_response_line, target_lines
from radio_ready.scoring import (
    SIMILARITY_THRESHOLD, TOXICITY_THRESHOLD, Fluency, Similarity, Toxicity,
)
from radio_ready.text import _phones, line_rhyme_keys, rhyme_words, syllables

VERSION = "solid-v1"
DRAFT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
FIX_MODEL = "Qwen/Qwen3-14B"

DRAFT_PROMPT = """\
Lines from "{title}" by {artist}:

{stanza}

The line marked >>> has words that radio would censor: {words}.
Replace only those words with clean words that mean the same thing here, so the line and the \
stanza keep their meaning. Keep every other word exactly as it is: do not correct slang, \
dialect, grammar or spelling (keep "they was", "it don't", "this happen", "runnin'", "gon'").

Answer with the rewritten line only, without quotes or anything else."""

FIX_PROMPT = """\
Lines from "{title}" by {artist}:

{stanza}

The line marked >>> was edited for radio and now sounds flat next to the others. Rewrite it so \
it works in this verse:
- keep what it says, in the voice, slang and dialect of the verse;
- {syllables} syllables, with the same rhythm as the lines around it;
- {rhyme}
- no word radio would censor, and no sound-alike of one.

Answer with the rewritten line only, without quotes or anything else."""


# --- helpers --------------------------------------------------------------------------------

def stanza_window(lines: list[str], i: int) -> str:
    start = (i // 4) * 4
    return "\n".join((">>> " if j == i else "    ") + lines[j]
                     for j in range(start, min(len(lines), start + 4)))


def parse_line(text: str, stanza_lines: list[str], i: int) -> str | None:
    """First answer row; None if empty or a copy of another line of the stanza."""
    rows = [clean_response_line(r.strip().strip("`\"'")) for r in text.strip().splitlines()]
    rows = [r.lstrip(">").strip().strip("\"'") for r in rows if r]
    if not rows:
        return None
    line = rows[0]
    others = {l.lower().strip() for j, l in enumerate(stanza_lines) if j != i}
    return None if line.lower().strip() in others else line


TOKEN = re.compile(r"\S+")


def restore_untouched(original: str, draft: str, detector) -> str:
    """The draft with one-for-one word swaps away from the explicit words (more than one word
    from them) put back as in the original: grammar or spelling corrections and needless
    synonyms ("This stuff always happens to me" -> "... always happen to me"). Insertions,
    deletions and rewordings of several words are kept.
    """
    a, b = TOKEN.findall(original), TOKEN.findall(draft)
    explicit = {k for k, w in enumerate(a) if detector.find(w)}
    near = explicit | {k + d for k in explicit for d in (-1, 1)}
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, [w.lower() for w in a], [w.lower() for w in b], autojunk=False).get_opcodes():
        swap = op == "replace" and i2 - i1 == j2 - j1
        if swap and not any(k in near for k in range(i1, i2)):
            out += a[i1:i2]  # a word swapped away from the explicit words: keep the original
        else:
            out += b[j1:j2]
    return " ".join(out)


def rhyme_partners(record: dict, i: int) -> list[str]:
    scheme = record["rhyme_scheme"]
    return [l for j, l in enumerate(record["lines"]) if j != i and scheme[j] == scheme[i]]


def classify(line: str, words: list[str], detector) -> tuple[str, str]:
    found = rhyme_words(line)
    if found and detector.find(found[-1]):
        return "hard", "explicit end word"
    if len(set(words)) > 1:
        return "hard", "several explicit words"
    return "easy", "explicit word inside the line"


def rhyme_instruction(record: dict, i: int, detector) -> str:
    ends = []
    for line in [record["lines"][i]] + rhyme_partners(record, i):
        found = rhyme_words(line)
        if found and not detector.find(found[-1]) and found[-1] not in ends:
            ends.append(found[-1])
    if not ends:
        return "end on a word that rhymes the way the line did before;"
    return "end on a word that rhymes with " + ", ".join(f'"{e}"' for e in ends[:4]) + ";"


def rhymes(candidate: str, original: str, partners: list[str]) -> bool:
    """Shares an end-rhyme key with the original line or one of its rhyme partners."""
    targets = set(line_rhyme_keys(original))
    for p in partners:
        targets |= set(line_rhyme_keys(p))
    return bool(set(line_rhyme_keys(candidate)) & targets)


class Scorer:
    def __init__(self, detector):
        self.detector = detector
        self.similarity = Similarity(on="cpu")
        self.toxicity = Toxicity(on="cpu")
        self.fluency = Fluency(on="cpu")

    def draft_ok(self, draft: str, original: str) -> tuple[bool, dict]:
        sim = float(self.similarity.pairwise([draft], [original])[0])
        tox = float(self.toxicity.score([draft])[0])
        info = {"similarity": round(sim, 3), "toxicity": round(tox, 3),
                "explicit": sorted(set(self.detector.find(draft))),
                "syllable_diff": abs(syllables(draft) - syllables(original))}
        ok = (not info["explicit"] and tox < TOXICITY_THRESHOLD and sim >= SIMILARITY_THRESHOLD
              and info["syllable_diff"] <= 1)
        return ok, info

    def score(self, candidates: list[str], draft: str, original: str,
              partners: list[str]) -> list[dict]:
        sims = self.similarity.pairwise(candidates, [draft] * len(candidates))
        toxs = self.toxicity.score(candidates)
        flu = self.fluency.score([draft] + candidates)
        out = []
        for k, c in enumerate(candidates):
            explicit = sorted(set(self.detector.find(c)))
            out.append({
                "line": c, "similarity_to_draft": round(float(sims[k]), 3),
                "toxicity": round(float(toxs[k]), 3), "explicit": explicit,
                "syllable_diff": abs(syllables(c) - syllables(original)),
                "rhymes": rhymes(c, original, partners),
                "fluency_drop": round(max(float(flu[0] - flu[k + 1]), 0.0), 3),
                "passed": bool(not explicit and toxs[k] < TOXICITY_THRESHOLD
                               and sims[k] >= SIMILARITY_THRESHOLD)})
        return out


def rank(c: dict) -> tuple:
    return (not c["passed"], not c["rhymes"], c["syllable_diff"] > 1,
            -(c["similarity_to_draft"] * (1 - c["fluency_drop"])))


def load(name: str, cpu_embeddings: bool):
    from radio_ready.generate import load_model

    return load_model(name, True, cpu_embeddings)


def unload(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


# --- phases ---------------------------------------------------------------------------------

def phase_drafts(records, detector, scorer, args) -> dict[str, dict[int, dict]]:
    from radio_ready.generate import complete

    tok, model = load(args.draft_model, False)
    drafts: dict[str, dict[int, dict]] = {}
    for n, record in enumerate(records, start=1):
        targets = target_lines(record, detector)
        lines, out = list(record["lines"]), {}
        for i in sorted(targets):
            user = DRAFT_PROMPT.format(title=record["title"], artist=record["artist"],
                                       stanza=stanza_window(lines, i),
                                       words=", ".join(targets[i]))
            raw = complete(tok, model, [{"role": "system", "content": PLAIN_SYSTEM},
                                        {"role": "user", "content": user}],
                           1, 128, 0.0, 1.0, False)[0]
            start = (i // 4) * 4
            draft = parse_line(raw, lines[start:start + 4], i - start)
            if draft:
                draft = restore_untouched(record["lines"][i], draft, detector)
            cls, reason = classify(record["lines"][i], targets[i], detector)
            ok, info = scorer.draft_ok(draft, record["lines"][i]) if draft else (False, {})
            if cls == "easy" and not ok:
                cls, reason = "hard", "draft failed a check"
            out[i] = {"draft": draft, "raw": raw, "class": cls, "reason": reason,
                      "draft_ok": ok, "draft_scores": info}
            if draft:
                lines[i] = draft  # later lines see earlier drafts, never explicit text
        drafts[record["id"]] = out
        print(f"[draft {n}/{len(records)}] {record['id']} "
              f"{sum(d['class'] == 'hard' for d in out.values())}/{len(out)} hard", flush=True)
    unload(model, tok)
    return drafts


def phase_fix(records, drafts, detector, scorer, args) -> dict[str, dict[int, dict]]:
    from radio_ready.generate import complete

    hard = [(r, i) for r in records for i, d in drafts[r["id"]].items()
            if d["class"] == "hard" and d["draft"]]
    fixed: dict[str, dict[int, dict]] = {r["id"]: {} for r in records}
    if not hard:
        return fixed
    tok, model = load(args.fix_model, args.cpu_embeddings)
    for n, (record, i) in enumerate(hard, start=1):
        lines = [drafts[record["id"]].get(j, {}).get("draft") or l
                 for j, l in enumerate(record["lines"])]
        original, draft = record["lines"][i], drafts[record["id"]][i]["draft"]
        user = FIX_PROMPT.format(title=record["title"], artist=record["artist"],
                                 stanza=stanza_window(lines, i), syllables=syllables(original),
                                 rhyme=rhyme_instruction(record, i, detector))
        raws = complete(tok, model, [{"role": "system", "content": PLAIN_SYSTEM},
                                     {"role": "user", "content": user}],
                        args.samples, 128, 1.0, 0.9, False)
        start = (i // 4) * 4
        sung = [s for s in (parse_line(r, lines[start:start + 4], i - start) for r in raws) if s]
        scored = scorer.score([draft] + sung, draft, original, rhyme_partners(record, i))
        for k, c in enumerate(scored):
            c["source"] = "draft" if k == 0 else "14b"
        best = min(range(len(scored)), key=lambda k: rank(scored[k]))
        fixed[record["id"]][i] = {"chosen": best, "candidates": scored}
        print(f"[fix {n}/{len(hard)}] {record['id']} line {i + 1}: "
              f"{scored[best]['source']} rhymes={scored[best]['rhymes']}", flush=True)
    unload(model, tok)
    return fixed


def main() -> None:
    parser = argparse.ArgumentParser(description="7B drafts, 14B rewrites of hard lines.")
    parser.add_argument("--input", default=str(paths.KIDZBOP_DIR / "pilot.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--draft-model", default=DRAFT_MODEL)
    parser.add_argument("--fix-model", default=FIX_MODEL)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--cpu-embeddings", action="store_true", default=True,
                        help="14B on 8 GB: embeddings and output head on the CPU")
    args = parser.parse_args()

    detector = ExplicitDetector()
    skip = reserved_ids()
    records = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    records = [r for r in records if r["id"] not in skip and target_lines(r, detector)]
    records = records[: args.limit]
    scorer = Scorer(detector)
    started = time.time()
    drafts = phase_drafts(records, detector, scorer, args)
    fixed = phase_fix(records, drafts, detector, scorer, args)

    name = args.name or f"solid_{Path(args.input).stem}_{VERSION}"
    out = paths.GENERATIONS_DIR / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for record in records:
            lines, units, ok, passed = list(record["lines"]), [], True, True
            for i, d in sorted(drafts[record["id"]].items()):
                unit = {"start": i, "end": i + 1, "class": d["class"], "reason": d["reason"],
                        "draft": d["draft"], "draft_ok": d["draft_ok"],
                        "draft_scores": d["draft_scores"]}
                if i in fixed[record["id"]]:
                    fx = fixed[record["id"]][i]
                    unit.update(fx)
                    best = fx["candidates"][fx["chosen"]]
                    lines[i] = best["line"]
                    passed &= best["passed"] and best["rhymes"]
                elif d["draft"]:
                    lines[i] = d["draft"]
                    passed &= d["draft_ok"]
                else:
                    ok = False
                units.append(unit)
            f.write(json.dumps({
                "id": record["id"], "model": f"{args.draft_model} + {args.fix_model}",
                "prompt_version": VERSION, "style": "solid", "samples": args.samples,
                "temperature": 1.0, "targets": sorted(drafts[record["id"]]), "raw": "",
                "rewrites": {str(i): l for i, (l, o) in enumerate(zip(lines, record["lines"]))
                             if l != o},
                "lines": lines if ok else None, "valid": ok, "passed": ok and passed,
                "units": units}, ensure_ascii=False) + "\n")
    print(f"{len(records)} sections in {(time.time() - started) / 60:.1f} min, output: {out}")


if __name__ == "__main__":
    main()
