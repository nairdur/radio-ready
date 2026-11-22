"""Synthetic pairs from the end-rhyme thesaurus: each explicit word swapped for a clean word
that rhymes with it, then the line reworded around it by Qwen2.5-7B so it makes sense.

1. Swap. For every line with a detected explicit word (radio_ready.prompts.target_lines), every
   explicit token is masked in turn (the others stay as context) and
   radio_ready.rhymes.suggest_for_word proposes --k varied clean words for it: words that
   rhyme with the explicit word itself (perfect, slant or assonance), with its syllable count
   (within one), never a sound-alike of it or of its last sounds (dam for goddamn), scored by
   masked-LM fit (distilroberta-base, with the line before as context) and meaning, and picked
   to differ from each other in meaning and sound. Punctuation around the word and a leading
   capital are kept. Every explicit word of the line is swapped; the words of all tokens are
   combined (up to MAX_COMBINATIONS lines). The swaps are ranked without a model (filters
   below, then similarity + 0.05 x slot fit - fluency lost; the slot fit keeps the part of
   speech) and --drafts of them go on, the best first and then those using words the others
   do not (varied_drafts).
2. Reword (always). Qwen2.5-7B-Instruct (4-bit, radio_ready.generate.load_model; about 5.5 GB
   of GPU memory) sees the original line, the swapped line and the lines around it as plain
   text (no syllable counts, rhyme letters or other markup) and rewrites the words around the
   new word so the line makes sense and keeps its meaning, keeping the new word, the end rhyme
   and the flow. Greedy, one answer per draft. An answer is kept when it still has the new
   word(s), keeps the end rhyme (rhymes with the original line, radio_ready.generate.rhymes_with,
   or ends on the swapped line's end word, which rhymes with the explicit word it replaced),
   stays within MAX_SYLLABLE_DIFF syllables of it and is one line.
3. Pick. The kept answers are scored like radio_ready.generate (Selector: no explicit word,
   toxicity < 0.9, all-distilroberta-v1 cosine to the original >= 0.70, CoLA fluency relative
   to the original); best: usable (passes, loses at most MAX_FLUENCY_DROP of CoLA
   acceptability), then similarity - fluency lost. A line with no kept answer stays as it was
   and needs an edit in vetting; the bare swaps are never used as rewrites.

Output: outputs/generations/rhyme_<input stem>_<version>.jsonl in the radio_ready.generate
format (id, model, prompt_version, style, targets, raw, rewrites, lines, valid, passed, units
with every draft, answer and score). valid: at least one line rewritten; passed: every target
line got a usable rewrite. radio_ready.pairs lists it as source rhyme_<input>_<version>,
needing approval. Sections already in the output are skipped, so an interrupted run resumes.

Run:  python -m radio_ready.rhyme_swap [--input FILE] [--limit N] [--k 6] [--drafts 4]
"""

import argparse
import itertools
import json
import re
import time
from pathlib import Path

from radio_ready import paths
from radio_ready.commercial import reserved_ids
from radio_ready.explicit import ExplicitDetector
from radio_ready.generate import Selector, complete, load_model, rhymes_with
from radio_ready.prompts import target_lines
from radio_ready.rhymes import slot_fit, suggest_for_word
from radio_ready.text import end_word, syllables

VERSION = "rhyme-v2"
MAX_FLUENCY_DROP = 0.3  # as radio_ready.pairs.MAX_FLUENCY_DROP
REWORD_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TOKEN = re.compile(r"^([^A-Za-z']*)(.*?)([^A-Za-z']*)$")
MAX_COMBINATIONS = 64
FIT_WEIGHT = 0.05  # per nat of masked-LM log-probability of the new word(s) in the slot
MAX_SYLLABLE_DIFF = 2  # reworded line vs original line
CONTEXT = 2  # lines above and below shown to the model

REWORD_PROMPT = """\
These are lines from a song:

{context}

The line "{original}" has an explicit word in it. For a clean radio edit, the explicit word \
was replaced with {quoted}, which rhymes with it:

{draft}

Rewrite this line so it makes sense and says what the original line meant, changing the \
words around {quoted} as much as needed. Keep {quoted}. Keep the end of the line rhyming \
the way it does now, keep about the same number of syllables and the same rhythm so it can \
be sung in its place, and keep the artist's slang and grammar. Use no explicit words.
Answer with the rewritten line only."""


def slots(line: str, detector) -> list[dict]:
    """Explicit tokens of a line: token index, the word to rhyme with, punctuation to keep."""
    found = []
    for i, token in enumerate(line.split(" ")):
        lead, core, trail = TOKEN.match(token).groups()
        hits = detector.find(core) if core else []
        if hits:
            found.append({"index": i, "word": core.lower(), "lead": lead, "trail": trail,
                          "capital": core[:1].isupper()})
    return found


def suggestions(text: str, s: dict, embed, k: int) -> list[dict]:
    """Varied rhymes for one explicit token; the other explicit words stay as context."""
    return suggest_for_word(template(text, s), s["word"], text, embed, k=k,
                            before=s.get("before", ""))


def fill(line: str, slot: dict, word: str) -> str:
    tokens = line.split(" ")
    word = word.capitalize() if slot["capital"] else word
    tokens[slot["index"]] = f"{slot['lead']}{word}{slot['trail']}"
    return " ".join(tokens)


def template(line: str, slot: dict) -> str:
    tokens = line.split(" ")
    tokens[slot["index"]] = f"{slot['lead']}<mask>{slot['trail']}"
    return " ".join(tokens)


def substitutions(line: str, before: str, detector, embed, k: int) -> list[dict]:
    """Candidate lines with every explicit token swapped: {line, words}. Each token gets its
    own varied rhymes (the other explicit words stay as context), then every combination is
    made; with many tokens fewer words per token keep it within MAX_COMBINATIONS, and past
    that the i-th words of all tokens go together."""
    found = slots(line, detector)
    if not found:
        return []
    per = [suggestions(line, {**s, "before": before}, embed, k) for s in found]
    if not all(per):
        return []
    n = max(1, min(k, int(MAX_COMBINATIONS ** (1 / len(found)))))
    if n ** len(found) <= MAX_COMBINATIONS and n >= 2 or len(found) == 1:
        combos = itertools.product(*(p[:n] for p in per))
    else:
        combos = (tuple(p[j % len(p)] for p in per) for j in range(k))
    out = []
    for combo in combos:
        text = line
        for s, c in zip(found, combo):
            text = fill(text, s, c["word"])
        out.append({"line": text, "words": [
            {"explicit": s["word"], **{key: c[key] for key in
                                       ("word", "kind", "fit", "similarity", "score")}}
            for s, c in zip(found, combo)]})
    return out


def varied_drafts(swaps: list[dict], n: int) -> list[dict]:
    """n swaps for the model, best first, each using as few words the others already use as
    possible (so the drafts differ, not just the best word with its runners-up)."""
    ranked = sorted((c for c in swaps if c["lines"] is not None), key=rank)
    picked, used = [], set()
    while ranked and len(picked) < n:
        best = min(range(len(ranked)), key=lambda j: (
            len({w["word"] for w in ranked[j]["words"]} & used), j))
        c = ranked.pop(best)
        picked.append(c)
        used |= {w["word"] for w in c["words"]}
    return picked


def reword(llm, lines: list[str], i: int, original: str, draft: str, words: list[str]) -> str:
    context = "\n".join(lines[max(0, i - CONTEXT): i] + [original]
                         + lines[i + 1: i + 1 + CONTEXT])
    quoted = " and ".join(f'"{w}"' for w in words)
    prompt = REWORD_PROMPT.format(context=context, original=original, quoted=quoted,
                                  draft=draft)
    return complete(*llm, [{"role": "user", "content": prompt}], 1, 64, 0.0, 1.0, False)[0]


def check(raw: str, original: str, draft: str, words: list[str]) -> tuple[str | None, str]:
    """(the answer line or None, why it was rejected)."""
    found = [l.strip().strip('"').strip() for l in raw.strip().splitlines() if l.strip()]
    if len(found) != 1:
        return None, "not one line"
    text = found[0]
    if not all(re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE) for w in words):
        return None, "new word dropped"
    if not (rhymes_with(original, text) or end_word(text) == end_word(draft)):
        return None, "end rhyme lost"
    if abs(syllables(text) - syllables(original)) > MAX_SYLLABLE_DIFF:
        return None, "syllables"
    return text, ""


def rank(c: dict) -> tuple:
    """Lower is better: usable, passing the filters, then meaning + slot fit - fluency lost.
    The slot fit (distilroberta-base) keeps the part of speech of the slot: similarity alone
    prefers words close to the explicit word in meaning, even as the wrong kind of word
    ("money ain't exist"). Reworded lines (no slot fit) use meaning - fluency lost."""
    if c["lines"] is None:
        return True, True, 0.0
    fit = [w["fit"] for w in c["words"]] if c.get("how") == "swap" else [0.0]
    joint = c["similarity"] + FIT_WEIGHT * sum(fit) / len(fit) - c["fluency_drop"]
    return not c.get("usable"), not c["passed"], -joint


def choose(selector, original: str, cands: list[dict]) -> tuple[int | None, list[dict]]:
    """Score candidates {raw, lines, ...}; (index of the best or None, scored)."""
    _, scored = selector.pick([original], [(c["raw"], c["lines"]) for c in cands])
    for c, extra in zip(scored, cands):
        c.update({k: v for k, v in extra.items() if k not in ("raw", "lines")})
        c["usable"] = bool(c["passed"] and c.get("fluency_drop", 1) <= MAX_FLUENCY_DROP)
    ok = [j for j, c in enumerate(scored) if c["lines"] is not None]
    return (min(ok, key=lambda j: rank(scored[j])) if ok else None), scored


def swap_section(record: dict, targets: dict, detector, selector, embed, llm, args) -> dict:
    lines = list(record["lines"])
    units, raws, any_rw, all_ok = [], [], False, True
    for i in sorted(targets):
        original = record["lines"][i]
        subs = substitutions(original, lines[i - 1] if i else "", detector, embed, args.k)
        _, swaps = choose(selector, original, [
            {"raw": s["line"], "lines": [s["line"]], "how": "swap", "words": s["words"]}
            for s in subs])
        answers = []
        for d in varied_drafts(swaps, args.drafts):
            words = [w["word"] for w in d["words"]]
            raw = reword(llm, lines, i, original, d["lines"][0], words)
            text, why = check(raw, original, d["lines"][0], words)
            answers.append({"raw": raw, "lines": [text] if text else None, "how": "reword",
                            "draft": d["lines"][0], "words": d["words"], "rejected": why})
        best, scored = choose(selector, original, answers)
        units.append({"start": i, "end": i + 1, "chosen": best, "drafts": swaps,
                      "distinct": len({c["raw"] for c in scored}), "candidates": scored})
        if best is None:
            all_ok = False
            continue
        lines[i] = scored[best]["lines"][0]
        raws.append(scored[best]["raw"])
        any_rw = True
        all_ok &= scored[best]["usable"]
    rewrites = {str(i): l for i, (l, o) in enumerate(zip(lines, record["lines"])) if l != o}
    return {"raw": "\n".join(raws), "rewrites": rewrites, "lines": lines if any_rw else None,
            "valid": any_rw, "passed": any_rw and all_ok, "units": units}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rhyme-swap pairs from the rhyme thesaurus.")
    parser.add_argument("--input", default=str(paths.KIDZBOP_DIR / "pilot.jsonl"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--k", type=int, default=6, help="varied rhymes per explicit word")
    parser.add_argument("--drafts", type=int, default=4,
                        help="best swaps per line that the model rewords")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    version = VERSION
    args.name = args.name or f"rhyme_{Path(args.input).stem}_{version}"
    records = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    skip = reserved_ids()
    out = paths.GENERATIONS_DIR / f"{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [r for r in records if r["id"] not in done and r["id"] not in skip][: args.limit]
    print(f"sections: {len(records)}, done: {len(done)}, to swap: {len(todo)}", flush=True)

    detector = ExplicitDetector()
    selector = Selector(detector)
    embed = lambda t: selector.similarity.model.encode(t, normalize_embeddings=True)  # noqa: E731
    slot_fit()
    llm = load_model(REWORD_MODEL, True, False)
    model = f"distilroberta-base+{REWORD_MODEL}"
    lines_total = lines_sub = lines_usable = 0
    with open(out, "a", encoding="utf-8") as f:
        for n, record in enumerate(todo, start=1):
            start = time.time()
            targets = target_lines(record, detector)
            if not targets:
                print(f"[{n}/{len(todo)}] {record['id']} skipped: no line has a detected word")
                continue
            result = swap_section(record, targets, detector, selector, embed, llm, args)
            for u in result["units"]:
                lines_total += 1
                if u["chosen"] is not None:
                    lines_sub += 1
                    lines_usable += u["candidates"][u["chosen"]]["usable"]
            f.write(json.dumps({
                "id": record["id"], "model": model, "quantization": "nf4",
                "thinking": False, "temperature": 0.0, "top_p": None, "samples": args.k,
                "prompt_version": version, "style": "rhyme-swap",
                "targets": sorted(targets), **result,
                "seconds": round(time.time() - start, 1)}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(todo)}] {record['id']} valid={result['valid']} "
                  f"passed={result['passed']} {time.time() - start:.1f}s", flush=True)
    print(f"target lines: {lines_total}, reworded: {lines_sub}, usable "
          f"(filters + fluency drop <= {MAX_FLUENCY_DROP}): {lines_usable}; output: {out}")


if __name__ == "__main__":
    main()
