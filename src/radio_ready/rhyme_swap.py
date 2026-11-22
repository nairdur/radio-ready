"""Synthetic pairs from the end-rhyme thesaurus: each explicit word swapped for a clean word
that rhymes with it, the rest of the line kept.

For every line with a detected explicit word (radio_ready.prompts.target_lines), each explicit
token is masked in turn and radio_ready.rhymes.suggest_for_word proposes clean words for the
slot: words that rhyme with the explicit word itself, with its syllable count (within one),
never a sound-alike of it, ranked by masked-LM fit (distilroberta-base, with the line before as
context) and meaning. Also never used: words heard as part of the explicit word (damn, dam
for goddamn) and words with no vowel letter (mrs). Leading and trailing punctuation and a leading capital are kept. A line
with two explicit words gets every combination of the top --k words per slot.

The line candidates are scored like radio_ready.generate (Selector: no explicit word left,
toxicity < 0.9, all-distilroberta-v1 cosine to the original >= 0.70, CoLA fluency relative to
the original) and the best is kept: usable and passing first, then similarity + 0.05 x slot
fit (log-probability; keeps the part of speech) - fluency lost. A substitution
counts as usable when its line passes and loses at most MAX_FLUENCY_DROP of CoLA
acceptability (the noise sense filter of radio_ready.pairs).

--cleanup: Qwen2.5-7B-Instruct (4-bit, radio_ready.generate.load_model) also proposes, for the
best --cleanup-k substitutions of a line, the same line with only its grammar or a word or two
around the new word fixed; the new word must stay. These versions compete with the plain
substitutions in the same ranking. Needs about 5.5 GB of GPU memory.

Output: outputs/generations/rhyme_<input stem>_<version>.jsonl in the radio_ready.generate
format (id, model, prompt_version, style, targets, raw, rewrites, lines, valid, passed, units
with every candidate and its scores). valid: at least one line got a substitution (lines with
none stay as they were and still need an edit in vetting); passed: every target line got a
usable one. radio_ready.pairs lists it as source rhyme_<input>_<version>, needing approval.
Sections already in the output are skipped, so an interrupted run resumes.

Run:  python -m radio_ready.rhyme_swap [--input FILE] [--limit N] [--k 4] [--cleanup]
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
from radio_ready.generate import Selector
from radio_ready.prompts import target_lines
from radio_ready.rhymes import slot_fit, suggest_for_word
from radio_ready.text import _phones

VERSION = "rhyme-v1"
MAX_FLUENCY_DROP = 0.3  # as radio_ready.pairs.MAX_FLUENCY_DROP
CLEANUP_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TOKEN = re.compile(r"^([^A-Za-z']*)(.*?)([^A-Za-z']*)$")
MAX_COMBINATIONS = 16
FIT_WEIGHT = 0.05  # per nat of masked-LM log-probability of the new word(s) in the slot

CLEANUP_PROMPT = """\
A song lyric had an explicit word replaced by the clean word "{word}", which rhymes with \
the word it replaced:

{line}

If the line reads badly now, fix only its grammar or change a word or two next to "{word}" \
so it makes sense. Keep "{word}", keep everything else as it is, keep the same number of \
syllables, and add nothing explicit. If it already reads well, return it unchanged.
Answer with the line only."""


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


def part_alike(word: str, explicit: str) -> bool:
    """Heard as part of the explicit word: inside its spelling (damn in goddamn) or at most
    one phoneme from its last sounds (dam for goddamn)."""
    if word in explicit:
        return True
    a, b = _phones(word), _phones(explicit)
    if not a or len(b) <= len(a):
        return False
    strip = lambda p: [x.rstrip("012") for x in p]  # noqa: E731
    return sum(x != y for x, y in zip(strip(a), strip(b[-len(a):]))) <= 1


def suggestions(text: str, s: dict, before: str, embed, k: int) -> list[dict]:
    """suggest_for_word, without part sound-alikes and words with no vowel letter (mrs)."""
    found = suggest_for_word(template(text, s), s["word"], text, embed, k=3 * k, before=before)
    return [c for c in found if re.search("[aeiouy]", c["word"])
            and not part_alike(c["word"], s["word"])][:k]


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
    """Candidate lines with every explicit token swapped: {line, words, suggestions}. Slots
    are filled left to right; each later slot is suggested with the earlier ones in place."""
    found = slots(line, detector)
    if not found:
        return []
    partial = [(line, [])]
    for s in found:
        grown = []
        for text, words in partial:
            for c in suggestions(text, s, before, embed, k):
                grown.append((fill(text, s, c["word"]),
                              words + [{"explicit": s["word"], **{key: c[key] for key in
                                        ("word", "kind", "fit", "similarity", "score")}}]))
        partial = grown[:MAX_COMBINATIONS]
        if not partial:
            return []
    return [{"line": text, "words": words} for text, words in partial]


def cleanup(tok, model, line: str, words: list[str]) -> str | None:
    from radio_ready.generate import complete

    prompt = CLEANUP_PROMPT.format(word=", ".join(words), line=line)
    raw = complete(tok, model, [{"role": "user", "content": prompt}], 1, 64, 0.0, 1.0, False)[0]
    text = raw.strip().splitlines()[0].strip().strip('"') if raw.strip() else ""
    keeps = all(re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE) for w in words)
    return text if text and keeps else None


def rank(c: dict) -> tuple:
    """Lower is better: usable, passing the filters, then meaning + slot fit - fluency lost.
    The slot fit (distilroberta-base) keeps the part of speech of the slot: similarity alone
    prefers words close to the explicit word in meaning, even as the wrong kind of word
    ("money ain't exist")."""
    fit = [w["fit"] for w in json.loads(c["raw"])]
    joint = c["similarity"] + FIT_WEIGHT * sum(fit) / len(fit) - c["fluency_drop"]
    return not c.get("usable"), not c["passed"], -joint


def choose(selector, original: str, cands: list[tuple]) -> tuple[int | None, list[dict]]:
    _, scored = selector.pick([original], [(r, l) for r, l, _ in cands])
    for c, (_, _, how) in zip(scored, cands):
        c["how"] = how
        c["usable"] = bool(c["passed"] and c["fluency_drop"] <= MAX_FLUENCY_DROP)
    best = min(range(len(scored)), key=lambda j: rank(scored[j])) if scored else None
    return best, scored


def swap_section(record: dict, targets: dict, detector, selector, embed, args,
                 llm=None) -> dict:
    lines = list(record["lines"])
    units, raws, any_sub, all_ok = [], [], False, True
    for i in sorted(targets):
        before = lines[i - 1] if i else ""
        subs = substitutions(record["lines"][i], before, detector, embed, args.k)
        cands = [(json.dumps(s["words"]), [s["line"]], "substitution") for s in subs]
        if llm and subs:
            _, first = choose(selector, record["lines"][i], cands)
            ranked = sorted(range(len(first)), key=lambda j: rank(first[j]))
            for j in ranked[: args.cleanup_k]:
                words = [w["word"] for w in subs[j]["words"]]
                fixed = cleanup(*llm, subs[j]["line"], words)
                if fixed and fixed != subs[j]["line"]:
                    cands.append((json.dumps(subs[j]["words"]), [fixed], "cleanup"))
        best, scored = choose(selector, record["lines"][i], cands)
        units.append({"start": i, "end": i + 1, "chosen": best,
                      "distinct": len({c["lines"][0] for c in scored}), "candidates": scored})
        if best is None:
            all_ok = False
            continue
        lines[i] = scored[best]["lines"][0]
        raws.append(scored[best]["raw"])
        any_sub = True
        all_ok &= scored[best]["usable"]
    rewrites = {str(i): l for i, (l, o) in enumerate(zip(lines, record["lines"])) if l != o}
    return {"raw": "\n".join(raws), "rewrites": rewrites, "lines": lines if any_sub else None,
            "valid": any_sub, "passed": any_sub and all_ok, "units": units}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rhyme-swap pairs from the rhyme thesaurus.")
    parser.add_argument("--input", default=str(paths.KIDZBOP_DIR / "pilot.jsonl"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--k", type=int, default=4, help="suggestions tried per explicit word")
    parser.add_argument("--cleanup", action="store_true",
                        help=f"{CLEANUP_MODEL} also fixes grammar around the new word")
    parser.add_argument("--cleanup-k", type=int, default=2,
                        help="substitutions per line the model cleans up")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    version = VERSION + ("-cleanup" if args.cleanup else "")
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
    llm = None
    if args.cleanup:
        from radio_ready.generate import load_model

        llm = load_model(CLEANUP_MODEL, True, False)
    model = "distilroberta-base" + (f"+{CLEANUP_MODEL}" if args.cleanup else "")
    lines_total = lines_sub = lines_usable = 0
    with open(out, "a", encoding="utf-8") as f:
        for n, record in enumerate(todo, start=1):
            start = time.time()
            targets = target_lines(record, detector)
            if not targets:
                print(f"[{n}/{len(todo)}] {record['id']} skipped: no line has a detected word")
                continue
            result = swap_section(record, targets, detector, selector, embed, args, llm)
            for u in result["units"]:
                lines_total += 1
                if u["chosen"] is not None:
                    lines_sub += 1
                    lines_usable += u["candidates"][u["chosen"]]["usable"]
            f.write(json.dumps({
                "id": record["id"], "model": model, "quantization": "nf4" if llm else None,
                "thinking": False, "temperature": 0.0, "top_p": None, "samples": args.k,
                "prompt_version": version, "style": "rhyme-swap",
                "targets": sorted(targets), **result,
                "seconds": round(time.time() - start, 1)}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(todo)}] {record['id']} valid={result['valid']} "
                  f"passed={result['passed']} {time.time() - start:.1f}s", flush=True)
    print(f"target lines: {lines_total}, with a substitution: {lines_sub}, usable "
          f"(filters + fluency drop <= {MAX_FLUENCY_DROP}): {lines_usable}; output: {out}")


if __name__ == "__main__":
    main()
