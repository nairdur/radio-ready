"""Surgical, singable rewrites of the lines that keep a section from being paired.

Targets: the lines of the vetting app's human list (radio_ready.pairs.export_human): the
section has accepted lines, this explicit line has no accepted version in any source and no
proposal waiting for a verdict (one line per group of repeated lines).

Two passes with one model (default Qwen2.5-7B-Instruct, 4-bit):
1. Meaning draft: the 4-line stanza with the line marked and its explicit words named; the
   model replaces only those words with clean words that mean the same (greedy).
2. Sing: the stanza with the draft in place, the original line's syllable count and the end
   words it rhymes with; --samples answers rewrite the draft so it sings like the verse. The
   model never sees explicit text: other explicit lines of the stanza are shown with their
   first accepted version (any source), or with their draft.

Candidates (the draft and the sung answers) are checked like radio_ready.free (no explicit or
disguised word, toxicity < 0.9, real words, changed), must keep the meaning (cosine >= 0.70 to
the draft and to the original line) and must not copy another line of the section. Ranked: end rhyme kept (with the original or its rhyme
partners), within one syllable, then meaning to the draft x fluency (CoLA, relative to the
original). A line with no passing candidate gets no proposal.

Output: outputs/generations/sing_<name>_sing-v2.jsonl in the radio_ready.generate format; the
vetting app lists the runs as the source sing-v2. Liked lines complete their sections in the
human pairs.

--reselect applies the checks and ranking again to an existing run (scores are stored per
candidate), without a model.

Run:  python -m radio_ready.sing [--limit N] [--samples 4]
      python -m radio_ready.sing --reselect
"""

import argparse
import json
import time

from radio_ready import paths
from radio_ready.explicit import ExplicitDetector
from radio_ready.free import Scorer
from radio_ready.pairs import (
    HUMAN, HUMAN_REQUEST, accepted_texts, load_sections, load_sheet,
)
from radio_ready.prompts import clean_response_line
from radio_ready.rewrite import load, rhyme_instruction, rhyme_partners, rhymes, unload
from radio_ready.scoring import SIMILARITY_THRESHOLD
from radio_ready.text import syllables

VERSION = "sing-v2"
MODEL = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM = ("You are a professional lyricist who writes radio edits of songs. You change only what "
          "radio would censor and keep everything else as the artist wrote it.")

DRAFT_PROMPT = """\
Lines from "{title}" by {artist}:

{stanza}

The line marked >>> has words that radio would censor: {words}.
Replace only those words with clean words that mean the same thing here, so the line and the \
stanza keep their meaning. Keep every other word exactly as it is, including slang and dialect.

Answer with the rewritten line only, without quotes or anything else."""

SING_PROMPT = """\
Lines from "{title}" by {artist}:

{stanza}

The line marked >>> was edited for radio and now sounds flat next to the others. Rewrite it so \
it sings like the rest of this verse:
- keep what it says, in the voice, slang and dialect of the verse;
- {syllables} syllables, with the same rhythm as the lines around it;
- {rhyme}
- no word radio would censor, and no sound-alike of one.

Answer with the rewritten line only, without quotes or anything else."""


def stanza(lines: list[str], i: int) -> str:
    start = (i // 4) * 4
    return "\n".join((">>> " if j == i else "    ") + lines[j]
                     for j in range(start, min(len(lines), start + 4)))


def first_line(text: str) -> str | None:
    rows = [clean_response_line(r.strip().strip("`\"'")) for r in text.strip().splitlines()]
    rows = [r.lstrip(">").strip().strip("\"'") for r in rows if r]
    return rows[0] if rows else None


def check(c: dict, i: int, section: list[str]) -> bool:
    """The candidate passes: free.py's checks, meaning kept (to the draft and to the original
    line) and not a copy of another line of the section."""
    others = {l.strip().lower() for j, l in enumerate(section) if j != i}
    return bool(c["ok"] and c["meaning_vs_draft"] >= SIMILARITY_THRESHOLD
                and c["similarity"] >= SIMILARITY_THRESHOLD
                and c["line"].strip().lower() not in others)


def choose(scored: list[dict]) -> int | None:
    order = sorted(range(len(scored)), key=lambda k: (
        not scored[k]["passed"], not scored[k]["rhymes"], scored[k]["syllable_diff"] > 1,
        -scored[k]["meaning_vs_draft"] * (1 - scored[k]["fluency_drop"])))
    return order[0] if order and scored[order[0]]["passed"] else None


def reselect(path) -> None:
    sections = load_sections()
    gens = [json.loads(l) for l in open(path, encoding="utf-8")]
    before = sum(u["chosen"] is not None for g in gens for u in g["units"])
    for g in gens:
        original = sections[g["id"]][1]["lines"]
        final = list(original)
        for u in g["units"]:
            for c in u["candidates"]:
                c["passed"] = check(c, u["start"], original)
            u["chosen"] = choose(u["candidates"])
            if u["chosen"] is not None:
                final[u["start"]] = u["candidates"][u["chosen"]]["line"]
        g["lines"] = final
        g["rewrites"] = {str(i): l for i, (l, o) in enumerate(zip(final, original)) if l != o}
        done = [u for u in g["units"] if u["chosen"] is not None]
        g["valid"], g["passed"] = bool(done), len(done) == len(g["units"])
    with open(path, "w", encoding="utf-8") as f:
        for g in gens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    after = sum(u["chosen"] is not None for g in gens for u in g["units"])
    print(f"{path.name}: lines with a proposal {before} -> {after}")


def targets(rows: list[dict]) -> dict[str, list[int]]:
    """{section id: line indexes} of the human list still open (no accepted version, nothing
    pending)."""
    accepted = {(r["id"], int(r["line"])) for r in rows
                if r.get("verdict", "").strip().lower() in ("like", "superlike")}
    pending = {(r["id"], int(r["line"])) for r in rows
               if r.get("rewrite") and not r.get("verdict", "").strip()}
    found: dict[str, list[int]] = {}
    for r in rows:
        key = (r["id"], int(r["line"]))
        if (f"{HUMAN}: {HUMAN_REQUEST}" in r.get("sources", "") and key not in accepted
                and key not in pending):
            found.setdefault(r["id"], []).append(int(r["line"]) - 1)
    return {k: sorted(set(v)) for k, v in found.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Surgical, singable rewrites of missing lines.")
    parser.add_argument("--limit", type=int, default=None, help="sections")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--name", default="human-list")
    parser.add_argument("--reselect", action="store_true",
                        help="apply the checks and ranking again to existing runs, no model")
    args = parser.parse_args()
    if args.reselect:
        for path in sorted(paths.GENERATIONS_DIR.glob(f"sing_*_{VERSION}.jsonl")):
            reselect(path)
        return

    detector = ExplicitDetector()
    sections = load_sections()
    rows = load_sheet(sections)
    todo = targets(rows)
    accepted = accepted_texts(rows)
    ids = [i for i in sorted(todo) if i in sections][: args.limit]
    print(f"sections: {len(ids)}, lines: {sum(len(todo[i]) for i in ids)}", flush=True)
    scorer = Scorer(detector)
    tok, model = load(args.model, False)
    from radio_ready.generate import complete

    out = paths.GENERATIONS_DIR / f"sing_{args.name}_{VERSION}.jsonl"
    started = time.time()
    with open(out, "w", encoding="utf-8") as f:
        for n, sid in enumerate(ids, start=1):
            t0 = time.time()
            rec = sections[sid][1]
            original = rec["lines"]
            # what the model sees: no explicit text anywhere
            shown = list(original)
            for j, line in enumerate(original):
                if detector.find(line) and accepted.get((sid, j + 1)):
                    shown[j] = accepted[(sid, j + 1)][0][0]
            drafts = {}
            for i in todo[sid]:
                words = sorted(set(detector.find(original[i])))
                msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
                         DRAFT_PROMPT.format(title=rec["title"], artist=rec["artist"],
                                             stanza=stanza(shown, i), words=", ".join(words))}]
                drafts[i] = first_line(complete(tok, model, msgs, 1, 128, 0.0, 1.0, False)[0])
            for i, d in drafts.items():
                if d:
                    shown[i] = d
            final, units = list(original), []
            for i in todo[sid]:
                draft = drafts.get(i)
                if not draft:
                    units.append({"start": i, "end": i + 1, "chosen": None, "candidates": []})
                    continue
                msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
                         SING_PROMPT.format(title=rec["title"], artist=rec["artist"],
                                            stanza=stanza(shown, i),
                                            syllables=syllables(original[i]),
                                            rhyme=rhyme_instruction(rec, i, detector))}]
                sung = [s for s in (first_line(r) for r in complete(
                    tok, model, msgs, args.samples, 128, 1.0, 0.9, False)) if s]
                cands = list(dict.fromkeys([draft] + sung))
                scored = scorer.score(cands, [original[i]] * len(cands))
                to_draft = scorer.similarity.pairwise(cands, [draft] * len(cands))
                partners = rhyme_partners(rec, i)
                for c, m in zip(scored, to_draft):
                    c["source"] = "draft" if c["line"] == draft else "sung"
                    c["meaning_vs_draft"] = round(float(m), 3)
                    c["rhymes"] = rhymes(c["line"], original[i], partners)
                    c["passed"] = check(c, i, original)
                chosen = choose(scored)
                if chosen is not None:
                    final[i] = scored[chosen]["line"]
                units.append({"start": i, "end": i + 1, "draft": draft, "chosen": chosen,
                              "candidates": scored})
            done = [u for u in units if u["chosen"] is not None]
            f.write(json.dumps({
                "id": sid, "model": f"sing+{args.model}", "prompt_version": VERSION,
                "style": "sing", "samples": args.samples, "temperature": 1.0,
                "targets": todo[sid], "raw": "",
                "rewrites": {str(i): l for i, (l, o) in enumerate(zip(final, original)) if l != o},
                "lines": final, "valid": bool(done),
                "passed": len(done) == len(units), "units": units,
                "seconds": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(ids)}] {sid} {len(done)}/{len(units)} lines", flush=True)
    unload(model, tok)
    print(f"{len(ids)} sections in {(time.time() - started) / 60:.1f} min, output: {out}")


if __name__ == "__main__":
    main()
