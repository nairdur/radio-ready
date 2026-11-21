"""Export SFT pairs (explicit section -> radio edit) from generation files.

Inputs:  outputs/generations/<name>.jsonl (radio_ready.generate), and the section records they
         were generated from (Genius sections, Kidz Bop sections)
Output:  data/processed/pairs/<name>.jsonl, one pair per section:
  id, source (genius / kidzbop), artist, title, section_header, split
  original           the explicit section lines
  rewrite            the radio edit lines (same number of lines)
  changed            indexes of the lines that differ
  syllables          syllables per line of the original (control token values)
  rewrite_syllables  syllables per line of the rewrite
  rhyme_scheme       end-rhyme letters of the original (control token values)
  model, prompt_version, style
  scores             the chosen candidates' filter scores (similarity, toxicity, fluency_drop,
                     syllable difference, rhymes broken), one entry per request

A pair is kept when the generation is valid, every chosen candidate passed the ParaDeHate
filters (radio_ready.generate), at least one line changed, and the detector finds no explicit
word in the rewrite. Genius sections reserved for commercial LLMs are never exported (they
are an evaluation set). Splits: Genius pairs by artist (60/20/20, the same hash as
radio_ready.labels), Kidz Bop pairs keep their song split.

Run:  python -m radio_ready.pairs GENERATION [GENERATION ...] [--name NAME]
      python -m radio_ready.pairs qwen3-14b-4bit_genius_sections_v8-stanza-free_n5
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from radio_ready import paths
from radio_ready.commercial import reserved_ids
from radio_ready.explicit import ExplicitDetector
from radio_ready.labels import artist_split
from radio_ready.text import syllables

PAIRS_DIR = paths.PROCESSED / "pairs"
SECTION_FILES = {"genius": paths.GENIUS_SECTIONS_JSONL,
                 "kidzbop": paths.KIDZBOP_DIR / "sections.jsonl"}
SCORE_KEYS = ["similarity", "toxicity", "fluency_drop", "syllable_diff", "rhymes_broken"]


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
    return path if path.suffix == ".jsonl" and path.exists() else \
        paths.GENERATIONS_DIR / f"{Path(name).stem}.jsonl"


def to_pair(gen: dict, source: str, rec: dict) -> dict:
    original, rewrite = rec["lines"], gen["lines"]
    chosen = [u["candidates"][u["chosen"]] for u in gen.get("units", [])
              if u.get("chosen") is not None]
    return {
        "id": rec["id"], "source": source, "artist": rec["artist"], "title": rec["title"],
        "section_header": rec.get("section_header"),
        "split": rec.get("split") if source == "kidzbop" else artist_split(rec["artist"]),
        "original": original, "rewrite": rewrite,
        "changed": [i for i, (a, b) in enumerate(zip(original, rewrite)) if a != b],
        "syllables": rec["syllables"], "rewrite_syllables": [syllables(l) for l in rewrite],
        "rhyme_scheme": rec["rhyme_scheme"],
        "model": gen["model"], "prompt_version": gen["prompt_version"],
        "style": gen.get("style"),
        "scores": [{k: c.get(k) for k in SCORE_KEYS} for c in chosen],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SFT pairs from generation files.")
    parser.add_argument("generations", nargs="+", help="names in outputs/generations or paths")
    parser.add_argument("--name", default=None, help="output name (default: first input)")
    args = parser.parse_args()

    sections, reserved, detector = load_sections(), reserved_ids(), ExplicitDetector()
    pairs, dropped, seen = [], Counter(), set()
    for name in args.generations:
        for line in open(generation_path(name), encoding="utf-8"):
            gen = json.loads(line)
            if gen["id"] in seen:
                dropped["duplicate id"] += 1
            elif gen["id"] in reserved:
                dropped["reserved for commercial evaluation"] += 1
            elif gen["id"] not in sections:
                dropped["section record not found"] += 1
            elif not gen.get("valid"):
                dropped["no usable answer"] += 1
            elif gen.get("passed") is False:
                dropped["failed filters"] += 1
            elif gen["lines"] == sections[gen["id"]][1]["lines"]:
                dropped["nothing changed"] += 1
            elif detector.find("\n".join(gen["lines"])):
                dropped["explicit word left"] += 1
            else:
                pairs.append(to_pair(gen, *sections[gen["id"]]))
                seen.add(gen["id"])

    out = PAIRS_DIR / f"{args.name or Path(args.generations[0]).stem}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"pairs: {len(pairs)} {dict(Counter(p['split'] for p in pairs))}")
    print(f"dropped: {dict(dropped)}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
