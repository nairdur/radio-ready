"""Side-by-side comparison of rewrite systems on a section set (default: Kidz Bop pilot).

Systems: kidz_bop (sections with a Kidz Bop edit), every outputs/generations/*.jsonl, and every
filled output_<model> column of data/commercial/commercial_table.csv. A section is compared
only where a system has a valid rewrite.

Metrics per (section, system):
  explicit_left    detected explicit words left in the rewrite (radio_ready.explicit)
  clean            explicit_left == 0
  syl_exact        share of lines with the same syllable count as the original
  syl_abs_diff     mean |syllables(original line) - syllables(rewrite line)|
  rhyme_kept       share of lines whose end word is kept or rhymes (CMU rhyming part)
  clean_lines_kept share of original lines without an explicit word that are copied exactly
  slang_kept       share of benign slang tokens (vetting verdict not_explicit) in the original
                   that are still in the rewrite
  similarity       LaBSE cosine of original and rewritten section
  aae_delta        AAE proportion of the rewrite minus that of the original

Outputs in outputs/comparison/<name>/:
  scores.csv        one row per (section, system)
  summary.csv       mean per system
  side_by_side.md   line-by-line view
  ranking.csv       blinded A/B/C... order per section for human ranking; ranking_key.csv maps
                    letters to systems

--systems limits the comparison to the named systems; --common keeps only sections that every
compared system has.

Run:  python -m radio_ready.compare [--input FILE] [--name NAME] [--systems A B ...] [--common]
"""

import argparse
import csv
import json
import random

import numpy as np
import pandas as pd

from radio_ready import paths
from radio_ready.aae import AAEScorer, vernacular_shift
from radio_ready.commercial import load_rewrites
from radio_ready.explicit import ExplicitDetector
from radio_ready.prompts import slang_in
from radio_ready.scoring import Similarity
from radio_ready.text import end_rhyme_kept, syllables, tokenize

EVAL_SIMILARITY_MODEL = "sentence-transformers/LaBSE"


def load_systems(ids: set[str]) -> dict[str, dict[str, list[str]]]:
    systems: dict[str, dict[str, list[str]]] = {}
    for path in sorted(paths.GENERATIONS_DIR.glob("*.jsonl")):
        for rec in map(json.loads, open(path, encoding="utf-8")):
            if rec["id"] in ids and rec.get("valid"):
                systems.setdefault(path.stem, {})[rec["id"]] = rec["lines"]
    for model, rewrites in load_rewrites(ids).items():
        systems.setdefault(model, {}).update(rewrites)
    return systems


def line_metrics(original: list[str], rewrite: list[str], detector: ExplicitDetector) -> dict:
    syl = [(syllables(a), syllables(b)) for a, b in zip(original, rewrite)]
    clean_idx = [i for i, l in enumerate(original) if not detector.find(l)]
    slang = [(w, i) for i, l in enumerate(original) for w in slang_in(l)]
    return {
        "slang_kept": np.mean([w in tokenize(rewrite[i]) for w, i in slang]) if slang else np.nan,
        "explicit_left": len(detector.find("\n".join(rewrite))),
        "syl_exact": np.mean([a == b for a, b in syl]),
        "syl_abs_diff": np.mean([abs(a - b) for a, b in syl]),
        "rhyme_kept": np.mean([end_rhyme_kept(a, b) if b else False
                               for a, b in zip(original, rewrite)]),
        "clean_lines_kept": np.mean([original[i] == rewrite[i] for i in clean_idx])
        if clean_idx else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-by-side scores of rewrite systems.")
    parser.add_argument("--input", default=str(paths.KIDZBOP_DIR / "pilot.jsonl"))
    parser.add_argument("--name", default="kidzbop_pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--systems", nargs="+", default=None)
    parser.add_argument("--common", action="store_true")
    args = parser.parse_args()

    sections = {r["id"]: r for r in map(json.loads, open(args.input, encoding="utf-8"))}
    systems = {"kidz_bop": {i: r["kidz_bop_lines"] for i, r in sections.items()
                            if r.get("kidz_bop_lines")}}
    systems |= load_systems(set(sections))
    if args.systems:
        systems = {k: v for k, v in systems.items() if k in args.systems}
    if args.common:
        shared = set.intersection(*(set(v) for v in systems.values()))
        systems = {k: {i: l for i, l in v.items() if i in shared} for k, v in systems.items()}
        sections = {i: r for i, r in sections.items() if i in shared}
    detector = ExplicitDetector()
    aae = AAEScorer()

    rows = []
    for system, rewrites in systems.items():
        for sid, lines in rewrites.items():
            original = sections[sid]["lines"]
            m = line_metrics(original, lines, detector)
            m.update({"id": sid, "system": system, "clean": m["explicit_left"] == 0,
                      "aae_delta": vernacular_shift(aae, original, lines)["aae_delta"],
                      "original_text": "\n".join(original), "rewrite_text": "\n".join(lines)})
            rows.append(m)
    scores = pd.DataFrame(rows)
    sim = Similarity(EVAL_SIMILARITY_MODEL)
    scores["similarity"] = sim.pairwise(scores["original_text"].tolist(),
                                        scores["rewrite_text"].tolist()).round(4)

    out = paths.COMPARISON_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)
    cols = ["id", "system", "clean", "explicit_left", "syl_exact", "syl_abs_diff", "rhyme_kept",
            "clean_lines_kept", "slang_kept", "similarity", "aae_delta"]
    scores[cols].round(4).to_csv(out / "scores.csv", index=False)
    summary = scores.groupby("system").agg(
        sections=("id", "size"), clean=("clean", "mean"), explicit_left=("explicit_left", "mean"),
        syl_exact=("syl_exact", "mean"), syl_abs_diff=("syl_abs_diff", "mean"),
        rhyme_kept=("rhyme_kept", "mean"), clean_lines_kept=("clean_lines_kept", "mean"),
        slang_kept=("slang_kept", "mean"),
        similarity=("similarity", "mean"), aae_delta=("aae_delta", "mean")).round(3)
    summary.to_csv(out / "summary.csv")

    names = list(systems)
    rng = random.Random(args.seed)
    with open(out / "side_by_side.md", "w", encoding="utf-8") as md, \
            open(out / "ranking.csv", "w", encoding="utf-8", newline="") as rk, \
            open(out / "ranking_key.csv", "w", encoding="utf-8", newline="") as key:
        rank_w, key_w = csv.writer(rk), csv.writer(key)
        letters = [chr(ord("A") + i) for i in range(len(names))]
        rank_w.writerow(["id", "line", "original"] + [f"version_{l}" for l in letters]
                        + [f"rank_{l}" for l in letters] + ["notes"])
        key_w.writerow(["id", "letter", "system"])
        for sid, rec in sections.items():
            present = [n for n in names if sid in systems[n]]
            if not present:
                continue
            md.write(f"## {rec['artist']} - {rec['title']} | {rec['section_header']} | {sid}\n\n")
            md.write("| # | original | " + " | ".join(present) + " |\n")
            md.write("|---|---|" + "---|" * len(present) + "\n")
            for k, line in enumerate(rec["lines"]):
                cells = [systems[n][sid][k] for n in present]
                md.write(f"| {k + 1} | {line} | " + " | ".join(
                    c if c != line else "=" for c in cells) + " |\n")
            md.write("\n")
            order = present[:]
            rng.shuffle(order)
            for letter, n in zip(letters, order):
                key_w.writerow([sid, letter, n])
            for k, line in enumerate(rec["lines"]):
                versions = [systems[n][sid][k] for n in order]
                versions += [""] * (len(letters) - len(versions))
                rank_w.writerow([sid, k + 1, line] + versions + [""] * len(letters) + [""])

    print(summary.to_string())
    print(f"outputs: {out}")


if __name__ == "__main__":
    main()
