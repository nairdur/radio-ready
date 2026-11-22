"""Evaluation of rewrite systems on the held-out sections, built on radio_ready.compare.

radio_ready.compare (unchanged) gives the per-section line metrics and loads the generation runs
and commercial answers; this module adds the systems in outputs/systems (radio_ready.baselines,
radio_ready.train), the ParaDetox-style metrics, Radio-J and the reference scores.

Systems: every outputs/generations/*.jsonl and filled output_<model> column of
data/commercial/commercial_table.csv (radio_ready.compare.load_systems), every
outputs/systems/*.jsonl, kidz_bop when the input sections have Kidz Bop edits, and `copy` (the
original section unchanged; a reference row that shows what similarity alone rewards; left out
of the human-ranking files). A section is compared only where a system has a valid rewrite.

Metrics per (section, system):
  explicit_left    detected explicit words left in the rewrite (radio_ready.explicit)
  clean            explicit_left == 0
  lines_changed    number of lines that differ from the original
  syl_exact        share of changed lines with the same syllable count as the original
  syl_abs_diff     mean |syllables(original) - syllables(rewrite)| over changed lines
  rhyme_kept       share of changed lines whose end word is kept or rhymes (CMU rhyming part)
                   (unchanged lines are left out of these three: copying a line is not a
                   meter or rhyme success; NaN if nothing changed)
  clean_lines_kept share of original lines without an explicit word that are copied exactly
  slang_kept       share of benign slang tokens (vetting verdict not_explicit) in the original
                   that are still in the rewrite
  similarity       LaBSE cosine of original and rewritten section (SIM)
  aae_delta        AAE proportion of the rewrite minus that of the original
  sta              1 if no line of the rewrite is classified toxic (P(toxic) < 0.5) by
                   s-nlp/roberta_toxicity_classifier (ParaDetox STA; a different model from
                   the filters that select candidates)
  fl               share of changed lines that are fluent: CoLA acceptability (the ParaDetox
                   fluency model) >= min(0.5, acceptability of the original line), so lines in
                   dialect are not penalized for scoring low on both sides; 1 if nothing changed
  syl_within1      share of changed lines within one syllable of the original
  j                sta * sim * fl, per section (ParaDetox joint score)
  radio_j          j * syl_within1 * rhyme_kept (NaN counted as 1): our headline score, high
                   only if the rewrite is clean, faithful, fluent and singable
  ref_chrf         for sections with human-approved reference lines (radio_ready.references:
                   liked lines of test-split and reserved sections, pairs/lines.jsonl): chrF of
                   each referenced line against its best reference, averaged; references a
                   system proposed itself (its name in `proposed_by`) are skipped, so Claude is
                   never scored against Claude
  ref_exact        share of referenced lines that match a reference exactly (case and spacing
                   ignored)

Outputs in outputs/evaluation/<name>/ (default name: the input file, plus the group):
  scores.csv        one row per (section, system)
  summary.csv       mean per system
  side_by_side.md   line-by-line view
  ranking.csv       blinded A/B/C... order per section for human ranking; ranking_key.csv maps
                    letters to systems

--systems limits the comparison to the named systems; --common keeps only sections that every
compared system has.

Run:  python -m radio_ready.evaluate [--input FILE] [--group commercial|test] [--systems A B ...]
                                    [--common] [--no-copy]
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from radio_ready import paths
from radio_ready import compare
from radio_ready.aae import AAEScorer, vernacular_shift
from radio_ready.compare import EVAL_SIMILARITY_MODEL, line_metrics
from radio_ready.explicit import ExplicitDetector
import torch

from radio_ready.commercial import reserved_ids
from radio_ready.eval_set import EVAL_SECTIONS_JSONL, SYSTEMS_DIR
from radio_ready.references import chrf, references
from radio_ready.scoring import Fluency, Similarity
from radio_ready.text import syllables

OUT_DIR = paths.OUTPUTS / "evaluation"
COPY = "copy"
TOXIC = 0.5
FLUENT = 0.5


class ToxicityEval:
    """P(toxic) from s-nlp/roberta_toxicity_classifier, the ParaDetox style-accuracy (STA)
    classifier. Evaluation only: candidates are filtered with a different model."""

    MODEL = "s-nlp/roberta_toxicity_classifier"

    def __init__(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL).to(
            self.device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.index = labels.get("toxic", 1)

    @torch.no_grad()
    def score(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[i : i + batch_size], padding=True, truncation=True,
                                   max_length=128, return_tensors="pt").to(self.device)
            out.append(torch.softmax(self.model(**batch).logits, -1)[:, self.index]
                       .float().cpu().numpy())
        return np.concatenate(out) if out else np.array([])


def load_systems(ids: set[str]) -> dict[str, dict[str, list[str]]]:
    """compare.load_systems (generation runs, commercial answers) plus outputs/systems."""
    systems = compare.load_systems(ids)
    for path in sorted(SYSTEMS_DIR.glob("*.jsonl")):
        for rec in map(json.loads, open(path, encoding="utf-8")):
            if rec["id"] in ids and rec.get("valid"):
                systems.setdefault(path.stem, {})[rec["id"]] = rec["lines"]
    return systems


def syl_within1(original: list[str], rewrite: list[str]) -> float:
    """Share of changed lines within one syllable of the original (NaN if nothing changed)."""
    changed = [(a, b) for a, b in zip(original, rewrite) if a != b]
    return (np.mean([abs(syllables(a) - syllables(b)) <= 1 for a, b in changed])
            if changed else np.nan)


def reference_scores(system: str, rewrite: list[str], original: list[str], refs: dict) -> dict:
    scores, exact = [], []
    for i, options in refs.items():
        usable = [b for a, b, source in options if i < len(original)
                  and a == original[i].strip() and system.lower() not in source.lower()]
        if not usable:
            continue
        scores.append(max(chrf(rewrite[i], b) for b in usable))
        exact.append(any(" ".join(rewrite[i].lower().split()) == " ".join(b.lower().split())
                         for b in usable))
    return {"ref_lines": len(scores), "ref_chrf": np.mean(scores) if scores else np.nan,
            "ref_exact": np.mean(exact) if exact else np.nan}


def model_scores(scores: pd.DataFrame, sections: dict, systems: dict) -> pd.DataFrame:
    """Add sta, fl, j and radio_j (one pass of each scoring model over the unique lines)."""
    lines = {l for rw in systems.values() for ls in rw.values() for l in ls if l.strip()}
    lines |= {l for r in sections.values() for l in r["lines"] if l.strip()}
    lines = sorted(lines)
    tox = dict(zip(lines, ToxicityEval().score(lines)))
    acc = dict(zip(lines, Fluency().score(lines)))
    sta, fl = [], []
    for sid, system in zip(scores["id"], scores["system"]):
        original, rewrite = sections[sid]["lines"], systems[system][sid]
        sta.append(float(max([tox[l] for l in rewrite if l.strip()] or [0.0]) < TOXIC))
        changed = [(a, b) for a, b in zip(original, rewrite) if a != b]
        fl.append(np.mean([acc.get(b, 0.0) >= min(FLUENT, acc.get(a, 0.0)) for a, b in changed])
                  if changed else 1.0)
    scores["sta"], scores["fl"] = sta, fl
    scores["j"] = scores["sta"] * scores["similarity"].clip(lower=0) * scores["fl"]
    scores["radio_j"] = (scores["j"] * scores["syl_within1"].fillna(1.0)
                         * scores["rhyme_kept"].fillna(1.0))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-by-side scores of rewrite systems.")
    parser.add_argument("--input", default=str(EVAL_SECTIONS_JSONL))
    parser.add_argument("--name", default=None, help="output folder (default: input file stem)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--systems", nargs="+", default=None)
    parser.add_argument("--group", default=None,
                        help="only sections with this eval_group (commercial or test)")
    parser.add_argument("--common", action="store_true")
    parser.add_argument("--no-copy", action="store_true", help="leave out the copy reference row")
    args = parser.parse_args()

    sections = {r["id"]: r for r in map(json.loads, open(args.input, encoding="utf-8"))
                if not args.group or r.get("eval_group") == args.group}
    systems = {"kidz_bop": {i: r["kidz_bop_lines"] for i, r in sections.items()
                            if r.get("kidz_bop_lines")}}
    systems |= load_systems(set(sections))
    if args.systems:
        systems = {k: v for k, v in systems.items() if k in args.systems}
    systems = {k: v for k, v in systems.items() if v}  # e.g. kidz_bop on Genius sections
    if not systems:
        raise SystemExit("no system has a valid rewrite of these sections")
    if args.common:
        shared = set.intersection(*(set(v) for v in systems.values()))
        systems = {k: {i: l for i, l in v.items() if i in shared} for k, v in systems.items()}
        sections = {i: r for i, r in sections.items() if i in shared}
    ranked_names = list(systems)  # systems shown to human rankers
    if not args.no_copy:
        systems[COPY] = {i: list(r["lines"]) for i, r in sections.items()}
    detector = ExplicitDetector()
    aae = AAEScorer()
    refs = references(extra_ids=reserved_ids())

    rows = []
    for system, rewrites in systems.items():
        for sid, lines in rewrites.items():
            original = sections[sid]["lines"]
            m = line_metrics(original, lines, detector)
            m["syl_within1"] = syl_within1(original, lines)
            m.update({"id": sid, "system": system, "clean": m["explicit_left"] == 0,
                      "aae_delta": vernacular_shift(aae, original, lines)["aae_delta"],
                      "original_text": "\n".join(original), "rewrite_text": "\n".join(lines)})
            m.update(reference_scores(system, lines, original, refs.get(sid, {})))
            rows.append(m)
    scores = pd.DataFrame(rows)
    sim = Similarity(EVAL_SIMILARITY_MODEL)
    scores["similarity"] = sim.pairwise(scores["original_text"].tolist(),
                                        scores["rewrite_text"].tolist()).round(4)
    scores = model_scores(scores, sections, systems)

    out = OUT_DIR / (args.name or Path(args.input).stem
                                  + (f"_{args.group}" if args.group else ""))
    out.mkdir(parents=True, exist_ok=True)
    cols = ["id", "system", "clean", "explicit_left", "lines_changed", "syl_exact", "syl_within1",
            "syl_abs_diff", "rhyme_kept", "clean_lines_kept", "slang_kept", "similarity", "sta",
            "fl", "j", "radio_j", "aae_delta", "ref_lines", "ref_chrf", "ref_exact"]
    scores[cols].round(4).to_csv(out / "scores.csv", index=False)
    summary = scores.groupby("system").agg(
        sections=("id", "size"), clean=("clean", "mean"), explicit_left=("explicit_left", "mean"),
        lines_changed=("lines_changed", "mean"),
        syl_exact=("syl_exact", "mean"), syl_within1=("syl_within1", "mean"),
        syl_abs_diff=("syl_abs_diff", "mean"),
        rhyme_kept=("rhyme_kept", "mean"), clean_lines_kept=("clean_lines_kept", "mean"),
        slang_kept=("slang_kept", "mean"), similarity=("similarity", "mean"),
        sta=("sta", "mean"), fl=("fl", "mean"), j=("j", "mean"), radio_j=("radio_j", "mean"),
        aae_delta=("aae_delta", "mean"), ref_lines=("ref_lines", "sum"),
        ref_chrf=("ref_chrf", "mean"), ref_exact=("ref_exact", "mean")).round(3).sort_values("radio_j", ascending=False)
    summary.to_csv(out / "summary.csv")

    names = ranked_names
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
