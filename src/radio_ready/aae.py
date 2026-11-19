"""Proportion of African American English (AAE) vernacular in a section, as a preservation metric.

Model: TwitterAAE (Blodgett, Green & O'Connor, EMNLP 2016), used unmodified. Its learned count
tables are downloaded with `python -m radio_ready.download twitteraae`. It is a mixed-membership
model: for a text it infers what proportion of the *words* come from an AAE-associated dialect
topic. That is a property of the language in the text, not a label on the text or its author,
so it is stored as a number, never as a class.

Metric use: `vernacular_shift(original, rewrite)` compares the proportion before and after a
rewrite. A large drop means the rewrite standardized the vernacular instead of only removing
explicit content.

Run:  python -m radio_ready.aae [path.jsonl ...]
Adds `aae_proportion` (None if too few tokens are in the model's vocabulary) to every record
with a `lines` field, rewriting the file in place. Default: data/processed/genius_sections.jsonl.

The inference below (`proportion`) is adapted from slanglab/twitteraae code/predict.py:
    Copyright (c) 2019 Su Lin Blodgett and Brendan O'Connor
    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files (the "Software"), to deal in the Software
    without restriction, including without limitation the rights to use, copy, modify, merge,
    publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
    to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
"""

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from radio_ready import paths

MODEL_DIR = paths.RAW / "twitteraae"
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)*|[^\sa-z0-9]")
AAE_TOPIC = 0  # model topics: African-American, Hispanic, Asian, White
OLD_FIELDS = ("aae_aa", "aae_white", "aae_coverage", "aae_lines")  # from an earlier version


class AAEScorer:
    def __init__(self, alpha: float = 1.0, passes: int = 5, min_tokens: int = 1,
                 min_coverage: float = 0.2):
        counts = np.loadtxt(MODEL_DIR / "model_count_table.txt")
        self.wordprobs = (counts + 1) / counts.sum(0)
        with open(MODEL_DIR / "model_vocab.txt", encoding="utf-8") as f:
            vocab = [line.rstrip("\n").split("\t")[-1].strip() for line in f]
        assert len(vocab) == counts.shape[0]
        self.index = {w: i for i, w in enumerate(vocab)}
        self.alpha, self.passes = alpha, passes
        self.min_tokens, self.min_coverage = min_tokens, min_coverage

    @lru_cache(maxsize=100_000)
    def proportion(self, text: str) -> float | None:
        """Share of words from the AAE topic (CVB0 inference, as in predict.py)."""
        tokens = TOKEN.findall(text.lower().replace("’", "'"))
        known = [self.index[t] for t in tokens if t in self.index]
        if len(known) < self.min_tokens or len(known) / max(len(tokens), 1) < self.min_coverage:
            return None
        lik = self.wordprobs[known]
        lik = lik / lik.sum(1, keepdims=True)
        q = lik.copy()
        q_k = q.sum(0)
        for _ in range(1, self.passes):
            for i in range(len(known)):
                q_k -= q[i]
                q[i] = lik[i] * (q_k + self.alpha)
                q[i] /= q[i].sum()
                q_k += q[i]
        return round(float(q_k[AAE_TOPIC] / q_k.sum()), 4)


def _text(lines: list[str] | str) -> str:
    return "\n".join(lines) if isinstance(lines, list) else lines


def vernacular_shift(scorer: AAEScorer, original: list[str] | str,
                     rewrite: list[str] | str) -> dict:
    """AAE proportion before and after a rewrite; a negative delta means vernacular was lost."""
    before, after = scorer.proportion(_text(original)), scorer.proportion(_text(rewrite))
    delta = None if before is None or after is None else round(after - before, 4)
    return {"aae_before": before, "aae_after": after, "aae_delta": delta}


def tag_file(scorer: AAEScorer, path: Path) -> list:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for r in records:
        for field in OLD_FIELDS:
            r.pop(field, None)
        if "lines" in r:
            r["aae_proportion"] = scorer.proportion(_text(r["lines"]))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Add AAE vernacular proportions to JSONL records.")
    parser.add_argument("files", nargs="*", type=Path,
                        default=[paths.PROCESSED / "genius_sections.jsonl"])
    args = parser.parse_args()
    scorer = AAEScorer()
    for path in args.files:
        records = tag_file(scorer, path)
        scored = np.array([r["aae_proportion"] for r in records
                           if r.get("aae_proportion") is not None])
        print(f"{path}: {len(records)} records, {len(scored)} scored, "
              f"median {np.median(scored):.3f}, p10 {np.percentile(scored, 10):.3f}, "
              f"p90 {np.percentile(scored, 90):.3f}")


if __name__ == "__main__":
    main()
