"""ParaDeHate pairs (hate speech -> detoxified text) in the lyric pair format.

Inputs:  data/raw/paradehate (python -m radio_ready.download paradehate; ScaDSAI/ParaDeHate,
         8,276 pairs rewritten by GPT-4o-mini, Yuan et al. 2025; no license stated)
Output:  data/processed/paradehate/pairs.jsonl

Each pair uses the fields of radio_ready.pairs (PAIR_KEYS): the text is one line, so
original = [toxic text], rewrite = [detoxified text], changed = [0]; artist, title and
section_header are null; source = "paradehate", model = "gpt-4o-mini", style = "paradehate".
Added fields: explicit_left (detected explicit words in the rewrite), aae_original and
aae_rewrite (AAE proportion, radio_ready.aae; null if too few known tokens).

Kept: non-empty pairs whose rewrite differs from the original; duplicates of an original text
are kept once. With --keep-explicit, rewrites with a detected explicit word are kept too
(default: dropped). Split: 80/10/10 by a hash of the original text.

Run:  python -m radio_ready.paradehate [--keep-explicit]
"""

import argparse
import hashlib
import json
from statistics import mean

import pandas as pd

from radio_ready import paths
from radio_ready.aae import AAEScorer
from radio_ready.explicit import ExplicitDetector
from radio_ready.pairs import PAIR_KEYS
from radio_ready.text import normalize

SPLITS = [("train", 0.8), ("val", 0.9), ("test", 1.0)]


def split_of(text: str) -> str:
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return next(name for name, upper in SPLITS if h < upper)


def main() -> None:
    parser = argparse.ArgumentParser(description="ParaDeHate pairs in the lyric pair format.")
    parser.add_argument("--keep-explicit", action="store_true",
                        help="keep rewrites that still have a detected explicit word")
    args = parser.parse_args()

    raw = pd.read_parquet(paths.PARADEHATE_PARQUET)
    detector, aae = ExplicitDetector(), AAEScorer()
    seen, pairs, dropped = set(), [], {"empty_or_same": 0, "duplicate": 0, "explicit_left": 0}
    for toxic, clean in zip(raw["Original Text"], raw["Converted Text"]):
        original, rewrite = normalize(str(toxic or "")), normalize(str(clean or ""))
        if not original or not rewrite or original == rewrite:
            dropped["empty_or_same"] += 1
            continue
        if original in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(original)
        left = detector.find(rewrite)
        if left and not args.keep_explicit:
            dropped["explicit_left"] += 1
            continue
        pair = {
            "id": hashlib.md5(original.encode("utf-8")).hexdigest()[:12],
            "source": "paradehate", "from": None, "edited": False, "artist": None,
            "title": None, "section_header": None, "split": split_of(original),
            "original": [original], "rewrite": [rewrite], "changed": [0],
            "model": "gpt-4o-mini", "prompt_version": None, "style": "paradehate",
            "human_edited": [], "human_verdicts": None,
        }
        pair = {k: pair[k] for k in PAIR_KEYS}
        pair.update({"explicit_left": left, "aae_original": aae.proportion(original),
                     "aae_rewrite": aae.proportion(rewrite)})
        pairs.append(pair)

    paths.PARADEHATE_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.PARADEHATE_PAIRS, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    both = [(p["aae_original"], p["aae_rewrite"]) for p in pairs
            if p["aae_original"] is not None and p["aae_rewrite"] is not None]
    splits = pd.Series([p["split"] for p in pairs]).value_counts().to_dict()
    print(f"raw: {len(raw)}, kept: {len(pairs)}, dropped: {dropped}, splits: {splits}")
    print(f"AAE proportion, original -> rewrite: {mean(a for a, _ in both):.3f} -> "
          f"{mean(b for _, b in both):.3f} (pairs with both: {len(both)}); "
          f"share losing >= 0.2: {mean((a - b) >= 0.2 for a, b in both):.1%}")
    print(f"output: {paths.PARADEHATE_PAIRS}")


if __name__ == "__main__":
    main()
