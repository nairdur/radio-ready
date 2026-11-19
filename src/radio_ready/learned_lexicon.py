"""Learn an explicit-word dictionary from labelled songs (Fell et al., RANLP 2019, Sec. 3.1).

Importance of word w:  I(w) = rate(w, explicit songs) / rate(w, clean songs), where rate is the
share of songs containing w (add-one smoothed). Words in fewer than --min-df explicit songs
(too unique) or in more than --max-clean-rate of clean songs (too common) are dropped, and the
top 1,000 are kept as candidates.

Two classifiers, as in the paper:
  lookup      explicit if the song contains any of the top-n words (n tuned on val)
  regression  balanced logistic regression on counts of the top-n words

Baseline: lookup with the Surge AI Obscenity List (human severity ratings), single-word entries
with mean severity >= --surge-min-severity. Surge ratings are also attached to every learned
word as a cross-check.

Run:  python -m radio_ready.learned_lexicon
Needs data/processed/labels/songs.parquet (python -m radio_ready.labels).
Writes data/artifacts/learned_explicit_words.csv and data/artifacts/lexicon_metrics.json.
"""

import argparse
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from radio_ready import paths

SONGS = paths.PROCESSED / "labels" / "songs.parquet"
SURGE_CSV = paths.RAW / "surge-profanity" / "profanity_en.csv"
HEADER_LINE = re.compile(r"^\s*\[[^\]]*\]\s*$", re.MULTILINE)
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)*")
SIZES = [4, 8, 16, 32, 64, 128, 256, 512]


def tokenize(lyrics: str) -> list[str]:
    return TOKEN.findall(HEADER_LINE.sub(" ", str(lyrics).lower()).replace("’", "'"))


def metrics(y_true, y_pred) -> dict:
    return {
        "macro_f1": round(100 * f1_score(y_true, y_pred, average="macro"), 1),
        "explicit_f1": round(100 * f1_score(y_true, y_pred), 1),
        "precision": round(100 * precision_score(y_true, y_pred, zero_division=0), 1),
        "recall": round(100 * recall_score(y_true, y_pred), 1),
        "n": int(len(y_true)),
    }


def load_surge() -> pd.DataFrame:
    s = pd.read_csv(SURGE_CSV)
    s["text"] = s["text"].str.lower().str.strip()
    return s.groupby("text").agg(
        surge_severity=("severity_rating", "max"), surge_category=("category_1", "first")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn an explicit-word dictionary (Fell 2019).")
    parser.add_argument("--min-df", type=int, default=10)
    parser.add_argument("--max-clean-rate", type=float, default=0.3)
    parser.add_argument("--surge-min-severity", type=float, default=2.0)
    args = parser.parse_args()

    songs = pd.read_parquet(SONGS)
    songs["tokens"] = songs["lyrics"].map(tokenize)
    songs["vocab"] = songs["tokens"].map(set)

    sets = {
        "train": songs[songs["split"] == "train"],
        "val": songs[songs["split"] == "val"],
        "test": songs[songs["split"] == "test"],
        "gold": songs[songs["mgphot_label"].notna()],
    }
    labels = {
        k: (v["mgphot_label"] if k == "gold" else v["spotify_explicit"]).astype(bool).to_numpy()
        for k, v in sets.items()
    }

    train, y = sets["train"], labels["train"]
    df_ex, df_cl = Counter(), Counter()
    for vocab, label in zip(train["vocab"], y):
        (df_ex if label else df_cl).update(vocab)
    n_ex, n_cl = int(y.sum()), int((~y).sum())

    rows = []
    for w, dfe in df_ex.items():
        dfc = df_cl.get(w, 0)
        if dfe < args.min_df or dfc / n_cl > args.max_clean_rate:
            continue
        ratio = ((dfe + 1) / (n_ex + 2)) / ((dfc + 1) / (n_cl + 2))
        rows.append((w, ratio, dfe, dfc))
    lex = pd.DataFrame(rows, columns=["word", "importance", "df_explicit", "df_clean"])
    lex = lex.sort_values("importance", ascending=False).head(1000).reset_index(drop=True)
    lex.insert(0, "rank", lex.index + 1)
    lex = lex.join(load_surge(), on="word")

    def lookup(split: str, words: set[str]) -> np.ndarray:
        return np.array([bool(v & words) for v in sets[split]["vocab"]])

    def counts(split: str, vocab: list[str]) -> np.ndarray:
        index = {w: i for i, w in enumerate(vocab)}
        x = np.zeros((len(sets[split]), len(vocab)), dtype=np.float32)
        for r, tokens in enumerate(sets[split]["tokens"]):
            for t in tokens:
                if t in index:
                    x[r, index[t]] += 1
        return np.log1p(x)

    val_scores = {}
    for n in SIZES:
        words = set(lex["word"].head(n))
        val_scores[("lookup", n)] = f1_score(labels["val"], lookup("val", words), average="macro")
        vocab = lex["word"].head(n).tolist()
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(counts("train", vocab), y)
        val_scores[("regression", n)] = f1_score(
            labels["val"], clf.predict(counts("val", vocab)), average="macro"
        )

    best = {m: max(SIZES, key=lambda n: val_scores[(m, n)]) for m in ("lookup", "regression")}
    results = {"label_counts": {k: {"songs": int(len(v)), "explicit": int(v.sum())}
                                for k, v in labels.items()},
               "val_macro_f1_by_size": {f"{m}@{n}": round(100 * s, 1)
                                        for (m, n), s in val_scores.items()},
               "best_size": best, "test": {}, "gold": {}}

    lookup_words = set(lex["word"].head(best["lookup"]))
    reg_vocab = lex["word"].head(best["regression"]).tolist()
    clf = LogisticRegression(class_weight="balanced", max_iter=2000).fit(counts("train", reg_vocab), y)
    surge = load_surge()
    surge_words = set(surge.index[(surge["surge_severity"] >= args.surge_min_severity)
                                  & ~surge.index.str.contains(" ")])
    for split in ("test", "gold"):
        results[split]["majority_clean"] = metrics(labels[split], np.zeros_like(labels[split]))
        results[split][f"surge_lookup_sev>={args.surge_min_severity}"] = metrics(
            labels[split], lookup(split, surge_words))
        results[split][f"learned_lookup@{best['lookup']}"] = metrics(
            labels[split], lookup(split, lookup_words))
        results[split][f"learned_regression@{best['regression']}"] = metrics(
            labels[split], clf.predict(counts(split, reg_vocab)))

    lex["in_lookup"] = lex["rank"] <= best["lookup"]
    lex["in_regression"] = lex["rank"] <= best["regression"]
    lex["importance"] = lex["importance"].round(2)
    paths.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    lex.to_csv(paths.ARTIFACTS / "learned_explicit_words.csv", index=False)
    with open(paths.ARTIFACTS / "lexicon_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results["label_counts"]))
    print("best sizes:", best)
    for split in ("test", "gold"):
        print(f"\n{split}:")
        print(pd.DataFrame(results[split]).T.to_string())
    top = lex.head(best["lookup"])
    print(f"\ntop-{best['lookup']} learned words with a Surge rating: "
          f"{top['surge_severity'].notna().sum()}/{len(top)}")
    print(f"outputs: {paths.ARTIFACTS / 'learned_explicit_words.csv'}, lexicon_metrics.json")


if __name__ == "__main__":
    main()
