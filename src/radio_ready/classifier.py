"""Section-level explicit-lyrics classifier trained from song-level labels.

Recipe (MiSonGyny 2026, CorpusDelicti-UC3M): songs are split into sections (genius.split_sections);
each epoch samples one random section per training song and gives it the song's Spotify label.
At inference a song's score is the max over its sections, and the decision threshold is tuned
on val for macro-F1. Reported on the Spotify test split and on the MGPHot gold split, next to
the lexicon baselines in data/artifacts/lexicon_metrics.json (Fell et al. 2019 setup).

The trained model scores single sections, so it can label Genius sections as explicit (rewrite)
or clean (keep as no-change examples).

Run:  python -m radio_ready.classifier [--epochs 3]
Writes outputs/explicit_classifier/{model/, metrics.json}.
"""

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from radio_ready import paths
from radio_ready.genius import split_sections
from radio_ready.learned_lexicon import metrics

SONGS = paths.PROCESSED / "labels" / "songs.parquet"
OUT_DIR = paths.ROOT / "outputs" / "explicit_classifier"
BASE_MODEL = "distilroberta-base"


def song_windows(lyrics: str, window: int = 8) -> list[str]:
    """Sections as text; songs without usable sections fall back to fixed line windows."""
    sections = ["\n".join(s.lines) for s in split_sections(lyrics) if len(s.lines) >= 2]
    if sections:
        return sections
    lines = [l for l in str(lyrics).splitlines() if l.strip()]
    return ["\n".join(lines[i : i + window]) for i in range(0, len(lines), window)] or [""]


class ExplicitClassifier:
    def __init__(self, model_dir: str = BASE_MODEL, max_length: int = 128):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, num_labels=1).to(self.device)
        self.max_length = max_length

    def _batch(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length,
                              return_tensors="pt").to(self.device)

    @torch.no_grad()
    def score(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        self.model.eval()
        out = []
        for i in range(0, len(texts), batch_size):
            with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                logits = self.model(**self._batch(texts[i : i + batch_size])).logits[:, 0]
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    def song_scores(self, windows: list[list[str]]) -> np.ndarray:
        flat = [w for ws in windows for w in ws]
        scores = self.score(flat)
        out, i = [], 0
        for ws in windows:
            out.append(scores[i : i + len(ws)].max())
            i += len(ws)
        return np.array(out)


def train(clf: ExplicitClassifier, windows, labels, epochs, batch_size, lr, seed) -> None:
    rng = random.Random(seed)
    pos_weight = torch.tensor((~labels).sum() / max(labels.sum(), 1), device=clf.device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(clf.model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(windows) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=clf.device == "cuda")
    order = list(range(len(windows)))
    for epoch in range(epochs):
        clf.model.train()
        rng.shuffle(order)
        total = 0.0
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            texts = [rng.choice(windows[i]) for i in idx]
            y = torch.tensor(labels[idx], dtype=torch.float32, device=clf.device)
            with torch.autocast(clf.device, dtype=torch.float16, enabled=clf.device == "cuda"):
                logits = clf.model(**clf._batch(texts)).logits[:, 0]
            loss = loss_fn(logits.float(), y)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()
            total += loss.item() * len(idx)
        print(f"epoch {epoch + 1}: loss {total / len(order):.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the section-level explicit classifier.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    songs = pd.read_parquet(SONGS)
    data = {}
    for split in ("train", "val", "test", "gold"):
        part = songs[songs["mgphot_label"].notna()] if split == "gold" else songs[songs["split"] == split]
        label_col = "mgphot_label" if split == "gold" else "spotify_explicit"
        data[split] = ([song_windows(l) for l in part["lyrics"]],
                       part[label_col].astype(bool).to_numpy())
        print(f"{split}: {len(part)} songs, {data[split][1].mean():.1%} explicit", flush=True)

    clf = ExplicitClassifier()
    train(clf, *data["train"], args.epochs, args.batch_size, args.lr, args.seed)

    val_scores = clf.song_scores(data["val"][0])
    thresholds = np.linspace(0.05, 0.95, 91)
    threshold = float(max(thresholds, key=lambda t: f1_score(
        data["val"][1], val_scores >= t, average="macro")))
    results = {"threshold": round(threshold, 2), "base_model": BASE_MODEL, **vars(args)}
    for split in ("val", "test", "gold"):
        scores = val_scores if split == "val" else clf.song_scores(data[split][0])
        results[split] = metrics(data[split][1], scores >= threshold)
        print(split, results[split], flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clf.model.save_pretrained(OUT_DIR / "model")
    clf.tokenizer.save_pretrained(OUT_DIR / "model")
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"output: {OUT_DIR}")


if __name__ == "__main__":
    main()
