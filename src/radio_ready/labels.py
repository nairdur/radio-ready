"""Join song-level explicit labels onto Genius lyrics.

Label sources:
  spotify  Spotify API `explicit` flag (maharshipandya/spotify-tracks-dataset); a song is
           explicit if any of its versions is flagged.
  mgphot   Music Genome Project analysts' "Explicit Lyrics" score (0-1) for Billboard Hot 100
           songs (MGPHot, TISMIR 2025). Used as the gold evaluation set:
           explicit if score >= MGPHOT_EXPLICIT, clean if score == 0, otherwise ambiguous.

Songs are matched on a normalized primary-artist + title key. Matched songs are split by
artist into train/val/test (60/20/20, as in Fell et al. 2019); songs with an MGPHot label form
the `gold` split and are never trained on.

Inputs:  data/raw/genius, data/raw/spotify-tracks, data/raw/mgphot
Output:  data/processed/labels/songs.parquet (lyrics, labels, split)

Run:  python -m radio_ready.labels
"""

import hashlib
import json
import re

import pandas as pd
import pyarrow.parquet as pq

from radio_ready import paths

OUT_DIR = paths.PROCESSED / "labels"
MGPHOT_DIR = paths.RAW / "mgphot"
SPOTIFY_CSV = paths.RAW / "spotify-tracks" / "dataset.csv"
MGPHOT_EXPLICIT = 0.6

FEAT = re.compile(r"\s*[\(\[](?:feat|ft|with|prod)\.?[^\)\]]*[\)\]]", re.IGNORECASE)
VERSION = re.compile(
    r"\s*(?:-\s*|\()\s*(?:\d{4}\s*)?(?:remaster|radio edit|clean|explicit|single version|"
    r"album version|live|mono|stereo|edit)[^\)]*\)?\s*$",
    re.IGNORECASE,
)
ARTIST_SPLIT = re.compile(r"\s*(?:;|,|&|\bfeaturing\b|\bfeat\.?|\bft\.?|\bx\b|\bwith\b|\band\b)\s*",
                          re.IGNORECASE)


def primary_artist(artist: str) -> str:
    return ARTIST_SPLIT.split(str(artist).strip(), maxsplit=1)[0]


def clean_title(title: str) -> str:
    title = FEAT.sub("", str(title))
    return VERSION.sub("", title).strip()


def song_key(artist: str, title: str) -> str:
    raw = f"{primary_artist(artist)}|{clean_title(title)}".lower().replace("&", "and")
    return re.sub(r"[^a-z0-9|]", "", raw)


def load_spotify() -> pd.Series:
    sp = pd.read_csv(SPOTIFY_CSV, usecols=["artists", "track_name", "explicit"])
    sp = sp.dropna(subset=["artists", "track_name"])
    sp["key"] = [song_key(a, t) for a, t in zip(sp["artists"], sp["track_name"])]
    return sp.groupby("key")["explicit"].any().rename("spotify_explicit")


def load_mgphot() -> pd.Series:
    genes = pd.read_csv(MGPHOT_DIR / "mgphot_genes.tsv", sep="\t")
    gene_id = int(genes.loc[genes["name"] == "Explicit Lyrics", "gene_id"].iloc[0])
    mg = pd.read_csv(MGPHOT_DIR / "mgphot_gene_values.tsv", sep="\t")
    mg["score"] = mg["gene_values"].map(lambda v: json.loads(v)[gene_id])
    mg["key"] = [song_key(a, t) for a, t in zip(mg["artist"], mg["title"])]
    return mg.groupby("key")["score"].max().rename("mgphot_score")


def artist_split(artist: str) -> str:
    bucket = int(hashlib.md5(primary_artist(artist).lower().encode()).hexdigest(), 16) % 10
    return "train" if bucket < 6 else "val" if bucket < 8 else "test"


def main() -> None:
    spotify, mgphot = load_spotify(), load_mgphot()
    print(f"spotify songs: {len(spotify)} ({spotify.mean():.1%} explicit), mgphot songs: {len(mgphot)}")

    parts = []
    for shard in sorted(paths.GENIUS_DIR.rglob("*.parquet")):
        df = pq.read_table(shard, columns=["title", "artist", "tag", "year", "lyrics"]).to_pandas()
        df["key"] = [song_key(a, t) for a, t in zip(df["artist"], df["title"])]
        df = df[df["key"].isin(spotify.index) | df["key"].isin(mgphot.index)]
        parts.append(df)
    songs = pd.concat(parts).drop_duplicates("key").reset_index(drop=True)
    songs = songs.join(spotify, on="key").join(mgphot, on="key")

    songs["mgphot_label"] = pd.NA
    songs.loc[songs["mgphot_score"] >= MGPHOT_EXPLICIT, "mgphot_label"] = True
    songs.loc[songs["mgphot_score"] == 0, "mgphot_label"] = False
    songs["split"] = songs["artist"].map(artist_split)
    songs.loc[songs["mgphot_score"].notna(), "split"] = "gold"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "songs.parquet"
    songs.to_parquet(out, index=False)

    print(f"matched genius songs: {len(songs)}")
    summary = songs.groupby("split").agg(
        songs=("key", "size"),
        spotify_labelled=("spotify_explicit", "count"),
        spotify_explicit=("spotify_explicit", "mean"),
        mgphot_labelled=("mgphot_label", "count"),
    )
    print(summary.round(3).to_string())
    gold = songs[songs["mgphot_label"].notna() & songs["spotify_explicit"].notna()]
    if len(gold):
        agree = (gold["mgphot_label"].astype(bool) == gold["spotify_explicit"]).mean()
        print(f"spotify vs mgphot agreement on {len(gold)} songs: {agree:.1%}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
