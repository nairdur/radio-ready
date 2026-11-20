"""Download raw datasets into data/raw/ and the scoring models into the Hugging Face cache.

Run:
    python -m radio_ready.download all
    python -m radio_ready.download paradetox kidz-bop
    python -m radio_ready.download spotify-tracks mgphot surge-profanity
    python -m radio_ready.download models
"""

import argparse
import urllib.request

from huggingface_hub import snapshot_download

from radio_ready.paths import KIDZ_BOP_CSV, RAW

HF_DATASETS = {
    "paradetox": "s-nlp/paradetox",
    "genius": "theelderemo/genius-lyrics-cleaned",
    "spotify-tracks": "maharshipandya/spotify-tracks-dataset",  # Spotify API `explicit` flag
    "surge-profanity": "mmathys/profanity",  # Surge AI Obscenity List mirror (MIT)
}

# MGPHot (TISMIR 2025): expert "Explicit Lyrics" scores for 21k Billboard Hot 100 songs.
MGPHOT_URL = "https://zenodo.org/records/15372063/files/{}?download=1"
MGPHOT_FILES = ["mgphot_genes.tsv", "mgphot_gene_values.tsv"]

# TwitterAAE (Blodgett et al., EMNLP 2016, MIT): demographic dialect model, data files only.
TWITTERAAE_URL = "https://raw.githubusercontent.com/slanglab/twitteraae/master/model/{}"
TWITTERAAE_FILES = ["model_vocab.txt", "model_count_table.txt"]

KIDZ_BOP_URL = (
    "https://raw.githubusercontent.com/the-pudding/data/master/kidz-bop/KB_censored-lyrics.csv"
)

# Filtering models (ParaDeHate, Sec. 3.2-3.3) and evaluation models (Sec. 3.4).
MODELS = [
    "sentence-transformers/all-distilroberta-v1",  # similarity filter, cosine >= 0.70
    "unitary/unbiased-toxic-roberta",  # toxicity filter, score >= 0.9 is toxic
    "sentence-transformers/LaBSE",  # content-preservation metric
    "s-nlp/roberta_toxicity_classifier",  # style-accuracy metric (ParaDetox classifier)
    "cointegrated/roberta-large-cola-krishna2020",  # fluency (ParaDetox), candidate ranking
]

SOURCES = [
    "paradetox", "kidz-bop", "genius", "spotify-tracks", "surge-profanity", "mgphot", "twitteraae",
    "models",
]


def download_hf_dataset(name: str) -> None:
    target = RAW / name
    snapshot_download(
        repo_id=HF_DATASETS[name],
        repo_type="dataset",
        local_dir=target,
        allow_patterns=["*.csv", "*.tsv", "*.parquet", "*.jsonl", "README.md"],
    )
    print(f"{name}: {target}")


def download_kidz_bop() -> None:
    target = KIDZ_BOP_CSV
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(KIDZ_BOP_URL, target)
    print(f"kidz-bop: {target}")


def download_mgphot() -> None:
    target = RAW / "mgphot"
    target.mkdir(parents=True, exist_ok=True)
    for name in MGPHOT_FILES:
        urllib.request.urlretrieve(MGPHOT_URL.format(name), target / name)
    print(f"mgphot: {target}")


def download_twitteraae() -> None:
    target = RAW / "twitteraae"
    target.mkdir(parents=True, exist_ok=True)
    for name in TWITTERAAE_FILES:
        urllib.request.urlretrieve(TWITTERAAE_URL.format(name), target / name)
    print(f"twitteraae: {target}")


def download_models() -> None:
    for repo_id in MODELS:
        path = snapshot_download(
            repo_id=repo_id,
            ignore_patterns=["onnx/*", "openvino/*", "*.onnx", "*.h5", "*.msgpack", "*.ot",
                             "tf_model*", "flax_model*", "rust_model*"],
        )
        print(f"model {repo_id}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="+", choices=SOURCES + ["all"])
    args = parser.parse_args()

    selected = SOURCES if "all" in args.sources else args.sources
    for source in selected:
        if source in HF_DATASETS:
            download_hf_dataset(source)
        elif source == "kidz-bop":
            download_kidz_bop()
        elif source == "mgphot":
            download_mgphot()
        elif source == "twitteraae":
            download_twitteraae()
        elif source == "models":
            download_models()


if __name__ == "__main__":
    main()
