import argparse
import shutil
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY_ROOT / "data"

DATASETS = {
    "genius-kaggle": (
        "carlosgdcj/genius-song-lyrics-with-language-information",
        DATA_ROOT / "genius-song-lyrics",
    ),
    "spotify-kaggle": (
        "devdope/900k-spotify",
        DATA_ROOT / "500k-spotify-lyrics",
    ),
}


def copy_download(download_path: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0

    for source in download_path.rglob("*"):
        if not source.is_file():
            continue

        target = destination / source.name
        shutil.copy2(source, target)
        copied += 1

    return copied


def download_dataset(dataset_name: str) -> tuple[Path, int]:
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "kagglehub is not installed. Run: python -m pip install kagglehub"
        ) from error

    handle, destination = DATASETS[dataset_name]
    download_path = Path(kagglehub.dataset_download(handle))
    copied = copy_download(download_path, destination)
    return destination, copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Kaggle datasets used by radio-ready."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=["all", *DATASETS],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = list(DATASETS) if args.dataset == "all" else [args.dataset]

    for dataset_name in dataset_names:
        destination, copied = download_dataset(dataset_name)
        print(
            f"Copied {copied} file(s) for {dataset_name} "
            f"to {destination}"
        )


if __name__ == "__main__":
    main()
