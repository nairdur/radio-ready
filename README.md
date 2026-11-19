# radio-ready

## Extract explicit-word stanzas

`src/extract_explicit_stanzas.py` searches the JSONL lyrics dataset for words
listed in `reference/explicit_words.csv`. It writes one CSV row per
matching lyrics line, including the four surrounding lines above and below.

Run it from the repository root:

```powershell
python src/extract_explicit_stanzas.py
```

Options:

- `--input` selects the JSONL lyrics file.
- `--words-csv` selects the explicit-word CSV.
- `--output` selects the output CSV.
- `--context` changes the number of surrounding lines (default: `4`).

The word CSV can have a `word` header and one term per row. Whole-word,
case-insensitive matching is used.

The script's explicit defaults are:

- Input: `data/genius-expertise/lyrics.jl`
- Word CSV: `reference/explicit_words.csv`
- Output: `output/genius_explicit_stanzas.csv`

## Set up datasets

The project uses three data sources:

| Source | Local folder |
| --- | --- |
| Cornell genius-expertise dataset: https://www.cs.cornell.edu/~arb/data/genius-expertise/ | `data/genius-expertise` |
| Kaggle Genius lyrics: `carlosgdcj/genius-song-lyrics-with-language-information` | `data/genius-song-lyrics` |
| Kaggle 900K Spotify: `devdope/900k-spotify` | `data/500k-spotify-lyrics` |

Install the Kaggle download dependency:

```powershell
python -m pip install -r requirements-data.txt
```

Authenticate Kaggle before downloading. The Kaggle API checks for credentials
through the usual local Kaggle configuration (normally `~/.kaggle/kaggle.json`
or Kaggle environment variables).

Download both Kaggle datasets:

```powershell
python src/setup_data.py all
```

Or download one dataset at a time:

```powershell
python src/setup_data.py genius-kaggle
python src/setup_data.py spotify-kaggle
```

The Cornell [genius-expertise data](https://www.cs.cornell.edu/~arb/data/genius-expertise/)
is not downloaded by the Kaggle setup script. Download the Cornell archive or
files listed on that page and place them in `data/genius-expertise`.

