# radio-ready

## Extract explicit-word stanzas by artist

`src/extract_explicit_stanzas.py` reads JSONL lyrics, infers artists by
comparing consecutive song-title slugs, derives the song title from the remaining
slug text, chooses one random song per artist, and splits each song into
bracket-section chunks.

Run it from the repository root:

```powershell
python src/extract_explicit_stanzas.py
```

Options:

- `--input` selects the JSONL lyrics file.
- `--words-csv` selects the explicit-word CSV.
- `--output` selects the output JSON file.
- `--stanza-size` sets the number of lyric lines per chunk (default: `4`).
- `--seed` sets the random-song sampling seed (default: `42`).

The word CSV can have a `word` header and one term per row. Whole-word,
case-insensitive matching is used.

The script's explicit defaults are:

- Input: `data/genius-expertise/lyrics.jl`
- Word CSV: `reference/explicit_words.csv`
- Output: `output/genius_explicit_stanzas_by_artist.json`

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

