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

- Input: `C:\Users\light\Development\radio-ready\data\genius-expertise\lyrics.jl`
- Word CSV: `C:\Users\light\Development\radio-ready\reference\explicit_words.csv`
- Output: `C:\Users\light\Development\radio-ready\output\genius_explicit_stanzas.csv`

