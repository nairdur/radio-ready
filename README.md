# RadioReady

Clean radio edits of explicit song lyrics that keep the original meaning, syllable count, and
end rhyme.

## Approach

1. **Data**: explicit lyric sections from Genius for synthetic rewriting; sections of Kidz Bop
   songs as a separate side-by-side and evaluation set.
2. **SFT**: LoRA fine-tune Qwen2.5-1.5B-Instruct on ParaDetox, then on lyric pairs conditioned
   on syllable and rhyme control tokens.
3. **DPO**: preference pairs ranked by the constraint scorers.
4. **Evaluation**: explicit-content removal, semantic similarity, syllable and rhyme accuracy,
   human ratings.

## Layout

```
annotations/  human-edited files (committed)
data/         raw/, artifacts/, processed/ (gitignored)
notebooks/    exploration
outputs/      models and metrics (gitignored)
src/radio_ready/
  download.py         raw data and models
  labels.py           song-level explicit labels
  learned_lexicon.py  learned explicit-word list + human vetting
  classifier.py       section-level explicit classifier
  explicit.py         explicit-word detector used by the pipeline
  kidzbop.py          Kidz Bop song sections and word stats
  genius.py           Genius cleaning, sections, sampling
  aae.py              AAE vernacular proportion
  scoring.py          similarity and toxicity models
  text.py             normalization, syllables, stress, rhyme
  paths.py            file locations
  llm_judgment/       LLM-judged artifacts, not used by the pipeline
```

## Setup

```bash
python -m venv .venv
```

```bash
pip install -r requirements.txt
```

```bash
pip install -e .
```

## Pipeline

Run from the repository root, in order.

| Step | Command | Inputs | Outputs |
| --- | --- | --- | --- |
| 1 | `python -m radio_ready.download all` | — | `data/raw/*`, model cache |
| 2 | `python -m radio_ready.labels` | Genius, Spotify, MGPHot | `data/processed/labels/songs.parquet` |
| 3 | `python -m radio_ready.learned_lexicon` | labels, Surge, vetting | `data/artifacts/learned_explicit_words.csv`, `lexicon_metrics.json` |
| 4 | `python -m radio_ready.classifier` | labels | `outputs/explicit_classifier/` |
| 5 | `python -m radio_ready.kidzbop` | Kidz Bop, Genius, step 3 | `data/artifacts/kidzbop_*.csv`, `data/processed/kidzbop/{sections,pilot}.jsonl` |
| 6 | `python -m radio_ready.genius` | Genius, Kidz Bop, step 3 | `data/processed/genius_sections.jsonl` |
| 7 | `python -m radio_ready.aae` | step 6, TwitterAAE | adds `aae_proportion` to step 6 output |

### 2. Labels

Song-level labels joined onto Genius by normalized primary artist + title:

- Spotify `explicit` flag: a song is explicit if any version is flagged. Split by artist
  60/20/20 into train/val/test.
- MGPHot "Explicit Lyrics" score: explicit if ≥ 0.6, clean if 0. These songs form the `gold`
  split and are not trained on.

### 3. Learned lexicon and vetting

Explicit-word list learned from the labels (Fell et al. 2019):
`importance(w) = rate(w | explicit songs) / rate(w | clean songs)`. The lookup size is tuned on
val. `lexicon_metrics.json` reports lookup and logistic regression on test and gold, with the
Surge list (severity ≥ 1 and ≥ 2) and the combined list as comparisons.

`annotations/explicit_word_vetting.csv` holds a `verdict` per word:

| Verdict | Effect |
| --- | --- |
| `not_explicit` | removed before ranking |
| `explicit` | always included |
| `unsure` / blank | none |

`--add-to-vetting N` appends unvetted words from the top N with a blank verdict. Existing rows
are never changed.

### 4. Classifier

DistilRoBERTa trained on one random section per song with the song's Spotify label; a song's
score is the max over its sections; the threshold is tuned on val.

### Explicit-word detection

`explicit.py` is used by steps 5 and 6. A text is explicit if any token is in the word list,
or it contains a censored token (`f*ck`). Word lists (`--detector`):

- `combined` (default): `learned`, plus Surge entries with severity ≥ 1 that are also among the
  learned candidates (`learned_explicit_words.csv`, after vetting)
- `learned`: lookup words from step 3 after vetting, plus words marked `explicit`
- `surge`: Surge AI Obscenity List entries with severity ≥ 1 (`SURGE_MIN_SEVERITY`)

Gold-set lookup results are in `lexicon_metrics.json` (`combined_sev>=*`, `learned_lookup@*`,
`surge_lookup_sev>=*`).

### 5. Kidz Bop songs

A separate dataset, not part of the Genius sample. Uses: 7B rewrites for side-by-side human
ranking against the Kidz Bop version, evaluation of the trained model, and a pilot subset for
testing 7B generation before running it on all Genius sections.

Each Kidz Bop song is matched to Genius (fuzzy title + artist), each Kidz Bop line is located in
the cleaned sections (line similarity ≥ 0.75, `--min-line-match`), and every eligible explicit
section (same rules as step 6) is kept. Sections with an edited line also get `kidz_bop_lines`
(the section with the Kidz Bop lines in place) and `kidz_bop_edits` (position, original, Kidz
Bop line, cut flag, matched bad words). Splits are by song. `pilot` marks a seeded sample of
sections with an edit (`--pilot-size`, default 50).

Outputs: `data/processed/kidzbop/sections.jsonl`, `data/processed/kidzbop/pilot.jsonl`.

Word statistics: per censored word, category, row/occurrence/song/artist counts, cut rate, and
the most common replacements from a word-level diff of original and Kidz Bop lines
(`data/artifacts/kidzbop_words.csv`, `kidzbop_substitutions.csv`).

### 6. Genius sections

Cleaning: section headers become metadata (type, performers); credit headers, inline tags and
scrape artifacts are removed; look-alike Cyrillic/Greek letters are mapped to Latin;
parenthetical ad-libs move to `adlibs`; lines without syllables are dropped; repeated sections
are kept once.

Sampling: a section is eligible if its type is verse, chorus, hook, pre-/post-chorus, bridge or
refrain, it has 4–16 lines, and it contains a detected explicit word. One random eligible song
per artist, one random eligible section from it; artists with fewer than 3 eligible songs are
dropped; Kidz Bop songs are excluded. Each record has per-line syllables, stress, rhyme scheme
(stressed-vowel classes), ad-libs, and up to 8 lines of context before and after.

### 7. AAE proportion

Proportion of tokens from the African American English topic of the TwitterAAE model
(Blodgett et al. 2016). `aae.vernacular_shift(scorer, original, rewrite)` returns the
proportion before and after a rewrite.

## Scoring models

| Model | Use | Threshold |
| --- | --- | --- |
| `sentence-transformers/all-distilroberta-v1` | similarity filter | cosine ≥ 0.70 (ParaDeHate) |
| `unitary/unbiased-toxic-roberta` | toxicity filter | ≥ 0.9 (ParaDeHate) |
| `sentence-transformers/LaBSE` | meaning-preservation metric | — |
| `s-nlp/roberta_toxicity_classifier` | clean-output metric | — |

## LLM-judged artifacts (not in the pipeline)

`src/radio_ready/llm_judgment/` contains artifacts based on Claude's judgment. No pipeline step
imports them.

- `claude_word_tiers.py`: strong / mild / kb_only tiers for Kidz Bop words, with word sets
  written by Claude. Output: `data/artifacts/llm_judgment/claude_word_tiers.csv`.

## Licenses

The code in this repository is MIT-licensed ([LICENSE](LICENSE)). Data, models and third-party
code keep their own licenses:

### Datasets

| Dataset | License |
| --- | --- |
| [ParaDetox](https://huggingface.co/datasets/s-nlp/paradetox) | OpenRAIL++ |
| [The Pudding data (Kidz Bop)](https://github.com/the-pudding/data/tree/master/kidz-bop) | MIT |
| [Genius lyrics (cleaned)](https://huggingface.co/datasets/theelderemo/genius-lyrics-cleaned) | MIT (dataset); lyrics are copyrighted by their rights holders |
| [Spotify tracks](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) | BSD |
| [MGPHot](https://zenodo.org/records/15372063) | CC BY-NC-ND 4.0 |
| [Surge AI Obscenity List](https://huggingface.co/datasets/mmathys/profanity) | MIT |
| [TwitterAAE model](https://github.com/slanglab/twitteraae) | MIT |

### Models

| Model | License |
| --- | --- |
| `sentence-transformers/all-distilroberta-v1` | Apache-2.0 |
| `unitary/unbiased-toxic-roberta` | Apache-2.0 |
| `sentence-transformers/LaBSE` | Apache-2.0 |
| `s-nlp/roberta_toxicity_classifier` | OpenRAIL++ |
| `distilroberta-base` | Apache-2.0 |
| `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` | Apache-2.0 |

### Code

| Component | License |
| --- | --- |
| `aae.py` inference, adapted from `slanglab/twitteraae` `predict.py` | MIT, © 2019 Su Lin Blodgett and Brendan O'Connor (notice in file) |
| `pronouncing` | BSD-3-Clause |
| `cmudict` (Python package, dependency of `pronouncing`) | GPL-3.0-or-later; dictionary data under the CMUdict BSD-style license |

## References

- Blodgett, Green, O'Connor. Demographic Dialectal Variation in Social Media: A Case Study of
  African-American English. EMNLP 2016.
- Fell, Cabrio, Corazza, Gandon. Comparing Automated Methods to Detect Explicit Content in Song
  Lyrics. RANLP 2019.
- Logacheva et al. ParaDetox: Detoxification with Parallel Data. ACL 2022.
- Yuan et al. LLM in the Loop: Creating the ParaDeHate Dataset for Hate Speech
  Detoxification. Findings of IJCNLP-AACL 2025.
