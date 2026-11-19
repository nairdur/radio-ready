# RadioReady

Clean "radio edits" of explicit song lyrics that keep the original meaning, syllable count, and
end rhyme.

## Approach

1. **Data**: pair explicit stanzas with clean rewrites, using human edits plus LLM-generated
   ones filtered for cleanliness, meaning, syllables, and rhyme.
2. **SFT**: LoRA fine-tune Qwen2.5-1.5B-Instruct on general detox, then on lyric pairs
   conditioned on syllable and rhyme control tokens.
3. **DPO**: rank sampled rewrites with the constraint scorers and train on preference pairs.
4. **Evaluation**: explicit-content removal, semantic similarity, syllable and rhyme accuracy,
   and human ratings.

## Layout

```
data/        datasets (gitignored)
notebooks/   exploration
outputs/     checkpoints, generations, metrics (gitignored)
src/         project code (radio_ready package)
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

## Data

Download everything into `data/raw/` and pre-cache the scoring models:

```bash
python -m radio_ready.download all
```

Or pick sources: `paradetox`, `kidz-bop`, `genius`, `spotify-tracks`, `surge-profanity`,
`mgphot`, `models`.

| Source | Folder | Use |
| --- | --- | --- |
| [ParaDetox](https://huggingface.co/datasets/s-nlp/paradetox) | `data/raw/paradetox` | general detox SFT |
| [The Pudding: Kidz Bop censored lyrics](https://github.com/the-pudding/data/tree/master/kidz-bop) | `data/raw/kidz-bop` | word substitutions, explicit-word list, human eval pairs |
| [Genius lyrics (cleaned)](https://huggingface.co/datasets/theelderemo/genius-lyrics-cleaned) (~2.4 GB) | `data/raw/genius` | explicit stanzas for synthetic rewrites |
| [Spotify tracks](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) | `data/raw/spotify-tracks` | Spotify API `explicit` flag per track (training labels) |
| [MGPHot](https://zenodo.org/records/15372063) (TISMIR 2025, CC BY-NC-ND) | `data/raw/mgphot` | expert "Explicit Lyrics" scores, Billboard Hot 100 (gold labels) |
| [Surge AI Obscenity List](https://huggingface.co/datasets/mmathys/profanity) (MIT) | `data/raw/surge-profanity` | 1.6k profanities with human severity ratings |

Scoring models (Hugging Face cache), following ParaDeHate:

| Model | Use |
| --- | --- |
| `sentence-transformers/all-distilroberta-v1` | similarity filter (cosine ≥ 0.70) |
| `unitary/unbiased-toxic-roberta` | toxicity filter (≥ 0.9 is toxic) |
| `sentence-transformers/LaBSE` | meaning-preservation metric |
| `s-nlp/roberta_toxicity_classifier` | clean-output metric |

## Data preparation

Run in order after downloading:

```bash
python -m radio_ready.lexicon
```

```bash
python -m radio_ready.kidzbop
```

```bash
python -m radio_ready.genius
```

1. **`lexicon`** builds `data/artifacts/explicit_words.csv` from the Kidz Bop censored words,
   sorted by frequency, with per-word stats (rows, songs, artists, cut rate) and the most
   common replacements found by diffing original vs. Kidz Bop lines
   (`data/artifacts/kidzbop_substitutions.csv`). Each word has a tier: `strong` (bleeped on
   radio, used for detection), `mild` (damn, hell), or `kb_only` (censored only by Kidz Bop).
2. **`kidzbop`** drops cut verses and duplicates, scores each human-written pair (similarity,
   toxicity + lexicon, syllables, end rhyme, edit ratio), keeps radio-relevant pairs (original
   has a strong explicit word or toxicity ≥ 0.5) minus near-duplicate lines within a song, takes
   the top 30% by `quality = similarity × clean × syllable score × rhyme bonus`, and splits by song into
   `data/processed/kidzbop/{train,val,test}.jsonl`. All scores are in `pairs_scored.csv`.
3. **`genius`** cleans lyrics (section headers → metadata, ad-libs separated, scrape junk and
   repeated choruses removed), finds sections with strong explicit words, and picks one random
   song per artist and one random explicit section from it. Kidz Bop songs are excluded.
   Output: `data/processed/genius_sections.jsonl`, with per-line syllables, stress, rhyme
   scheme, and surrounding context, ready for synthetic rewriting.

## Explicit-content labels and detection

```bash
python -m radio_ready.labels
```

```bash
python -m radio_ready.learned_lexicon
```

```bash
python -m radio_ready.classifier
```

1. **`labels`** joins song-level labels onto Genius by normalized artist + title: the Spotify
   API flag (explicit if any version is) and MGPHot expert scores (explicit ≥ 0.6, clean = 0).
   Spotify and MGPHot agree on 98% of the songs that have both. Spotify-labelled songs are
   split by artist 60/20/20; MGPHot songs form a held-out `gold` split.
   Output: `data/processed/labels/songs.parquet`.
2. **`learned_lexicon`** learns explicit words from the labels as in Fell et al. (RANLP 2019):
   importance = rate in explicit songs / rate in clean songs. It reports dictionary lookup and
   regression against a Surge-list baseline (`data/artifacts/lexicon_metrics.json`). The
   learned list partly captures rap slang (genre confound), so it is an analysis artifact,
   not the section filter.
3. **`classifier`** fine-tunes DistilRoBERTa on one random section per song with the song's
   label, and scores a song by its highest section (MiSonGyny 2026 recipe). The section-level
   scores separate explicit sections (to rewrite) from clean ones (no-change examples).
   Output: `outputs/explicit_classifier/`.
