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
  scoring.py          similarity, toxicity and fluency models
  prompts.py          rewrite prompts and answer parsers
  generate.py         local-model rewrites with candidate selection
  pairs.py            SFT pairs from generations
  commercial.py       copy-paste table for commercial LLMs
  compare.py          side-by-side scores of rewrite systems
  text.py             normalization, syllables, stress, rhyme
  cmu.py              CMU Pronouncing Dictionary lookups (reads data/raw/cmudict)
  paths.py            file locations
  pipeline.py         runs all steps in order
  llm_judgment/       LLM-judged artifacts, not used by the pipeline
```

## Quick start

From the repository root:

```bash
python -m venv .venv && source .venv/Scripts/activate && pip install -r requirements.txt && pip install -e . && python -m radio_ready.pipeline
```

On Linux/macOS use `source .venv/bin/activate`. The full run downloads ~4 GB and takes roughly
45 minutes; later runs reuse downloaded files.

## Pipeline

```bash
python -m radio_ready.pipeline
```

Runs every step below in order and stops at the first failure. Options:
`--from STEP`, `--to STEP`, `--only STEP [STEP ...]`, `--with-classifier` (step 4 is skipped
by default; no other step depends on it). Example, after editing the vetting file:

```bash
python -m radio_ready.pipeline --from learned_lexicon
```

| Step | Module | Inputs | Outputs |
| --- | --- | --- | --- |
| 1 | `download` | — | `data/raw/*`, model cache |
| 2 | `labels` | Genius, Spotify, MGPHot | `data/processed/labels/songs.parquet` |
| 3 | `learned_lexicon` | labels, Surge, vetting | `data/artifacts/learned_explicit_words.csv`, `lexicon_metrics.json` |
| 4 | `classifier` | labels | `outputs/explicit_classifier/` |
| 5 | `kidzbop` | Kidz Bop, Genius, step 3 | `data/artifacts/kidzbop_*.csv`, `data/processed/kidzbop/{sections,pilot}.jsonl` |
| 6 | `genius` | Genius, Kidz Bop, step 3 | `data/processed/genius_sections.jsonl` |
| 7 | `aae` | step 6, TwitterAAE | adds `aae_proportion` to step 6 output |
| 8 | `commercial` | steps 5, 6 | `data/commercial/{prompt.txt,commercial_table.csv}` |

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
refrain, it has 4–16 lines, and it contains a detected explicit word. Every artist with an
eligible section is kept: one random eligible song per artist, one random eligible section from
it. Kidz Bop songs are excluded.

`data/processed/genius_sections.jsonl` has one JSON object per line, one section per artist.
Example: Tink, "Ride It", Verse 3 (Dej Loaf).

| Field | Meaning | Example |
| --- | --- | --- |
| `id` | section id (hash of artist, title, section index) | `"793b09d47609"` |
| `artist` | Genius artist | `"Tink"` |
| `title` | song title | `"Ride It"` |
| `tag` | Genius genre tag | `"rb"` |
| `year` | release year | `2019` |
| `section_index` | position of the section in the song (0-based) | `6` |
| `n_sections` | sections in the song after cleaning | `8` |
| `section_header` | original header text | `"Verse 3: Dej Loaf"` |
| `section_type` | normalized type | `"verse"` |
| `performers` | names after the colon in the header | `["Dej Loaf"]` |
| `repeats` | times the section appears in the song | `1` |
| `lines` | lyric lines to rewrite, ad-libs removed | `["Put that pussy on Versace plates", "Make me feel some type of way", ...]` |
| `syllables` | syllables per line | `[9, 7, 8, 9, 9, 9, 9, 9]` |
| `stress` | CMU stress per line (1 primary, 2 secondary, 0 unstressed, ? unknown word) | `["111010101", "1111111", "??111101", ...]` |
| `rhyme_scheme` | end-rhyme letter per line: stressed vowel plus the final vowel of multi-syllable rhymes, trailing fillers ("yeah") skipped, weak tails ("love me") joined | `"AAAAAAAA"` |
| `adlibs` | parenthetical ad-libs removed from the lines | `["Right, um", "Type of way", "Wide awake", ...]` |
| `explicit_words` | detected explicit tokens, one per occurrence | `["pussy", "pussy", "fuck", "bitch"]` |
| `eligible_sections_in_song` | eligible sections in this song | `1` |
| `artist_eligible_songs` | artist's songs with an eligible section | `91` |
| `context_before` | up to 8 lines of the previous section | `["Ride all night", "Oh, I wanna ride all night", ...]` |
| `context_after` | up to 8 lines of the next section | `["Let me ride it, let me ride it", "Ride all night", ...]` |
| `aae_proportion` | AAE proportion (step 7; null if too few known tokens) | `0.629` |

<details>
<summary>Full record</summary>

```json
{
  "id": "793b09d47609",
  "artist": "Tink",
  "title": "Ride It",
  "tag": "rb",
  "year": 2019,
  "section_index": 6,
  "n_sections": 8,
  "section_header": "Verse 3: Dej Loaf",
  "section_type": "verse",
  "performers": [
    "Dej Loaf"
  ],
  "repeats": 1,
  "lines": [
    "Put that pussy on Versace plates",
    "Make me feel some type of way",
    "Nightcap, it got him wide awake",
    "Pussy drippin', 'bout to ride his wave",
    "She's so bossy, she intimidate",
    "She so bossy, asked him on a date",
    "Might just fuck him on the interstate",
    "Told him he should give his bitch a break"
  ],
  "syllables": [
    9,
    7,
    8,
    9,
    9,
    9,
    9,
    9
  ],
  "stress": [
    "111010101",
    "1111111",
    "??111101",
    "101011111",
    "111012102",
    "111011101",
    "111110201",
    "111111101"
  ],
  "rhyme_scheme": "AAAAAAAA",
  "adlibs": [
    "Right, um",
    "Type of way",
    "Wide awake",
    "Intimidate",
    "So good",
    "Fuck you on the interstate",
    "Give it a break"
  ],
  "explicit_words": [
    "pussy",
    "pussy",
    "fuck",
    "bitch"
  ],
  "eligible_sections_in_song": 1,
  "context_before": [
    "Ride all night",
    "Oh, I wanna ride all night",
    "Ride all night",
    "Oh, I wanna ride all night"
  ],
  "context_after": [
    "Let me ride it, let me ride it",
    "Ride all night",
    "Oh, I wanna ride all night",
    "Ride all night",
    "Oh, I wanna ride all night"
  ],
  "artist_eligible_songs": 91,
  "aae_proportion": 0.629
}
```

</details>

`data/processed/kidzbop/sections.jsonl` uses the same fields plus `kidz_bop_artist`,
`kidz_bop_song`, `has_kidz_bop_edit`, `kidz_bop_lines`, `kidz_bop_edits`, `split`, `pilot`.

### 7. AAE proportion

Proportion of tokens from the African American English topic of the TwitterAAE model
(Blodgett et al. 2016). `aae.vernacular_shift(scorer, original, rewrite)` returns the
proportion before and after a rewrite.

## Rewrite generation

`generate.py` rewrites sections with a local model; `pairs.py` turns the results into SFT pairs.
Prompts are written out in full in `prompts.py`.

```bash
# pilot (Kidz Bop sections, default input)
python -m radio_ready.generate --model Qwen/Qwen3-14B --cpu-embeddings --style stanza-free --samples 5
# Genius sections for SFT pairs (--limit N for a partial run; interrupted runs resume)
python -m radio_ready.generate --model Qwen/Qwen3-14B --cpu-embeddings --style stanza-free --samples 5   --input data/processed/genius_sections.jsonl
```

Qwen2.5-7B-Instruct (default) and Qwen3-14B load in 4-bit (NF4, bitsandbytes); on an 8 GB GPU
Qwen3-14B needs `--cpu-embeddings`. Prompt styles (`--style`):

| Style | Request | Hints in the prompt |
| --- | --- | --- |
| `surgical` (default) | one per line with a detected word, 4 lines above and below as context | detected words |
| `surgical-free` | as `surgical`; the whole line may be reworded | detected words, syllable count |
| `stanza` | one per 4-line stanza with a detected word | detected words |
| `stanza-free` | as `stanza`; lines may be reworded | detected words |
| `plain` | whole section; the model decides what radio would censor | none |
| `annotated` | whole section with syllable counts, rhyme letters and REWRITE marks | all |

Every style asks to keep the meaning, rhythm and syllable count, the end rhyme, the artist's
voice, and slang, dialect, grammar and spelling. `--rhyme` (free styles) makes the model write
the rhyme pattern before its answer; in the pilot it copied the example pattern and did not
help.

`--samples N` draws N candidates per request (temperature 1.0, top-p 0.9: top-p cuts the
unlikely tail, the temperature varies the wording) and keeps the best:

1. Filters (ParaDeHate): no detected explicit word, toxicity < 0.9, similarity of the changed
   lines to their originals ≥ 0.70.
2. Ranking: fewest broken end rhymes, every line within one syllable, then the highest
   similarity × fluency, where fluency is the CoLA acceptability of each changed line relative
   to its original (the ParaDetox fluency model). It is relative because lines in dialect score
   low on both sides; it catches nonsense such as "I should be a sad stuff".

Output: `outputs/generations/<name>.jsonl`, one record per section with the merged lines,
`valid` (every request gave a usable answer), `passed` (every chosen candidate passed the
filters) and every candidate with its scores. Genius sections reserved for commercial LLMs are
skipped.

### Synthetic pairs

`rewrite.py`: every explicit line gets a Qwen2.5-7B draft that replaces only the explicit words
(no rhyme or syllable hints; one-for-one word swaps away from the explicit words are put back,
so dialect stays as written). Hard lines (explicit end word, several explicit words, or a draft
that fails a check) are rewritten by Qwen3-14B, which sees only the drafts, scored on meaning
against the draft and on syllables and end rhyme against the original.

```bash
python -m radio_ready.rewrite [--input FILE] [--limit N]    # pilot by default
```

`pairs.py` turns each source into pairs. Sources: the `solid_*` rewrite runs, the commercial
models (`commercial_*`, from `python -m radio_ready.commercial --export`; they need your
approval and may use the reserved commercial-evaluation sections) and `noise` (every
noise-style generation; needs approval). In `pairs/`:

| File | Committed | Contents |
| --- | --- | --- |
| `<source>.patch.jsonl` | yes | ids, split, how the rewrite was made, your verdicts, and a word-level diff of each changed line (`linediff.py`); no lyrics |
| `<source>.jsonl` | no (lyrics) | full pairs: `original`, `rewrite`, `changed`, `split`, model and prompt version; no training features |
| `<source>.md` | no (lyrics) | side-by-side view |
| `final.*` | patch yes | every source combined; identical pairs (same section, same rewrite) kept once |

A version of a section becomes a pair when it passed the filters (or you approved all its
lines), at least one line changed, nothing explicit is left, and none of its lines is disliked.
With the section records built (`download`, `kidzbop`, `genius`), the full pairs come back from
the committed patches:

```bash
python -m radio_ready.pairs --rebuild
python -m radio_ready.pairs --all --combine   # export every source, then pairs/final
```

### Human vetting

Double-click `vet.bat`: a start window lists every source with how many of its lines are
vetted, plus "All sources". Vet opens one; Export pairs writes its pairs (new runs appear
there); Combine writes `pairs/final`.

One line at a time: the original with its explicit words highlighted and the rewrite in an
edit field. Edit in place if needed, then Enter (like), Ctrl+Enter (superlike) or Esc (dislike);
Up/Down go back or skip; Tab shows the 4-line stanza, original and rewritten (it closes on
every new line). Alt+1-8 puts a suggestion in place of the explicit word: clean words that
rhyme with the explicit word itself, with its syllable count, never a sound-alike of it
(`rhymes.py`: CMU rhymes of words common in clean lyrics, ranked by masked-LM fit in the slot
and meaning; `python -m radio_ready.rhymes --build` once). Distinctive rewrites come first,
generic ones ("stuff", "things") last.

All verdicts go to one sheet, `annotations/vetting.csv` (committed, no lyrics: rewrites and
edits are diffs against the original line). It has one row per line and proposed rewrite, so a
rewrite several sources proposed (the 7B/14B run, a commercial model, noise versions) is vetted
once and counts for all of them; `sources` lists who proposed it.

### Commercial LLMs (copy-paste)

The pipeline's `commercial` step writes one generic prompt, `data/commercial/prompt.txt`, and
one table, `data/commercial/commercial_table.csv`, with a row per section: every Kidz Bop pilot
section plus 50 Genius sections chosen by a hash of the section id (the local model never
rewrites those Genius sections). The prompt does not mark lines, syllables or rhyme: the model
decides what radio would censor.

To fill it:
1. paste `prompt.txt` at the start of a chat (or in front of each section),
2. for each row, copy the `section` cell into the chat,
3. paste the answer into that model's column (`output_chatgpt`, `output_claude`,
   `output_gemini`; rename a column to the exact model, e.g. `output_gemini-2.5-pro`, or add
   one).

The prompt asks for the whole section on one line with lines separated by ` / `, so each
answer fits in one cell. No import step: `compare` reads the filled cells directly and reports
answers it cannot use (wrong number of lines). Re-running the step only adds new rows and model
columns and never changes filled cells; close the file in Excel first. Options:
`python -m radio_ready.commercial --genius-sections 100 --models gpt-5`.

### Side-by-side comparison

```bash
python -m radio_ready.compare
```

Scores Kidz Bop, every local generation file and every commercial model on the same sections:
explicit words left, lines changed, syllable match and end rhyme kept (over changed lines only,
so copying a line is not counted as a success), clean lines copied unchanged, benign slang
kept, LaBSE similarity, AAE delta. The evaluation models (LaBSE, s-nlp toxicity) differ from
the models that select candidates, so a rewrite is not scored by the model that picked it. Outputs in `outputs/comparison/<name>/`: `scores.csv`, `summary.csv`,
`side_by_side.md`, and a blinded `ranking.csv` with `ranking_key.csv` for human ranking.

## Scoring models

| Model | Use | Threshold |
| --- | --- | --- |
| `sentence-transformers/all-distilroberta-v1` | similarity filter | cosine ≥ 0.70 (ParaDeHate) |
| `unitary/unbiased-toxic-roberta` | toxicity filter | rewrite must score < 0.9 (ParaDeHate) |
| `cointegrated/roberta-large-cola-krishna2020` | fluency, candidate ranking (drop relative to the original line) | — |
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
| [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict) (`cmudict.dict`, pinned commit) | BSD-2-Clause-style, © 1993-2015 Carnegie Mellon University |

### Models

| Model | License |
| --- | --- |
| `sentence-transformers/all-distilroberta-v1` | Apache-2.0 |
| `unitary/unbiased-toxic-roberta` | Apache-2.0 |
| `sentence-transformers/LaBSE` | Apache-2.0 |
| `s-nlp/roberta_toxicity_classifier` | OpenRAIL++ |
| `distilroberta-base` | Apache-2.0 |
| `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3-14B` | Apache-2.0 |
| `cointegrated/roberta-large-cola-krishna2020` | see model card |

### Code

| Component | License |
| --- | --- |
| `aae.py` inference, adapted from `slanglab/twitteraae` `predict.py` | MIT, © 2019 Su Lin Blodgett and Brendan O'Connor (notice in file) |

## References

- Blodgett, Green, O'Connor. Demographic Dialectal Variation in Social Media: A Case Study of
  African-American English. EMNLP 2016.
- Fell, Cabrio, Corazza, Gandon. Comparing Automated Methods to Detect Explicit Content in Song
  Lyrics. RANLP 2019.
- Logacheva et al. ParaDetox: Detoxification with Parallel Data. ACL 2022.
- Yuan et al. LLM in the Loop: Creating the ParaDeHate Dataset for Hate Speech
  Detoxification. Findings of IJCNLP-AACL 2025.
