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
src/         project code
```

## Setup

```bash
python -m venv .venv
```

```bash
pip install -r requirements.txt
```

Kaggle downloads need credentials in `~/.kaggle/kaggle.json`.

## Data sources

- [ParaDetox](https://huggingface.co/datasets/s-nlp/paradetox)
- [The Pudding: Kidz Bop censored lyrics](https://github.com/the-pudding/data/tree/master/kidz-bop)
- [Genius lyrics (cleaned)](https://huggingface.co/datasets/theelderemo/genius-lyrics-cleaned)
- [Cornell genius-expertise](https://www.cs.cornell.edu/~arb/data/genius-expertise/)
- Kaggle: `carlosgdcj/genius-song-lyrics-with-language-information`
