# Findings

One line per experiment; details in the linked files. Human vetting numbers are from
`annotations/vetting.csv` (share of proposed lines accepted as written = like + superlike).

## Data preparation

| Topic | Result |
| --- | --- |
| Explicit-word detection (MGPHot gold, 16,036 songs) | Learned word list after human vetting: macro-F1 95.4 (precision 88.9, recall 96.2). Surge list at severity >= 2 misses common profanity (shit 1.2, bitch 1.4); at >= 1 it adds everyday words (finger, sheet). Pipeline detector = vetted learned list + Surge words among learned candidates (95.2). |
| Genius sections | 100,169 explicit sections, one per artist (75% rap); cleaning maps homoglyphs, drops scrape junk, filler lines and repeated choruses. |
| Kidz Bop | 300 of 1,353 rows delete the line; most censored words are not radio-explicit (kiss, body, god). Used only as a side-by-side and evaluation set: 119 explicit sections, 50 in the pilot. |
| Official clean versions (Genius) | 421 clean/explicit song pairs; explicit lines are rewritten (643), bleeped (548), muted (224), partial (87) or different (43). 218 fully rewritten sections serve as human references. |
| ParaDeHate | 7,637 usable pairs. The GPT-4o-mini rewrites formalize dialect: AAE proportion 0.225 -> 0.125; 18% of pairs lose >= 0.2. |
| Licenses | `cmudict` Python package is GPL; replaced by reading the BSD-style CMU dictionary file directly (identical output on 29,415 lines). |

## Rewrite generation

| Method | Result |
| --- | --- |
| **free v2** (`free.py`, 7B, section context, stock swaps named) | **Best local method: 66% accepted as written, 10% disliked (113 lines)**, close to Gemini (73% / 13%) and better than Claude (64% / 24%). Chosen for the training data. [free_prompt.md](free_prompt.md) |
| free v3 (mined stock swaps per explicit word) | Stock swaps in rewritten lines: 7 of 20 against 12 of 22 for v2 on the same sections. Running on Genius sections 118-817; not vetted yet. |
| free v1 | 61% accepted, 12% disliked. Naming the stock swaps (v2) cut them from 11 to 4 of 27 lines. |
| sing v2 (`sing.py`, 7B meaning draft + sing pass) on the human-list lines | 62 of 101 lines proposed (55 drafts, 7 sung); the draft wins unless it breaks rhyme or meter. The draft sometimes returns another line of the stanza; candidates must now keep meaning to the original and not copy a line. Not vetted yet. |
| solid (`rewrite.py`, 7B draft + 14B) | 53% accepted, 21% disliked; proposes far fewer lines (8 of 27 on Genius). |
| `generate.py` whole section (v1) | Left explicit words in 37% of pilot sections and changed 27% of clean lines. |
| `generate.py` marked lines (v3, v5) | Clean in 90% of pilot sections, close syllables and rhyme; 44% accepted when vetted. The 7B standardizes dialect (gon' -> 'll). |
| `generate.py` lines + neighbors (v4) | Freer, but shuffled or invented lines and normalized dialect in neighbor lines. |
| Qwen3-14B (4-bit, embeddings on CPU) | Best meter and rhyme on the pilot, but often returns the explicit line unchanged (7 of 10 clean). |
| Best-of-N with ParaDeHate filters | Used by `generate.py` and `free.py`; the filters remove explicit, toxic and off-meaning candidates. |
| Matching to real clean Genius lines | Retrieved lines rarely keep the meaning (cosine 0.35-0.6); a 7B word-replacement draft does (0.75-0.95). |
| Rhyme swap (`rhyme_swap.py`) | 17 usable of 67 pilot lines; 27% accepted, 63% disliked. |
| Noise styles (explicit words replaced by made-up sounds) | 32% accepted, 63% disliked. |
| Few-shot full sections (real clean versions as examples) | 1 of 10 pilot sections clean for both 7B and 14B: the models copied the section. Dropped. |
| Commercial windows (3-line windows, Claude) | 37% accepted, 53% disliked. |
