# Finding: a freer section-context prompt for Qwen2.5-7B, judged by human vetting

**Question.** Can a single 7B model, given the whole 4-line section and freedom to reword the flagged
lines, produce radio edits a human accepts as written, and does naming the generic replacements make
them less bland?

**Setup** (`src/radio_ready/free.py`, Qwen2.5-7B-Instruct, 4-bit). One request per 4-line block with
a flagged line (a line with a detected explicit word). The prompt marks the flagged lines with their
censored words, the original syllable count and the end words they rhyme with, and allows the line to
be reworded freely as long as the censored words are gone and meaning, voice, slang, syllables and
end rhyme are kept. Unflagged lines are always restored as written. Six lines the annotator liked
(never from the song being rewritten) are given as style examples. Per block: one greedy answer and 4
samples (temperature 1.0, top-p 0.9), scored per line; a line still holding a detected explicit word,
a homophone or spelling variant of the censored word (the detector misses "ho" for "hoe"), a
non-word (not in the CMU dictionary, the original line or clean lyrics 20+ times; slang such as
"homies" passes) or no change is never proposed. Lines with no clean candidate get one retry.

- **v1**: the prompt above.
- **v2**: adds the *stock swaps* from the vetting sheet (single-word replacements used 5+ times for
  an explicit word: girl, stuff, homie, mess, hell, damn, lady, chick, forget, nonsense, folks, ...),
  names the top 12 as what not to do and asks to rework the phrase with fresh words or an image that
  fits the verse. Rewrites containing a stock swap rank below those without; style examples contain
  none.

The most frequent swaps in the sheet before v2 were shit -> stuff (44), bitch -> girl (44),
nigga -> homie (20), fuck -> mess (18), bitch -> lady (18), bitch -> chick (17).

**Automatic checks** (Genius sections 11-20, 27 flagged lines; solid = the two-model `rewrite.py`
pipeline, 7B draft + Qwen3-14B for hard lines):

| Run | Proposed | Clean | Syllables within 1 | End rhyme kept | Stock swaps |
| --- | --- | --- | --- | --- | --- |
| solid | 8 | 8 | 8 | 8 | 2 |
| free v1 | 25 | 25 | 21 | 20 | 11 |
| free v2 | 25 | 25 | 23 | 19 | 4 |

On Genius sections 1-10 (20 flagged lines) v1 proposed 18 (all clean), solid 14.

**Human vetting** (every proposed line judged in the vetting app: kept as written, edited by the
annotator, or disliked; lines the annotator wrote themself are not counted):

| Prompt | Lines vetted | Kept as-is | Edited | Disliked | Superliked as-is |
| --- | --- | --- | --- | --- | --- |
| Gemini (commercial) | 27 | 85% | 7% | 7% | 30% |
| **free v2** | 74 | **64%** | 27% | **9%** | 27% |
| Claude (commercial) | 179 | 62% | 16% | 22% | 36% |
| free v1 | 43 | 49% | 35% | 16% | 12% |
| solid, Kidz Bop sections | 132 | 45% | 30% | 24% | 19% |
| noise styles | 151 | 26% | 7% | 67% | 16% |
| solid, Genius sections | 35 | 20% | 51% | 29% | 9% |

Same 10 sections, v1 against v2 (25 lines each): kept as-is 52% vs 64%, disliked 16% vs 8%,
superliked 2 vs 8.

**Finding.** Section context with freedom to reword, plus hard rejection of explicit, disguised and
non-words, makes one 7B model propose an edit for nearly every flagged line (25/27 against 8/27 for
the two-model pipeline on Genius lyrics) and roughly triples the share the annotator accepts as
written on Genius sections. Naming the stock swaps cuts them from 11 to 4 of 27 lines and is the
largest gain in human preference: v2 is accepted as written as often as Claude and disliked less
than half as often, with all candidates from a local 7B model.

**Known issues.** Shouted all-caps lines and lines with several stacked explicit words are still
weak (an all-caps expletive is replaced by an unrelated short word). Penalizing stock swaps
occasionally costs an end rhyme (19 vs 20). Some rewrites flip the meaning of the line; similarity
to the original does not catch these, the annotator does.
