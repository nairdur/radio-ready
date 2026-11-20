"""Rewrite prompt shared by every generator (local 7B, commercial LLMs), and its parsers.

Lines of a section:
  target    a line in which the explicit-word detector (radio_ready.explicit, the same one that
            selects sections) finds a word; it must be rewritten
  neighbor  a line next to a target line, or a line within NEIGHBOR_RHYME_SPAN lines of a target
            that shares its rhyme letter; it may be rewritten when the new target line needs it
            for rhyme or flow
  other     never rewritten
The model sees the whole section with syllable counts and rhyme letters, plus the lines around
it, and returns every line it rewrote. Unreturned lines are copied from the original.

The full prompt text is below (SYSTEM_PROMPT, INSTRUCTIONS, USER_TEMPLATE, BATCH_TEMPLATE).
"""

import re
from functools import lru_cache

import pandas as pd

from radio_ready import paths
from radio_ready.text import tokenize

PROMPT_VERSION = "v5"
NEIGHBOR_RHYME_SPAN = 2

SYSTEM_PROMPT = (
    "You are a professional lyricist who writes radio edits of explicit songs. A radio edit "
    "must sound like the same song: the same meaning, story, attitude, wordplay, rhythm and "
    "rhyme, written so that nothing in it would be bleeped or flagged."
)

INSTRUCTIONS = """\
Lines marked REWRITE contain explicit words, listed after them. Change as little as possible: \
replace the explicit words, and only the words right next to them if grammar or rhythm requires \
it, with clean words that keep the meaning, the syllable count and the rhyme. Leave the rest of \
the line exactly as written.

Slang, dialect and nonstandard spelling are not explicit. Never rewrite, correct or clean up \
slang words, dropped letters (runnin', gon', ya), nonstandard grammar (they was, it don't, \
this happen) or spelling. Words listed after "keep as written" are slang that must stay \
exactly as they are.

Lines marked "may rewrite" rhyme with or run into a REWRITE line. Change one only if your \
REWRITE line had to end on a new word and the rhyme would otherwise break; then change only its \
end word, or the fewest words needed. Otherwise leave it unchanged and do not list it. Unmarked \
lines are context only.

For every line you change:
1. Meaning and context: keep what the line says, its role in the section and the story of the \
song.
2. Meter: keep the syllable count in brackets (at most one syllable more or less) and the stress \
pattern, so the line fits the same flow and beat.
3. Rhyme: keep a clean end word as it is. If the end word is explicit, replace it with a word \
that rhymes with the lines that share its rhyme letter (exact or slant rhyme). Keep internal \
rhymes and repeated sounds.
4. Sound: keep the vowel sounds of the replaced words where possible, but never use a \
sound-alike that would be heard as an explicit word (for example "fudge", "shoot", "witch").
5. Style: keep the artist's voice, word choice and attitude.
6. Clean: no profanity, slurs or sexual terms, including misspelled, censored, split or \
abbreviated forms (f*ck, sh!t, effin, god damn), and no innuendo that keeps the explicit \
meaning."""

USER_TEMPLATE = """\
Write the radio edit of this section of "{title}" by {artist}.

{instructions}

{section}

Answer with every line you changed: all REWRITE lines ({targets}) and any "may rewrite" lines \
you had to change. Put one line per row, starting with its line number and a period, for example \
"{example}. ...". Do not add anything else."""

BATCH_TEMPLATE = """\
{system}

Write radio edits of the song sections below.

{instructions}

There are {n} sections. Each starts with ### and an id. For each section, answer with the same \
### id line, followed by every line you changed in that section: all its REWRITE lines and any \
"may rewrite" lines you had to change, one per row, each starting with its line number and a period. \
Answer in one code block and add nothing else.

{sections}"""

SECTION_TEMPLATE = """\
Lines before this section (context):
{before}

Section "{header}", rhyme scheme {scheme}:
{lines}

Lines after this section (context):
{after}"""

LINE_TEMPLATE = "{n}. {text}   [{syllables} syllables, rhyme {rhyme}]"
TARGET_MARK = "   <-- REWRITE (explicit: {words})"
NEIGHBOR_MARK = "   <-- may rewrite"
SLANG_MARK = " (keep as written: {words})"

LINE = re.compile(r"^\s*(\d+)[.)]\s?(.*)$")
SECTION_HEADER = re.compile(r"^\s*#{2,}\s*(\S+)\s*$")


def target_lines(record: dict, detector) -> dict[int, list[str]]:
    """{line index: explicit words found} for the lines that must be rewritten."""
    found = {}
    for i, line in enumerate(record["lines"]):
        hits = detector.find(line)
        if hits:
            found[i] = sorted(set(hits))
    return found


def neighbor_lines(record: dict, targets) -> list[int]:
    """Adjacent lines and nearby rhyme partners of target lines that may be rewritten."""
    scheme, n = record["rhyme_scheme"], len(record["lines"])
    found = set()
    for t in targets:
        found |= {t - 1, t + 1}
        found |= {j for j in range(max(0, t - NEIGHBOR_RHYME_SPAN),
                                   min(n, t + NEIGHBOR_RHYME_SPAN + 1)) if scheme[j] == scheme[t]}
    return sorted(i for i in found if 0 <= i < n and i not in targets)


@lru_cache(maxsize=1)
def benign_slang() -> frozenset[str]:
    """Words marked not_explicit in the human vetting file."""
    if not paths.VETTING_CSV.exists():
        return frozenset()
    vet = pd.read_csv(paths.VETTING_CSV, dtype=str, keep_default_na=False)
    keep = vet["verdict"].str.strip().str.lower() == "not_explicit"
    return frozenset(vet.loc[keep, "word"].str.strip().str.lower())


def slang_in(line: str) -> list[str]:
    return sorted({t for t in tokenize(line) if t in benign_slang()})


def section_text(record: dict, targets: dict[int, list[str]]) -> str:
    neighbors = set(neighbor_lines(record, targets))
    rows = []
    for i, (line, syl) in enumerate(zip(record["lines"], record["syllables"])):
        row = LINE_TEMPLATE.format(n=i + 1, text=line, syllables=syl,
                                   rhyme=record["rhyme_scheme"][i])
        if i in targets:
            row += TARGET_MARK.format(words=", ".join(targets[i]))
        elif i in neighbors:
            row += NEIGHBOR_MARK
        if (i in targets or i in neighbors) and slang_in(line):
            row += SLANG_MARK.format(words=", ".join(slang_in(line)))
        rows.append(row)
    return SECTION_TEMPLATE.format(
        before="\n".join(record.get("context_before", [])[-4:]) or "(start of song)",
        header=record["section_header"] or record["section_type"],
        scheme=record["rhyme_scheme"],
        lines="\n".join(rows),
        after="\n".join(record.get("context_after", [])[:4]) or "(end of song)",
    )


def target_numbers(targets) -> str:
    return ", ".join(str(i + 1) for i in sorted(targets))


def single_prompt(record: dict, targets: dict[int, list[str]]) -> list[dict]:
    """Chat messages for one section."""
    user = USER_TEMPLATE.format(
        title=record["title"], artist=record["artist"], instructions=INSTRUCTIONS,
        section=section_text(record, targets), targets=target_numbers(targets),
        example=min(targets) + 1)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def batch_prompt(items: list[tuple[dict, dict[int, list[str]]]]) -> str:
    """One copy-paste prompt for several (record, targets) sections."""
    sections = [f"### {r['id']}\n\"{r['title']}\" by {r['artist']}. REWRITE lines: "
                f"{target_numbers(t)}\n\n{section_text(r, t)}" for r, t in items]
    return BATCH_TEMPLATE.format(system=SYSTEM_PROMPT, instructions=INSTRUCTIONS,
                                 n=len(items), sections="\n\n".join(sections))


def clean_response_line(text: str) -> str:
    text = text.split("<--")[0]
    text = re.sub(r"\s*\[[^\]]*syllables[^\]]*\]\s*$", "", text)
    text = re.sub(r"\s*\[\d+\]\s*$", "", text)
    text = text.replace("**", "").replace("__", "")  # markdown emphasis
    return re.sub(r"^[.\s]+", "", text).strip()  # "4. .And ..." -> "And ..."


def parse_rewrites(text: str, record: dict, targets) -> dict[int, str] | None:
    """{line index: rewrite} for every target line and any returned neighbor line.

    None if a target line is missing or empty. Returned lines that are neither targets nor
    neighbors are ignored.
    """
    allowed = {i + 1 for i in targets} | {i + 1 for i in neighbor_lines(record, targets)}
    found = {}
    for raw in text.splitlines():
        m = LINE.match(raw.strip().strip("`"))
        if m and int(m.group(1)) in allowed:
            line = clean_response_line(m.group(2))
            if line:
                found.setdefault(int(m.group(1)) - 1, line)
    if not set(targets) <= set(found):
        return None
    return found


def merge(record: dict, rewrites: dict[int, str]) -> list[str]:
    """Original lines with the rewritten lines replaced."""
    lines = list(record["lines"])
    for i, text in rewrites.items():
        lines[i] = text
    return lines


def parse_batch(text: str) -> dict[str, str]:
    """Split a batch response into {section id: text of that section}."""
    sections: dict[str, list[str]] = {}
    current = None
    for raw in text.splitlines():
        m = SECTION_HEADER.match(raw.strip().strip("`"))
        if m:
            current = m.group(1)
            sections[current] = []
        elif current is not None:
            sections[current].append(raw)
    return {k: "\n".join(v) for k, v in sections.items()}
