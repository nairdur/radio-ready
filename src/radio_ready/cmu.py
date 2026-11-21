"""CMU Pronouncing Dictionary lookups, read directly from the dictionary file.

Data: cmusphinx/cmudict `cmudict.dict` at a pinned commit (BSD-2-Clause-style license,
Copyright (C) 1993-2015 Carnegie Mellon University), downloaded to data/raw/cmudict by
`python -m radio_ready.download cmudict`. The functions mirror the parts of the `pronouncing`
package that this project used.
"""

import re
from functools import lru_cache

from radio_ready import paths

ALTERNATE = re.compile(r"\(\d+\)$")


@lru_cache(maxsize=1)
def _dictionary() -> dict[str, list[str]]:
    if not paths.CMUDICT_FILE.exists():
        raise FileNotFoundError(
            f"{paths.CMUDICT_FILE} not found; run: python -m radio_ready.download cmudict")
    entries: dict[str, list[str]] = {}
    with open(paths.CMUDICT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            word, *phones = line.split()
            entries.setdefault(ALTERNATE.sub("", word).lower(), []).append(" ".join(phones))
    return entries


def phones_for_word(word: str) -> list[str]:
    """Pronunciations of a word (ARPAbet with stress digits), first is the most common."""
    return _dictionary().get(word.lower(), [])


def syllable_count(phones: str) -> int:
    return sum(1 for p in phones.split() if p[-1].isdigit())


def stresses(phones: str) -> str:
    """Stress digit per syllable: 0 unstressed, 1 primary, 2 secondary."""
    return "".join(p[-1] for p in phones.split() if p[-1].isdigit())


def rhyming_part(phones: str) -> str:
    """Phones from the last stressed vowel to the end."""
    parts = phones.split()
    for i in range(len(parts) - 1, -1, -1):
        if parts[i][-1] in "12":
            return " ".join(parts[i:])
    return phones
