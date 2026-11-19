"""Lyric text utilities: normalization, ad-libs, syllables, stress, end rhyme."""

import re
from functools import lru_cache

import pronouncing

WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*'?")
PARENTHETICAL = re.compile(r"\(([^()]*)\)")
VOWEL_GROUP = re.compile(r"[aeiouy]+")


# Cyrillic/Greek look-alikes Genius sometimes substitutes for Latin letters.
HOMOGLYPHS = str.maketrans(
    "аеорсухіјАЕОРСХ"
    "ВНКМТѕЅοΟ",
    "aeopcyxijAEOPCXBHKMTsSoO",
)


def normalize(text: str) -> str:
    """Fix homoglyphs, encoding damage and curly punctuation; collapse spaces."""
    text = text.translate(HOMOGLYPHS)
    text = text.replace("�", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"').replace(" ", " ")
    text = text.replace(" ", " ").replace(" ", " ").replace("​", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def words(text: str) -> list[str]:
    return WORD.findall(text)


def split_adlibs(line: str) -> tuple[str, list[str]]:
    """Remove parenthetical ad-libs, e.g. 'Pour up (Drank)' -> ('Pour up', ['Drank'])."""
    adlibs = [a.strip() for a in PARENTHETICAL.findall(line) if a.strip()]
    main = re.sub(r"\s+", " ", PARENTHETICAL.sub(" ", line))
    main = re.sub(r"\s+([,.!?;:])", r"\1", main).strip(" ,")
    return main, adlibs


def _lookup_word(word: str) -> str:
    """Map slang spellings to dictionary forms: runnin' -> running."""
    w = word.lower().strip("'")
    if word.lower().endswith("in'") or (w.endswith("in") and not pronouncing.phones_for_word(w)):
        if pronouncing.phones_for_word(w + "g"):
            return w + "g"
    return w


@lru_cache(maxsize=200_000)
def word_syllables(word: str) -> int:
    phones = pronouncing.phones_for_word(_lookup_word(word))
    if phones:
        return pronouncing.syllable_count(phones[0])
    w = word.lower().strip("'")
    count = len(VOWEL_GROUP.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(count, 1)


def syllables(line: str) -> int:
    return sum(word_syllables(w) for w in words(line))


@lru_cache(maxsize=200_000)
def word_stress(word: str) -> str:
    """CMU stress digits (0 unstressed, 1 primary, 2 secondary); '?' per syllable if unknown."""
    phones = pronouncing.phones_for_word(_lookup_word(word))
    if phones:
        return pronouncing.stresses(phones[0])
    return "?" * word_syllables(word)


def stress(line: str) -> str:
    return "".join(word_stress(w) for w in words(line))


@lru_cache(maxsize=200_000)
def rhyme_key(word: str) -> str:
    """Rhyming part (last stressed vowel onward); spelling fallback for unknown words."""
    phones = pronouncing.phones_for_word(_lookup_word(word))
    if phones:
        return re.sub(r"\d", "", pronouncing.rhyming_part(phones[0]))
    w = word.lower().strip("'")
    groups = list(VOWEL_GROUP.finditer(w))
    return w[groups[-1].start():] if groups else w


@lru_cache(maxsize=200_000)
def assonance_key(word: str) -> str:
    """Vowel of the last stressed syllable; matches slant rhymes common in rap (shots/notch)."""
    phones = pronouncing.phones_for_word(_lookup_word(word))
    if phones:
        part = pronouncing.rhyming_part(phones[0]).split()
        return re.sub(r"\d", "", part[0]) if part else ""
    groups = VOWEL_GROUP.findall(word.lower())
    return groups[-1] if groups else ""


def end_word(line: str) -> str:
    found = words(line)
    return found[-1].lower() if found else ""


def end_rhyme_kept(original: str, rewrite: str) -> bool:
    a, b = end_word(original), end_word(rewrite)
    return bool(a and b) and (a == b or rhyme_key(a) == rhyme_key(b))
