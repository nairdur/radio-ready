"""Replace detected explicit words with noise, so a model rewrites lines without seeing them.

Modes (one replacement per explicit token, same syllable count as the token):
  vowels     keep the word's vowel letters and final consonants, replace the other consonants
             ("fucking" -> e.g. "zuping"): keeps syllables, vowel sounds and most of the rhyme
  gibberish  random consonant-vowel syllables ("fucking" -> e.g. "boke"): keeps syllables only
  blank      a placeholder "___ (2 syllables)": keeps syllables only, no sound

Noise words are shown to the model in curly braces. A noise word is redrawn if the explicit-word
detector would flag it or it sounds like a Surge-list or detector word (sound_key). Draws are seeded by section id, so prompts are reproducible.
"""

import random
import re

from radio_ready.text import word_syllables

MODES = ("vowels", "gibberish", "blank")
ONSETS = ["b", "d", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "gl", "pl",
          "tr", "dr", "kl", "sp", "st"]
VOWELS = ["a", "e", "i", "o", "u", "ay", "ee", "oo"]
VOWEL_GROUP = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _vowels(word: str, rng: random.Random) -> str:
    groups = list(VOWEL_GROUP.finditer(word))
    if not groups:
        return _gibberish(word, rng)
    out = []
    for g in groups:
        out.append(rng.choice(ONSETS) + g.group().lower())
    coda = word[groups[-1].end():].lower()
    return "".join(out) + re.sub(r"[^a-z']", "", coda)


def _gibberish(word: str, rng: random.Random) -> str:
    return "".join(rng.choice(ONSETS) + rng.choice(VOWELS)
                   for _ in range(word_syllables(word)))


def sound_key(word: str) -> str:
    """Rough spelling-to-sound key: "klit" and "clit" share one."""
    w = re.sub(r"[^a-z]", "", word.lower())
    for a, b in (("ph", "f"), ("ck", "k"), ("c", "k"), ("q", "k"), ("z", "s"), ("x", "ks"),
                 ("ee", "i"), ("y", "i")):
        w = w.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", w)


_BANNED_SOUNDS: set[str] | None = None


def banned_sounds(detector) -> set[str]:
    """Sound keys of Surge entries and canonical forms (all severities) and detector words."""
    global _BANNED_SOUNDS
    if _BANNED_SOUNDS is None:
        import pandas as pd

        from radio_ready import paths

        surge = pd.read_csv(paths.SURGE_CSV)
        forms = ["text", "canonical_form_1", "canonical_form_2", "canonical_form_3"]
        words = {w for col in forms for w in surge[col].dropna().astype(str).str.lower()
                 if " " not in w}
        _BANNED_SOUNDS = {sound_key(w) for w in words | set(detector.words)}
        _BANNED_SOUNDS.discard("")
    return _BANNED_SOUNDS


def is_safe(fake: str, word: str, detector) -> bool:
    if fake.lower() == word.lower() or detector is None:
        return fake.lower() != word.lower()
    return not detector.find(fake) and sound_key(fake) not in banned_sounds(detector)


def noise_word(word: str, mode: str, rng: random.Random, detector=None) -> str:
    if mode == "blank":
        n = word_syllables(word.replace("*", "u"))
        return f"___ ({n} syllable{'s' if n != 1 else ''})"
    for _ in range(50):
        fake = (_vowels if mode == "vowels" else _gibberish)(word.replace("*", "u"), rng)
        if is_safe(fake, word, detector):
            return fake
    while True:
        fake = _gibberish(word, rng)
        if is_safe(fake, word, detector):
            return fake


def noise_lines(lines: list[str], record_id: str, mode: str, detector) -> list[str]:
    """Lines with every detected explicit token replaced by {noise}."""
    rng = random.Random(f"{record_id}:{mode}")
    out = []
    for line in lines:
        for token in sorted(set(detector.find(line)), key=len, reverse=True):
            pattern = re.compile(rf"(?<![A-Za-z']){re.escape(token)}(?![A-Za-z'])", re.IGNORECASE)
            line = pattern.sub(lambda m: "{" + noise_word(m.group(), mode, rng, detector) + "}",
                               line)
        out.append(line)
    return out
