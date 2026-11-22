"""End-rhyme thesaurus: clean words that rhyme, fit the syllable slot and keep the meaning.

Index (built once, cached in data/artifacts/rhyme_vocab.json): every CMU dictionary word that
occurs at least --min-count times in clean lyrics (lines of Genius songs with no detected
explicit word), with its syllable count and two rhyme keys:
  perfect  the rhyming part, last stressed vowel to the end (bitch -> IH CH: rich, pitch)
  slant    stressed vowel and final sound (riches, wishes, kisses -> IH-Z)

suggest(line, targets, ...) proposes words for the end of `line`: words sharing a perfect or
slant key with a target word, within one syllable of the word they replace, never explicit.
They are ranked by two scores, z-scored and added:
  fit      how likely a masked language model (distilroberta-base) finds the word in that
           slot, given the line and the line before it; this rules out function words and
           words of the wrong kind ("a sad it")
  meaning  sentence-embedding cosine of the line with the word in place to the line as given
Never proposed: the 150 most frequent lyric words (function words), words already in the line,
and sound-alikes of the explicit word (one phoneme or less apart: witch, ditch for bitch).
Targets are the end words of the line's rhyme partners, or the original end word itself when
it has no partner.

suggest_for_word(template, explicit, ...) proposes words for the slot of one explicit word
(the vetting app, radio_ready.rhyme_swap): words rhyming with the explicit word itself,
perfect, slant or assonance (stressed vowel and the kind of final sound: bitch -> wish, kiss),
never a sound-alike of it or of its last sounds (dam for goddamn), never on the Surge list or
Kidz Bop's profanity list (never_suggest). The k returned are picked to differ from each other
in meaning and sound (diverse(): maximal marginal relevance).

Run:  python -m radio_ready.rhymes --build [--min-count 20]
      python -m radio_ready.rhymes "Been through some tough times, I should be a sad one" bitch
"""

import argparse
import json
import re
from collections import Counter
from functools import lru_cache

import numpy as np

from radio_ready import cmu, paths
from radio_ready.explicit import ExplicitDetector
from radio_ready.text import _phones, rhyme_key, word_syllables

VOCAB_JSON = paths.ARTIFACTS / "rhyme_vocab.json"
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
CLEAN_LINES = [paths.PROCESSED / "match" / "corpus.jsonl",
               paths.PROCESSED / "match" / "corpus_200k.jsonl"]


def lyric_counts(max_songs: int = 60_000) -> Counter:
    """Word counts in clean lyric lines (match corpus if built, else clean Genius songs)."""
    counts: Counter = Counter()
    for path in CLEAN_LINES:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    counts.update(WORD.findall(json.loads(line)["text"].lower()))
            return counts
    from radio_ready.genius import iter_rows
    detector, n = ExplicitDetector(), 0
    for row in iter_rows({"rap", "trap", "pop", "rb", "rock", "country"}, None):
        lyrics = str(row["lyrics"])
        if detector.find(lyrics):
            continue
        counts.update(WORD.findall(lyrics.lower()))
        n += 1
        if n >= max_songs:
            break
    return counts


def build(min_count: int) -> None:
    detector, counts = ExplicitDetector(), lyric_counts()
    vocab = {}
    for word, n in counts.items():
        if n < min_count or detector.find(word) or not cmu.phones_for_word(word):
            continue
        phones = _phones(word)
        vocab[word] = {"count": n, "syllables": word_syllables(word),
                       "perfect": rhyme_key(word), "slant": slant_key(phones)}
    VOCAB_JSON.parent.mkdir(parents=True, exist_ok=True)
    VOCAB_JSON.write_text(json.dumps(vocab), encoding="utf-8")
    print(f"rhyme vocabulary: {len(vocab)} words (count >= {min_count}) -> {VOCAB_JSON}")


@lru_cache(maxsize=1)
def index() -> tuple[dict, dict, dict]:
    if not VOCAB_JSON.exists():
        raise FileNotFoundError(f"{VOCAB_JSON} missing; run: python -m radio_ready.rhymes --build")
    vocab = json.loads(VOCAB_JSON.read_text(encoding="utf-8"))
    perfect, slant = {}, {}
    for w, v in vocab.items():
        perfect.setdefault(v["perfect"], []).append(w)
        slant.setdefault(v["slant"], []).append(w)
    return vocab, perfect, slant


def slant_key(phones: list[str]) -> str:
    """Stressed vowel of the rhyme and the final sound: riches / wishes / kisses -> IH-Z."""
    stressed = [i for i, p in enumerate(phones) if p[-1:] in ("1", "2")]
    vowel = re.sub(r"\d", "", phones[stressed[-1]] if stressed else phones[-1])
    last = phones[-1]
    return f"{vowel}-{'V' if last[-1:].isdigit() else last}"


SOUND_CLASS = {**dict.fromkeys(["P", "B", "T", "D", "K", "G"], "stop"),
               **dict.fromkeys(["CH", "JH", "F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"],
                               "hiss"),
               **dict.fromkeys(["M", "N", "NG"], "nasal"), **dict.fromkeys(["L", "R"], "liquid"),
               **dict.fromkeys(["W", "Y"], "glide")}


def assonance_key(phones: list[str]) -> str:
    """Stressed vowel of the rhyme and the kind of final sound (stop, hiss, nasal, liquid):
    bitch / wish / kiss / bridge -> IH-hiss. Looser than slant_key; for lyrics, where
    assonance counts as a rhyme."""
    stressed = [i for i, p in enumerate(phones) if p[-1:] in ("1", "2")]
    vowel = re.sub(r"\d", "", phones[stressed[-1]] if stressed else phones[-1])
    last = phones[-1]
    return f"{vowel}-{'V' if last[-1:].isdigit() else SOUND_CLASS.get(last, last)}"


@lru_cache(maxsize=1)
def assonance_index() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for w in index()[0]:
        phones = _phones(w)
        if phones:
            found.setdefault(assonance_key(phones), []).append(w)
    return found


def keys(word: str) -> tuple[str, str]:
    phones = _phones(word)
    return rhyme_key(word), (slant_key(phones) if phones else "")


class SlotFit:
    """Log-probability of a word in a masked slot (distilroberta-base)."""

    def __init__(self, model: str = "distilroberta-base"):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForMaskedLM.from_pretrained(model).eval()

    def score(self, context: str, words: list[str]) -> np.ndarray:
        enc = self.tok(context.replace("<mask>", self.tok.mask_token), return_tensors="pt",
                       truncation=True, max_length=128)
        pos = (enc["input_ids"][0] == self.tok.mask_token_id).nonzero()[0, 0]
        with self.torch.no_grad():
            logp = self.torch.log_softmax(self.model(**enc).logits[0, pos], -1)
        out = []
        for w in words:
            ids = self.tok(" " + w, add_special_tokens=False)["input_ids"]
            out.append(float(logp[ids[0]]) - 1.5 * (len(ids) - 1))  # penalize split words
        return np.array(out)


_SLOT_FIT: list = []
_SLOT_LOCK = __import__("threading").Lock()


def slot_fit() -> SlotFit:
    """One masked LM per process, safe to ask for from several threads (vetting windows)."""
    with _SLOT_LOCK:
        if not _SLOT_FIT:
            _SLOT_FIT.append(SlotFit())
        return _SLOT_FIT[0]


def replace_end(line: str, word: str) -> str:
    """The line with its last word replaced (trailing punctuation and fillers kept)."""
    m = re.search(r"([A-Za-z']+)([^A-Za-z']*(?:\b(?:yeah|uh|oh|ooh|ay|ayy)\b[^A-Za-z']*)*)$",
                  line, re.IGNORECASE)
    return line[:m.start(1)] + word + m.group(2) if m else f"{line} {word}"


def end_of(line: str) -> str:
    m = re.search(r"([A-Za-z']+)[^A-Za-z']*(?:\b(?:yeah|uh|oh|ooh|ay|ayy)\b[^A-Za-z']*)*$",
                  line, re.IGNORECASE)
    return m.group(1).lower() if m else ""


def candidates(targets: list[str], n_syllables: int, exclude: set[str],
               loose: bool = False) -> list[tuple[str, str]]:
    """(word, 'perfect' | 'slant' | 'assonance') rhyming with any target, within one syllable;
    assonance (assonance_key) only when `loose`."""
    vocab, perfect, slant = index()
    found: dict[str, str] = {}
    for t in targets:
        pk, sk = keys(t)
        for w in perfect.get(pk, []):
            found.setdefault(w, "perfect")
        for w in slant.get(sk, []) if sk else []:
            found.setdefault(w, "slant")
        phones = _phones(t)
        for w in assonance_index().get(assonance_key(phones), []) if loose and phones else []:
            found.setdefault(w, "assonance")
    return [(w, kind) for w, kind in found.items()
            if w not in exclude and w not in common_words() and "'" not in w  # no possessives
            and re.search("[aeiouy]", w)  # no abbreviations (mrs)
            and w not in never_suggest()
            and abs(vocab[w]["syllables"] - n_syllables) <= 1]


MILD = {"pissed", "pissing", "butt", "butts", "damned", "crappy", "screwed"}  # in neither list


@lru_cache(maxsize=1)
def never_suggest() -> frozenset[str]:
    """Vulgar words the explicit-word detector lets through, never proposed as clean: every
    Surge AI Obscenity List entry (any severity), the words Kidz Bop treats as profanity
    (kidzbop_words.csv, category profanity: piss, crap, hell) and MILD."""
    import pandas as pd

    from radio_ready.explicit import surge_words

    found = set(MILD) | surge_words(0)
    if paths.KIDZBOP_WORDS_CSV.exists():
        kb = pd.read_csv(paths.KIDZBOP_WORDS_CSV, keep_default_na=False)
        found |= set(kb.loc[kb["category"] == "profanity", "word"].str.lower())
    return frozenset(found)


@lru_cache(maxsize=1)
def common_words(n: int = 150) -> frozenset[str]:
    """The n most frequent lyric words: function words, never a useful end word."""
    vocab = index()[0]
    return frozenset(sorted(vocab, key=lambda w: -vocab[w]["count"])[:n])


def sounds_like(word: str, explicit: str) -> bool:
    """One phoneme or less apart (witch / ditch / pitch for bitch): would be heard as it.
    Unstressed vowels count as one sound (witches / bitches: AH0 = IH0)."""
    a, b = _phones(word), _phones(explicit)
    if not a or not b:
        return False
    plain = lambda ps: [re.sub(r"\d", "", "AH" if p.endswith("0") else p) for p in ps]  # noqa: E731
    a, b = plain(a), plain(b)
    if abs(len(a) - len(b)) > 1:
        return False
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] <= 1


def part_alike(word: str, explicit: str) -> bool:
    """Heard as part of the explicit word: inside its spelling (damn in goddamn) or at most
    one phoneme from its last sounds (dam for goddamn)."""
    if word in explicit:
        return True
    a, b = _phones(word), _phones(explicit)
    if not a or len(b) <= len(a):
        return False
    strip = lambda p: [x.rstrip("012") for x in p]  # noqa: E731
    return sum(x != y for x, y in zip(strip(a), strip(b[-len(a):]))) <= 1


def suggest(line: str, targets: list[str], n_syllables: int, embed, k: int = 8,
            exclude: set[str] | None = None, before: str = "",
            explicit: list[str] | None = None) -> list[dict]:
    """Rhyming end words for `line`, best first: {word, kind, line, fit, similarity, score}.

    `explicit`: the explicit words being replaced; their sound-alikes are never proposed."""
    exclude = (exclude or set()) | {t.lower() for t in targets} | set(WORD.findall(line.lower()))
    found = [(w, kind) for w, kind in candidates(targets, n_syllables, exclude)
             if not any(sounds_like(w, e) for e in explicit or [])]
    if not found:
        return []
    texts = [replace_end(line, w) for w, _ in found]
    vecs = embed([line] + texts)
    sims = vecs[1:] @ vecs[0]
    fit = slot_fit().score(f"{before}\n{replace_end(line, '<mask>')}".strip(),
                           [w for w, _ in found])
    z = lambda x: (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else x * 0  # noqa: E731
    score = z(fit) + z(sims) + np.array([0.3 if kind == "perfect" else 0 for _, kind in found])
    order = np.argsort(-score)
    return [{"word": found[i][0], "kind": found[i][1], "line": texts[i],
             "fit": round(float(fit[i]), 2), "similarity": round(float(sims[i]), 3),
             "score": round(float(score[i]), 2)} for i in order[:k]]


KIND_BONUS = {"perfect": 0.3, "slant": 0.0, "assonance": -0.3}
DIVERSITY = 3.0  # weight of the closeness to words already picked (diverse())
SAME_RHYME = 0.4  # added closeness for two words with the same rhyming part (pinch / cinch)


def diverse(words: list[str], score: np.ndarray, embed, k: int, pool: int = 30) -> list[int]:
    """k indexes, best first, as different from each other as possible (maximal marginal
    relevance): each pick maximizes its score minus DIVERSITY x its closeness to the words
    already picked (cosine of the word embeddings, + SAME_RHYME when the rhyming part is the
    same). Only the `pool` best-scored words are considered, so every pick still fits."""
    order = [int(i) for i in np.argsort(-score)[:pool]]
    if len(order) <= 1:
        return order[:k]
    vecs = embed([words[i] for i in order])
    close = vecs @ vecs.T
    keys = [rhyme_key(words[i]) for i in order]
    close = close + SAME_RHYME * np.array([[a == b for b in keys] for a in keys])
    rel = score[order]
    picked = [0]
    while len(picked) < min(k, len(order)):
        rest = [j for j in range(len(order)) if j not in picked]
        gain = [rel[j] - DIVERSITY * max(close[j, p] for p in picked) for j in rest]
        picked.append(rest[int(np.argmax(gain))])
    return [order[j] for j in picked]


def suggest_for_word(template: str, explicit: str, reference: str, embed, k: int = 4,
                     before: str = "", varied: bool = True) -> list[dict]:
    """Clean words for the slot `<mask>` in `template` that rhyme with the explicit word they
    replace (bitch -> rich, switch), with its syllable count (within one), never a sound-alike
    of it or of its last sounds (dam for goddamn). Scored by masked-LM fit in the slot and by how close the filled line stays in
    meaning to `reference` (the current rewrite). `varied` (default) also takes assonance
    rhymes (bitch -> wish, kiss, bridge; scored a little lower) and returns the k picked by
    diverse() instead of the k best. Returns {word, kind, line, fit, ...}."""
    exclude = set(WORD.findall(template.lower().replace("<mask>", " "))) | {explicit.lower()}
    found = [(w, kind) for w, kind in candidates([explicit], word_syllables(explicit), exclude,
                                                 loose=varied)
             if not sounds_like(w, explicit) and not part_alike(w, explicit)]
    if not found:
        return []
    texts = [template.replace("<mask>", w) for w, _ in found]
    vecs = embed([reference] + texts)
    sims = vecs[1:] @ vecs[0]
    fit = slot_fit().score(f"{before}\n{template}".strip(), [w for w, _ in found])
    z = lambda x: (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else x * 0  # noqa: E731
    score = z(fit) + z(sims) + np.array([KIND_BONUS[kind] for _, kind in found])
    order = (diverse([w for w, _ in found], score, embed, k) if varied
             else [int(i) for i in np.argsort(-score)[:k]])
    return [{"word": found[i][0], "kind": found[i][1], "line": texts[i],
             "fit": round(float(fit[i]), 2), "similarity": round(float(sims[i]), 3),
             "score": round(float(score[i]), 2)} for i in order]


def main() -> None:
    parser = argparse.ArgumentParser(description="End-rhyme thesaurus.")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("line", nargs="?")
    parser.add_argument("targets", nargs="*", help="words the end should rhyme with")
    args = parser.parse_args()
    if args.build:
        build(args.min_count)
        return
    from radio_ready.scoring import Similarity

    sim = Similarity(on="cpu")
    embed = lambda t: sim.model.encode(t, normalize_embeddings=True)  # noqa: E731
    n = word_syllables(end_of(args.line))
    for s in suggest(args.line, args.targets, n, embed, explicit=args.targets):
        print(f"{s['score']:5.2f} fit={s['fit']:6.2f} sim={s['similarity']:.3f} "
              f"{s['kind']:7s} {s['line']}")


if __name__ == "__main__":
    main()
