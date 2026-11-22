"""Simple rewrite baselines, written as generation files so radio_ready.compare scores them.

Systems (one output file each):
  mute        delete every detected explicit word (what a muted or bleeped radio edit leaves)
  dictionary  replace detected words with a fixed clean word (REPLACEMENTS, hand-written,
              chosen to keep the syllable count where possible); unlisted words are deleted
  bart-detox  s-nlp/bart-base-detox (Logacheva et al., 2022), a general detoxification model
              trained on ParaDetox, run on each line with a detected word; greedy decoding

Only lines with a detected word (radio_ready.explicit, the same detector that selects the
sections) are changed; every other line is copied.

Inputs:  a sections JSONL (default data/processed/eval_sections.jsonl)
Output:  outputs/systems/baseline-<system>_<input stem>.jsonl, one record per section with id,
         model, style, lines, valid (radio_ready.compare reads id, valid and lines). Not in
         outputs/generations, so the vetting app does not list them as sources.

Run:  python -m radio_ready.baselines [--systems mute dictionary bart-detox] [--input FILE]
"""

import argparse
import json
import re
from pathlib import Path

from radio_ready.eval_set import EVAL_SECTIONS_JSONL, SYSTEMS_DIR
from radio_ready.explicit import CENSORED, ExplicitDetector
from radio_ready.prompts import target_lines

BART_MODEL = "s-nlp/bart-base-detox"
SYSTEMS = ["mute", "dictionary", "bart-detox"]

# Hand-written replacements; the baseline is crude on purpose (no context, no grammar).
REPLACEMENTS = {
    "fuck": "freak", "fucks": "freaks", "fucked": "freaked", "fucking": "freaking",
    "fuckin": "freakin", "fucker": "freaker", "fuckers": "freakers",
    "motherfucker": "mother-lover", "motherfuckers": "mother-lovers",
    "motherfuckin": "mother-lovin", "motherfucking": "mother-loving",
    "shit": "stuff", "shits": "stuff", "shitty": "lousy", "bullshit": "nonsense",
    "bitch": "chick", "bitches": "chicks", "ass": "butt", "asses": "butts",
    "asshole": "loser", "assholes": "losers", "damn": "dang", "goddamn": "doggone",
    "hell": "heck", "dick": "jerk", "dicks": "jerks", "hoe": "girl", "hoes": "girls",
    "pussy": "kitty",
}
# Explicit stems inside other tokens ("fuckboy", "shitload"): replace the stem.
STEMS = {"fuck": "freak", "shit": "stuff", "bitch": "chick"}
CENSORED_GUESS = {"f": "freak", "s": "stuff", "b": "chick"}


def _match_case(original: str, new: str) -> str:
    if original.isupper() and len(original) > 1:
        return new.upper()
    return new[:1].upper() + new[1:] if original[:1].isupper() else new


def _cleanup(line: str) -> str:
    line = re.sub(r"\s+([,.!?;:])", r"\1", re.sub(r"\s{2,}", " ", line))
    return line.strip(" ,")


def replace_words(line: str, words: set[str], table: dict[str, str] | None) -> str:
    """Replace (or, with table None or a missing entry, delete) each detected word."""
    # already-censored tokens such as "f*ck": guess the word from its first letter
    line = CENSORED.sub(lambda m: "" if table is None else _match_case(
        m.group(0), CENSORED_GUESS.get(m.group(0)[0].lower(), "")), line)

    def swap(m: re.Match) -> str:
        token = m.group(0)
        low = token.lower().replace("’", "'")
        bare = low.rstrip("'")
        if bare not in words and low not in words:
            return token
        if table is None:
            return ""
        if bare in table:
            return _match_case(token, table[bare] + ("'" if low.endswith("'") else ""))
        for stem, new in STEMS.items():
            if stem in bare:
                return _match_case(token, low.replace(stem, new))
        return ""

    return _cleanup(re.sub(r"[A-Za-z]+(?:['’][A-Za-z]+)*['’]?", swap, line))


class BartDetox:
    def __init__(self, model: str = BART_MODEL):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).to(self.device).eval()

    def rewrite(self, lines: list[str], batch_size: int = 32) -> list[str]:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(lines), batch_size):
                enc = self.tokenizer(lines[i : i + batch_size], return_tensors="pt", padding=True,
                                     truncation=True, max_length=128).to(self.device)
                gen = self.model.generate(**enc, max_new_tokens=64, do_sample=False)
                out += self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return [o.strip() for o in out]


def run(system: str, records: list[dict], detector: ExplicitDetector) -> list[dict]:
    targets = {r["id"]: target_lines(r, detector) for r in records}
    rewritten: dict[tuple[str, int], str] = {}
    if system == "bart-detox":
        keys = [(r["id"], i) for r in records for i in sorted(targets[r["id"]])]
        lines = {r["id"]: r["lines"] for r in records}
        outputs = BartDetox().rewrite([lines[sid][i] for sid, i in keys])
        rewritten = dict(zip(keys, outputs))
    results = []
    for r in records:
        new = list(r["lines"])
        for i, words in targets[r["id"]].items():
            if system == "bart-detox":
                new[i] = rewritten[(r["id"], i)]
            else:
                table = REPLACEMENTS if system == "dictionary" else None
                new[i] = replace_words(r["lines"][i], set(words), table)
        results.append({"id": r["id"], "model": BART_MODEL if system == "bart-detox" else system,
                        "style": f"baseline-{system}", "prompt_version": None,
                        "targets": sorted(targets[r["id"]]), "lines": new,
                        # a deleted line is still a valid answer (the radio would mute it)
                        "valid": True})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Mute, dictionary and BART-detox baselines.")
    parser.add_argument("--input", default=str(EVAL_SECTIONS_JSONL))
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=SYSTEMS)
    args = parser.parse_args()

    records = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    detector = ExplicitDetector()
    SYSTEMS_DIR.mkdir(parents=True, exist_ok=True)
    for system in args.systems:
        results = run(system, records, detector)
        out = SYSTEMS_DIR / f"baseline-{system}_{Path(args.input).stem}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{system}: {len(results)} sections, output: {out}")


if __name__ == "__main__":
    main()
