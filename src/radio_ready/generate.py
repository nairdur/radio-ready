"""Generate clean rewrites of sections with a local model (4-bit by default).

Inputs:  a sections JSONL (default data/processed/kidzbop/pilot.jsonl)
Output:  outputs/generations/<name>.jsonl, one record per section:
         id, model, prompt_version, style, samples, temperature,
         targets (line indexes with a detected explicit word), raw (chosen responses),
         rewrites ({line index: text} for changed lines), lines (the merged section; None if
         a request gave no usable answer), valid, passed (every chosen candidate passed the
         filters), units (per request: line span and every candidate with its scores), seconds

--style picks the prompt (radio_ready.prompts):
  annotated  (v5) whole section with syllable counts, rhyme letters and REWRITE marks
  plain      whole section with no annotations; the model decides what radio would censor
  surgical   (default) one request per detected line with 4 lines above and below (--context)
             and the detected words named; the model returns only that line
  stanza     one request per 4-line stanza with a detected word (consecutive chunks of the
             section), detected words named; freer rewrite that keeps flow and rhythm
  surgical-free, stanza-free  (v8) like surgical / stanza, but lines may be reworded as a
             whole, keeping meaning, syllables, rhythm and end rhyme; --rhyme makes the model
             write the rhyme pattern first (end words it rhymes with / letters like AABB)

Candidates: --samples N draws N answers per request (temperature 1.0 with top-p 0.9 when
N > 1: top-p drops the unlikely tail, so the higher temperature varies the wording without
junk) and keeps the best, ParaDeHate-style. A candidate passes when
  - the explicit-word detector finds nothing in it,
  - unbiased-toxic-roberta toxicity < 0.9           (ParaDeHate threshold, radio_ready.scoring)
  - all-distilroberta-v1 cosine to the original >= 0.70 (ParaDeHate threshold)
Among passing candidates (or all, if none passes) the best breaks the fewest end rhymes
(end word kept or a shared rhyme key with the original line), keeps every line within one
syllable, then has the highest similarity x fluency. Fluency is the CoLA acceptability of each
changed line relative to its original (the ParaDetox fluency model); it catches nonsense such
as "I should be a sad stuff" and is relative because dialect scores low on both sides. The scorers run on the CPU.

Genius sections reserved for commercial LLMs (rows of data/commercial/commercial_table.csv)
are skipped. Sections already in the output file are skipped, so an interrupted run resumes.

Models slightly larger than the GPU (e.g. Qwen3-14B in 4-bit on 8 GB): --cpu-embeddings keeps
the 4-bit transformer layers on the GPU and puts the unquantized input embeddings and output
head on the CPU. Qwen3 models: --thinking enables thinking mode; the <think> block is
dropped from the response. The output name defaults to
<model>-<quant>_<input>_<prompt version>[_n<samples>].

Run:  python -m radio_ready.generate [--model NAME] [--input FILE] [--limit N]
      python -m radio_ready.generate --model Qwen/Qwen3-14B --cpu-embeddings --style stanza --samples 5
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from radio_ready import paths
from radio_ready.commercial import reserved_ids
from radio_ready.explicit import ExplicitDetector
from radio_ready.prompts import (
    FREE_VERSION, PLAIN_VERSION, PROMPT_VERSION, STANZA_VERSION, SURGICAL_CONTEXT,
    SURGICAL_VERSION, model_rhyme, parse_plain, parse_rewrites, parse_stanza, parse_surgical,
    parse_surgical_free, plain_prompt, single_prompt, stanza_free_prompt, stanza_prompt, stanzas,
    surgical_free_prompt, surgical_prompt, target_lines,
)
from radio_ready.scoring import (
    SIMILARITY_THRESHOLD, TOXICITY_THRESHOLD, Fluency, Similarity, Toxicity,
)
from radio_ready.text import end_word, line_rhyme_keys, syllables

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
VERSIONS = {"annotated": PROMPT_VERSION, "plain": PLAIN_VERSION, "surgical": SURGICAL_VERSION,
            "stanza": STANZA_VERSION, "surgical-free": f"{FREE_VERSION}-surgical-free",
            "stanza-free": f"{FREE_VERSION}-stanza-free"}
SAMPLING_TEMPERATURE = 1.0
SAMPLING_TOP_P = 0.9
MAX_SYLLABLE_DIFF = 1  # per line, before candidates are ranked by similarity x fluency

CPU_EMBEDDINGS_MAP = {"model.embed_tokens": "cpu", "lm_head": "cpu", "model.layers": 0,
                      "model.norm": 0, "model.rotary_emb": 0}


def load_model(name: str, four_bit: bool, cpu_embeddings: bool):
    tok = AutoTokenizer.from_pretrained(name)
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=cpu_embeddings) if four_bit else None
    device_map = CPU_EMBEDDINGS_MAP if cpu_embeddings else "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=quant, dtype=torch.bfloat16, device_map=device_map).eval()
    return tok, model


@torch.no_grad()
def complete(tok, model, messages: list[dict], samples: int, max_new_tokens: int,
             temperature: float, top_p: float, thinking: bool) -> list[str]:
    """`samples` responses to one chat (one batch: num_return_sequences)."""
    text = tok.apply_chat_template(messages, add_generation_prompt=True,
                                   tokenize=False, enable_thinking=thinking)
    device = getattr(model, "hf_device_map", {}).get("model.embed_tokens", model.device)
    enc = tok(text, return_tensors="pt").to(device)
    sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p,
                "num_return_sequences": samples} if temperature > 0 else {"do_sample": False}
    out = model.generate(**enc, max_new_tokens=max_new_tokens, **sampling)
    start = enc["input_ids"].shape[1]
    return [tok.decode(o[start:], skip_special_tokens=True).split("</think>")[-1].strip()
            for o in out]


def rhymes_with(original: str, rewrite: str) -> bool:
    """End word kept, or the rewritten line shares an end-rhyme key with the original."""
    return (end_word(original) == end_word(rewrite)
            or bool(set(line_rhyme_keys(original)) & set(line_rhyme_keys(rewrite))))


class Selector:
    """Scores candidates for one span of lines and picks the best (ParaDeHate filters)."""

    def __init__(self, detector: ExplicitDetector):
        self.detector = detector
        self.similarity = Similarity(on="cpu")
        self.toxicity = Toxicity(on="cpu")
        self.fluency = Fluency(on="cpu")

    def pick(self, original: list[str], candidates: list[tuple[str, list[str] | None]]):
        """(index of the best candidate or None, scored candidates)."""
        scored = []
        usable = [(raw, lines) for raw, lines in candidates if lines is not None]
        if usable:
            texts = ["\n".join(lines) for _, lines in usable]
            sims = self.similarity.pairwise(texts, ["\n".join(original)] * len(texts))
            toxs = self.toxicity.score(texts)
            base = self.fluency.score(list(original))
            fl = self.fluency.score([line for _, ls in usable for line in ls])
        k = 0
        for raw, lines in candidates:
            if lines is None:
                scored.append({"raw": raw, "lines": None, "passed": False})
                continue
            text = "\n".join(lines)
            explicit = sorted(set(self.detector.find(text)))
            sim, tox = float(sims[k]), float(toxs[k])
            n = len(original)
            # fluency of each changed line relative to its original line: dialect scores low
            # on both sides, so only a drop counts
            drop = max([float(base[j] - fl[k * n + j]) for j in range(n)
                        if lines[j] != original[j]] or [0.0])
            k += 1
            scored.append({
                "raw": raw, "lines": lines, "explicit": explicit,
                "toxicity": round(tox, 3), "similarity": round(sim, 3),
                "syllable_diff": sum(abs(syllables(a) - syllables(b))
                                     for a, b in zip(original, lines)),
                "rhymes_broken": sum(not rhymes_with(a, b) for a, b in zip(original, lines)),
                "max_line_syllable_diff": max(abs(syllables(a) - syllables(b))
                                              for a, b in zip(original, lines)),
                "fluency_drop": round(max(drop, 0.0), 3),
                "passed": not explicit and tox < TOXICITY_THRESHOLD
                          and sim >= SIMILARITY_THRESHOLD,
            })
        ranked = [i for i, c in enumerate(scored) if c["lines"] is not None]
        if not ranked:
            return None, scored
        best = min(ranked, key=lambda i: rank_key(scored[i]))
        return best, scored


def rank_key(c: dict) -> tuple:
    """Lower is better: filters, end rhymes kept, every line within MAX_SYLLABLE_DIFF, then
    similarity x fluency (ParaDetox-style joint score)."""
    joint = c["similarity"] * (1 - c["fluency_drop"])
    return (not c["passed"], c["rhymes_broken"],
            c["max_line_syllable_diff"] > MAX_SYLLABLE_DIFF, -joint)


def units(record: dict, targets: dict, style: str, context: int, rhyme: bool = False):
    """Requests for a section: (start, end, prompt(lines) -> messages, parse(text) -> lines)."""
    n = len(record["lines"])
    if style == "annotated":
        def parse(raw):
            rw = parse_rewrites(raw, record, targets)
            return None if rw is None else [rw.get(i, l) for i, l in enumerate(record["lines"])]
        return [(0, n, lambda lines: single_prompt(record, targets), parse)]
    if style == "plain":
        def parse(raw):
            rw = parse_plain(raw, record)
            return None if rw is None else [rw.get(i, l) for i, l in enumerate(record["lines"])]
        return [(0, n, lambda lines: plain_prompt(record), parse)]
    if style == "surgical":
        def one(i):
            def parse(raw):
                line = parse_surgical(raw)
                return None if line is None else [line]
            return (i, i + 1, lambda lines: surgical_prompt(record, lines, i, targets[i], context),
                    parse)
        return [one(i) for i in sorted(targets)]
    if style == "surgical-free":
        def one_free(i):
            def parse(raw):
                line = parse_surgical_free(raw)
                return None if line is None else [line]
            return (i, i + 1, lambda lines: surgical_free_prompt(record, lines, i, targets[i],
                                                                 rhyme, context), parse)
        return [one_free(i) for i in sorted(targets)]
    if style == "stanza-free":
        def chunk_free(start, end):
            words = sorted({w for t in targets if start <= t < end for w in targets[t]})
            return (start, end,
                    lambda lines: stanza_free_prompt(record, lines, start, end, words, rhyme),
                    lambda raw: parse_stanza(raw, end - start))
        return [chunk_free(s, e) for s, e in stanzas(record, targets)]

    def chunk(start, end):
        words = sorted({w for t in targets if start <= t < end for w in targets[t]})
        return (start, end, lambda lines: stanza_prompt(record, lines, start, end, words),
                lambda raw: parse_stanza(raw, end - start))
    return [chunk(s, e) for s, e in stanzas(record, targets)]


def rewrite(tok, model, selector: Selector, record: dict, targets: dict, args) -> dict:
    lines = list(record["lines"])
    raws, scored_units, ok, passed = [], [], True, True
    for start, end, prompt, parse in units(record, targets, args.style, args.context,
                                           args.rhyme):
        responses = complete(tok, model, prompt(lines), args.samples, args.max_new_tokens,
                             args.temperature, args.top_p, args.thinking)
        best, scored = selector.pick(record["lines"][start:end],
                                     [(r, parse(r)) for r in responses])
        for c in scored:
            c["model_rhyme"] = model_rhyme(c["raw"]) if args.rhyme else None
        scored_units.append({"start": start, "end": end, "chosen": best,
                             "distinct": len({c["raw"] for c in scored}), "candidates": scored})
        if best is None:
            ok = False
            raws.append(responses[0])
            continue
        lines[start:end] = scored[best]["lines"]
        raws.append(scored[best]["raw"])
        passed &= scored[best]["passed"]
    rewrites = {str(i): l for i, (l, o) in enumerate(zip(lines, record["lines"])) if l != o}
    return {"raw": "\n\n".join(raws), "rewrites": rewrites, "lines": lines if ok else None,
            "valid": ok, "passed": ok and passed, "units": scored_units}


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-model rewrites of sections.")
    parser.add_argument("--input", default=str(paths.KIDZBOP_DIR / "pilot.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--name", default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--cpu-embeddings", action="store_true",
                        help="embeddings and output head on the CPU, layers on the GPU")
    parser.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode")
    parser.add_argument("--style", choices=list(VERSIONS), default="surgical")
    parser.add_argument("--context", type=int, default=SURGICAL_CONTEXT,
                        help="surgical: lines of context above and below")
    parser.add_argument("--rhyme", action="store_true",
                        help="free styles: the model writes the rhyme pattern before its answer")
    parser.add_argument("--top-p", type=float, default=SAMPLING_TOP_P)
    parser.add_argument("--samples", type=int, default=1,
                        help="candidates per request; the best is kept")
    parser.add_argument("--temperature", type=float, default=None,
                        help=f"default 0 (greedy) for 1 sample, {SAMPLING_TEMPERATURE} for more")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.temperature is None:
        args.temperature = SAMPLING_TEMPERATURE if args.samples > 1 else 0.0
    if args.temperature == 0:
        args.samples = 1
    version = VERSIONS[args.style]
    if args.rhyme:
        version += "-rhyme"
    if args.style.startswith("surgical") and args.context != SURGICAL_CONTEXT:
        version += f"-ctx{args.context}"
    if args.thinking and args.max_new_tokens == 512:
        args.max_new_tokens = 4096
    if args.name is None:
        quant = "bf16" if args.no_4bit else "4bit"
        source = Path(args.input).stem
        args.name = (f"{args.model.split('/')[-1].lower()}-{quant}"
                     f"{'-thinking' if args.thinking else ''}_{source}_{version}"
                     f"{f'_n{args.samples}' if args.samples > 1 else ''}")
    records = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    skip = reserved_ids()
    out = paths.GENERATIONS_DIR / f"{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [r for r in records if r["id"] not in done and r["id"] not in skip][: args.limit]
    print(f"sections: {len(records)}, done: {len(done)}, reserved for commercial: "
          f"{sum(r['id'] in skip for r in records)}, to generate: {len(todo)}")

    detector = ExplicitDetector()
    selector = Selector(detector)
    tok, model = load_model(args.model, not args.no_4bit, args.cpu_embeddings)
    valid = passed = 0
    with open(out, "a", encoding="utf-8") as f:
        for i, record in enumerate(todo, start=1):
            start = time.time()
            targets = target_lines(record, detector)
            if not targets:
                print(f"[{i}/{len(todo)}] {record['id']} skipped: no line has a detected word")
                continue
            result = rewrite(tok, model, selector, record, targets, args)
            valid += result["valid"]
            passed += result["passed"]
            f.write(json.dumps({
                "id": record["id"], "model": args.model, "quantization": None if args.no_4bit
                else "nf4", "thinking": args.thinking, "temperature": args.temperature,
                "top_p": args.top_p, "samples": args.samples, "prompt_version": version, "style": args.style,
                "targets": sorted(targets), **result,
                "seconds": round(time.time() - start, 1),
            }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {record['id']} valid={result['valid']} "
                  f"passed={result['passed']} {time.time() - start:.1f}s", flush=True)
    print(f"valid: {valid}/{len(todo)}, passed filters: {passed}/{len(todo)}, output: {out}")


if __name__ == "__main__":
    main()
