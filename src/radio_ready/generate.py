"""Generate clean rewrites of sections with a local model (4-bit by default).

Inputs:  a sections JSONL (default data/processed/kidzbop/pilot.jsonl)
Output:  outputs/generations/<name>.jsonl, one record per section:
         id, model, prompt_version, targets (line indexes that must be rewritten), raw,
         rewrites ({line index: text}, targets plus any neighbor lines the model changed),
         lines (the merged section; None if a target line is missing), valid, seconds

Target and neighbor lines are defined in radio_ready.prompts; the prompt shows the whole
section as context.

Genius sections reserved for commercial LLMs (rows of data/commercial/commercial_table.csv)
are skipped.
Sections already in the output file are skipped, so an interrupted run can be resumed.

Models slightly larger than the GPU (e.g. Qwen3-14B in 4-bit on 8 GB): --cpu-embeddings keeps
the 4-bit transformer layers on the GPU and puts the unquantized input embeddings and output
head on the CPU. Qwen3 models: --thinking enables thinking mode; the <think> block is
dropped from the response. The output name defaults to <model>-<quant>_<input>_<prompt version>.

Run:  python -m radio_ready.generate [--model NAME] [--input FILE] [--limit N]
      python -m radio_ready.generate --model Qwen/Qwen3-14B --cpu-embeddings
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
from radio_ready.prompts import PROMPT_VERSION, merge, parse_rewrites, single_prompt, target_lines

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


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
def rewrite(tok, model, record: dict, targets: dict, max_new_tokens: int,
            temperature: float, thinking: bool) -> str:
    text = tok.apply_chat_template(single_prompt(record, targets), add_generation_prompt=True,
                                   tokenize=False, enable_thinking=thinking)
    device = getattr(model, "hf_device_map", {}).get("model.embed_tokens", model.device)
    enc = tok(text, return_tensors="pt").to(device)
    sampling = {"do_sample": True, "temperature": temperature, "top_p": 0.9} \
        if temperature > 0 else {"do_sample": False}
    out = model.generate(**enc, max_new_tokens=max_new_tokens, **sampling)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.split("</think>")[-1].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-model rewrites of sections.")
    parser.add_argument("--input", default=str(paths.KIDZBOP_DIR / "pilot.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--name", default=None)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--cpu-embeddings", action="store_true",
                        help="embeddings and output head on the CPU, layers on the GPU")
    parser.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.thinking and args.max_new_tokens == 512:
        args.max_new_tokens = 4096
    if args.name is None:
        quant = "bf16" if args.no_4bit else "4bit"
        source = Path(args.input).stem
        args.name = (f"{args.model.split('/')[-1].lower()}-{quant}"
                     f"{'-thinking' if args.thinking else ''}_{source}_{PROMPT_VERSION}")
    records = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    skip = reserved_ids()
    out = paths.GENERATIONS_DIR / f"{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [r for r in records if r["id"] not in done and r["id"] not in skip][: args.limit]
    print(f"sections: {len(records)}, done: {len(done)}, reserved for commercial: "
          f"{sum(r['id'] in skip for r in records)}, to generate: {len(todo)}")

    detector = ExplicitDetector()
    tok, model = load_model(args.model, not args.no_4bit, args.cpu_embeddings)
    valid = 0
    with open(out, "a", encoding="utf-8") as f:
        for i, record in enumerate(todo, start=1):
            start = time.time()
            targets = target_lines(record, detector)
            if not targets:
                print(f"[{i}/{len(todo)}] {record['id']} skipped: no line has a detected word")
                continue
            raw = rewrite(tok, model, record, targets, args.max_new_tokens, args.temperature,
                          args.thinking)
            rewrites = parse_rewrites(raw, record, targets)
            lines = merge(record, rewrites) if rewrites is not None else None
            valid += lines is not None
            f.write(json.dumps({
                "id": record["id"], "model": args.model, "quantization": None if args.no_4bit
                else "nf4", "thinking": args.thinking, "temperature": args.temperature,
                "prompt_version": PROMPT_VERSION,
                "targets": sorted(targets), "raw": raw,
                "rewrites": {str(k): v for k, v in (rewrites or {}).items()},
                "lines": lines, "valid": lines is not None,
                "seconds": round(time.time() - start, 1),
            }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {record['id']} valid={lines is not None} "
                  f"{time.time() - start:.1f}s", flush=True)
    print(f"valid: {valid}/{len(todo)}, output: {out}")


if __name__ == "__main__":
    main()
