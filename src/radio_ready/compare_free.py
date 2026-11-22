"""Side by side: free_* runs against the solid run on the same sections, per flagged line.

Counts per run: lines proposed, clean (no detected explicit word), syllables within one, end
rhyme kept, stock swaps (a generic replacement from the vetting sheet: stuff, girl, mess...).
Writes outputs/comparison/<first run>_vs.md (local, gitignored: it has lyrics).

Run:  python -m radio_ready.compare_free [RUN ...] [--solid solid_genius300_solid-v1]
"""

import argparse
import json

from radio_ready import paths
from radio_ready.explicit import ExplicitDetector
from radio_ready.free import generic_in
from radio_ready.prompts import target_lines
from radio_ready.text import end_rhyme_kept, syllables


def load(name: str) -> dict:
    path = paths.GENERATIONS_DIR / f"{name}.jsonl"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {g["id"]: g for g in map(json.loads, f)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", default=["free_genius_sections_free-v1"])
    parser.add_argument("--solid", default="solid_genius300_solid-v1")
    args = parser.parse_args()
    detector = ExplicitDetector()
    names = [args.solid] + args.runs
    runs = {n: load(n) for n in names}
    ids = set(runs[args.runs[0]])
    with open(paths.GENIUS_SECTIONS_JSONL, encoding="utf-8") as f:
        recs = {r["id"]: r for r in map(json.loads, f) if r["id"] in ids}
    keys = ("proposed", "clean", "syllables", "rhyme", "stock swap")
    stats = {n: dict.fromkeys(keys, 0) for n in names}
    md, total = [], 0
    for sid, rec in recs.items():
        md += [f"## {rec['artist']} - {rec['title']}", "",
               "| line | original | " + " | ".join(names) + " |",
               "|---" * (len(names) + 2) + "|"]
        for i in sorted(target_lines(rec, detector)):
            total += 1
            old, row = rec["lines"][i], [str(i + 1), rec["lines"][i]]
            for n in names:
                gen = runs[n].get(sid)
                new = (gen or {}).get("rewrites", {}).get(str(i), "") if gen and \
                    gen.get("valid") else ""
                if new:
                    s = stats[n]
                    s["proposed"] += 1
                    s["clean"] += not detector.find(new)
                    s["syllables"] += abs(syllables(new) - syllables(old)) <= 1
                    s["rhyme"] += end_rhyme_kept(old, new)
                    s["stock swap"] += bool(generic_in(new, old))
                row.append(new or "-")
            md.append("| " + " | ".join(c.replace("|", "\\|") for c in row) + " |")
        md.append("")
    head = [f"# {' vs '.join(names)}", "", f"{len(recs)} sections, {total} flagged lines.", "",
            "| run | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for n, s in stats.items():
        head.append(f"| {n} | " + " | ".join(str(s[k]) for k in keys) + " |")
        print(f"{n:40s}", s, f"of {total}")
    out = paths.ROOT / "outputs" / "comparison" / f"{args.runs[0]}_vs.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head + [""] + md) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
