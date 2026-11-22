"""Side by side: a free_* run against the solid run on the same sections, per flagged line.

Counts per run: lines proposed, clean (no detected explicit word), syllables within one, end
rhyme kept. Writes outputs/comparison/<free>_vs_<solid>.md (local, gitignored: it has lyrics).

Run:  python -m radio_ready.compare_free [FREE] [SOLID]
"""

import argparse
import json

from radio_ready import paths
from radio_ready.explicit import ExplicitDetector
from radio_ready.prompts import target_lines
from radio_ready.text import end_rhyme_kept, syllables


def load(name: str) -> dict:
    with open(paths.GENERATIONS_DIR / f"{name}.jsonl", encoding="utf-8") as f:
        return {g["id"]: g for g in map(json.loads, f)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("free", nargs="?", default="free_genius_sections_free-v1")
    parser.add_argument("solid", nargs="?", default="solid_genius300_solid-v1")
    args = parser.parse_args()
    detector = ExplicitDetector()
    free, solid = load(args.free), load(args.solid)
    with open(paths.GENIUS_SECTIONS_JSONL, encoding="utf-8") as f:
        recs = {r["id"]: r for r in map(json.loads, f) if r["id"] in free}
    stats = {n: dict(proposed=0, clean=0, syllables=0, rhyme=0) for n in ("free", "solid")}
    md, total = [], 0
    for sid, rec in recs.items():
        md += [f"## {rec['artist']} - {rec['title']}", "",
               "| line | original | solid | free |", "|---|---|---|---|"]
        for i in sorted(target_lines(rec, detector)):
            total += 1
            row = [str(i + 1), rec["lines"][i]]
            for n, gen in (("solid", solid.get(sid)), ("free", free[sid])):
                new = (gen or {}).get("rewrites", {}).get(str(i), "")
                if gen and not gen.get("valid"):
                    new = ""
                if new:
                    s = stats[n]
                    s["proposed"] += 1
                    s["clean"] += not detector.find(new)
                    s["syllables"] += abs(syllables(new) - syllables(rec["lines"][i])) <= 1
                    s["rhyme"] += end_rhyme_kept(rec["lines"][i], new)
                row.append(new or "-")
            md.append("| " + " | ".join(c.replace("|", "\\|") for c in row) + " |")
        md.append("")
    head = [f"# {args.free} vs {args.solid}", "", f"{len(recs)} sections, {total} flagged lines.",
            "", "| run | proposed | clean | syllables within 1 | end rhyme kept |",
            "|---|---|---|---|---|"]
    for n, s in stats.items():
        head.append(f"| {n} | {s['proposed']} | {s['clean']} | {s['syllables']} | {s['rhyme']} |")
        print(n, s, f"of {total} flagged lines")
    out = paths.ROOT / "outputs" / "comparison" / f"{args.free}_vs_{args.solid}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head + [""] + md) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
