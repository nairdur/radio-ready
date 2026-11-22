"""Run the data pipeline end to end.

Steps, in order:
  download         raw datasets and scoring models (existing files are reused)
  labels           song-level explicit labels
  learned_lexicon  learned explicit-word list with human vetting
  classifier       section-level explicit classifier (only with --with-classifier)
  kidzbop          Kidz Bop song sections and word statistics
  genius           Genius sections
  aae              AAE proportion for Genius sections
  rhyme_vocab      rhyme thesaurus for the vetting app's suggestions (radio_ready.rhymes
                   --build; CPU)
  commercial       add new sections to the commercial-LLM copy-paste table (never edits
                   filled cells)
  noise            (only with --with-noise) noise-style rewrites of the Kidz Bop pilot:
                   explicit words replaced by made-up sounds before Qwen2.5-7B rewrites
                   (radio_ready.generate --style noise-vowels --samples 5; needs the GPU)
  rhyme            (only with --with-rhyme) rhyme swaps of the Kidz Bop pilot: thesaurus
                   rhymes for every explicit word, reworded by Qwen2.5-7B
                   (radio_ready.rhyme_swap; needs the GPU)
  pairs            rebuild every pair set (pairs/<source>.jsonl/.md, final, lines) from the
                   committed patches and the section records: the vetted pairs, reconstructed
                   without rerunning any model

New noise or rhyme runs show up as sources in the vetting app (vet.bat / vet.command).
Each step runs as `python -m radio_ready.<step>` and the run stops at the first failure.

Run:  python -m radio_ready.pipeline                       # data, then the pairs rebuilt
      python -m radio_ready.pipeline --only pairs          # reconstruct the pairs only
      python -m radio_ready.pipeline --from kidzbop
      python -m radio_ready.pipeline --with-rhyme --with-noise    # also new noise/rhyme runs
      python -m radio_ready.pipeline --only rhyme
"""

import argparse
import subprocess
import sys
import time

STEPS = ["download", "labels", "learned_lexicon", "classifier", "kidzbop", "genius", "aae",
         "rhyme_vocab", "commercial", "noise", "rhyme", "pairs"]
MODULES = {"noise": "generate", "rhyme": "rhyme_swap", "rhyme_vocab": "rhymes"}
STEP_ARGS = {"download": ["all"], "pairs": ["--rebuild"], "rhyme_vocab": ["--build"],
             "noise": ["--style", "noise-vowels", "--samples", "5"]}
OPTIONAL = {"classifier": "with_classifier", "noise": "with_noise", "rhyme": "with_rhyme"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RadioReady data pipeline.")
    parser.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0])
    parser.add_argument("--to", dest="end", choices=STEPS, default=STEPS[-1])
    parser.add_argument("--only", nargs="+", choices=STEPS)
    parser.add_argument("--with-classifier", action="store_true")
    parser.add_argument("--with-noise", action="store_true", help="also run the noise step")
    parser.add_argument("--with-rhyme", action="store_true", help="also run the rhyme step")
    args = parser.parse_args()

    steps = args.only or STEPS[STEPS.index(args.start) : STEPS.index(args.end) + 1]
    if not args.only:  # steps named with --only always run
        steps = [s for s in steps if s not in OPTIONAL or getattr(args, OPTIONAL[s])]

    timings = []
    for step in steps:
        print(f"\n=== {step} ===", flush=True)
        started = time.time()
        module = MODULES.get(step, step)
        result = subprocess.run([sys.executable, "-m", f"radio_ready.{module}",
                                 *STEP_ARGS.get(step, [])])
        timings.append((step, time.time() - started))
        if result.returncode != 0:
            print(f"\nstep '{step}' failed (exit {result.returncode})")
            sys.exit(result.returncode)

    print("\n=== done ===")
    for step, seconds in timings:
        print(f"{step:16s} {seconds / 60:6.1f} min")


if __name__ == "__main__":
    main()
