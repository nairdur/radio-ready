"""Run the data pipeline end to end.

Steps, in order:
  download         raw datasets and scoring models (existing files are reused)
  labels           song-level explicit labels
  learned_lexicon  learned explicit-word list with human vetting
  classifier       section-level explicit classifier (only with --with-classifier)
  kidzbop          Kidz Bop song sections and word statistics
  genius           Genius sections
  aae              AAE proportion for Genius sections
  commercial       add new sections to the commercial-LLM copy-paste table (never edits
                   filled cells)

Each step runs as `python -m radio_ready.<step>` and the run stops at the first failure.

Run:  python -m radio_ready.pipeline
      python -m radio_ready.pipeline --from kidzbop
      python -m radio_ready.pipeline --only genius aae
"""

import argparse
import subprocess
import sys
import time

STEPS = ["download", "labels", "learned_lexicon", "classifier", "kidzbop", "genius", "aae",
         "commercial"]
STEP_ARGS = {"download": ["all"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RadioReady data pipeline.")
    parser.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0])
    parser.add_argument("--to", dest="end", choices=STEPS, default=STEPS[-1])
    parser.add_argument("--only", nargs="+", choices=STEPS)
    parser.add_argument("--with-classifier", action="store_true")
    args = parser.parse_args()

    steps = args.only or STEPS[STEPS.index(args.start) : STEPS.index(args.end) + 1]
    if not args.with_classifier and not args.only:
        steps = [s for s in steps if s != "classifier"]

    timings = []
    for step in steps:
        print(f"\n=== {step} ===", flush=True)
        started = time.time()
        result = subprocess.run([sys.executable, "-m", f"radio_ready.{step}", *STEP_ARGS.get(step, [])])
        timings.append((step, time.time() - started))
        if result.returncode != 0:
            print(f"\nstep '{step}' failed (exit {result.returncode})")
            sys.exit(result.returncode)

    print("\n=== done ===")
    for step, seconds in timings:
        print(f"{step:16s} {seconds / 60:6.1f} min")


if __name__ == "__main__":
    main()
