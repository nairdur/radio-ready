#!/bin/bash
# Double-click to vet synthetic rewrites on a Mac (see src/radio_ready/vet.py).
cd "$(dirname "$0")"
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export PYTHONPATH="$PWD/src"
python3 -m radio_ready.vet "$@" || read -n 1 -p "Press any key to close"
