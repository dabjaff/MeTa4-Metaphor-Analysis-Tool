#!/usr/bin/env bash
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
python3 -m meta4.cli || python -m meta4.cli
read -r -p "Press Return to close... " _
