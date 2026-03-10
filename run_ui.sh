#!/bin/bash
# Run the FM Synth UI

cd "$(dirname "$0")"
source .venv/bin/activate
python3 -m synth_ui
