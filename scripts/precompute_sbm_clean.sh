#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

cd "$ROOT"

for family in sbm lfr; do
    for noise_type in clean random targeted_betweenness; do
        echo "Running family=$family noise_type=$noise_type"
        uv run data/precompute_spectra.py \
            --family "$family" \
            --noise-type "$noise_type" \
            --root data/cache/synthetic
    done
done
