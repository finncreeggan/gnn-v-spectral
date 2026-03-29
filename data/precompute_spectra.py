"""Precompute whole and regularized eigenspectra for cached synthetic graphs.

Saves one .pt file per graph under data/cache/synthetic/sbm/{graph_id}.pt:
    {
        "whole_V":    Float[Tensor, "num_nodes num_nodes"],
        "whole_evals": Float[Tensor, "num_nodes"],
        "reg_V":      Float[Tensor, "num_nodes n_eigenvectors_plus_1"],
        "reg_evals":  Float[Tensor, "n_eigenvectors_plus_1"],
    }

Usage:
    python data/precompute_spectra.py --family lfr --noise-type clean
"""

# TODO: Need a way to verify correctness. Probably use old assignment function to plot eigenspectra and then check with plot from saved eigenspectra

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.generators.io import load_edge_index
from methods.spectral.embeddings import regularized_eigenspectrum, whole_eigenspectrum

FAMILIES = ("lfr", "sbm")
NOISE_TYPES = ("clean", "random", "targeted_betweenness")



def precompute(
    *,
    root: Path,
    families: tuple[str, ...],
    noise_types: tuple[str, ...],
) -> None:
    for idx, family in enumerate(families):
        out_dir = root / family / noise_types[idx] / "spectra"  # 
        out_dir.mkdir(exist_ok=True)

        meta_path = root / "metadata" / f"graph_index_{family}.csv"
        if not meta_path.exists():
            print(f"[skip] no metadata found at {meta_path}")
            continue

        df = pd.read_csv(meta_path)
        rows = df[df["noise_type"].isin(noise_types)]
        print(f"[{family}] {len(rows)} graphs to process")

        for _, row in rows.iterrows():
            graph_id = str(row["graph_id"])
            out_path = out_dir / f"{graph_id}.pt"
            if out_path.exists():
                print(f"  [skip] {out_path} already exists")
                continue

            edge_path = root / Path(str(row["edge_path"]))
            label_path = root / Path(str(row["label_path"]))

            labels = np.load(label_path)
            num_nodes = len(labels)
            edge_index = load_edge_index(edge_path)

            print(f"  {graph_id} ({num_nodes} nodes) ...", end=" ", flush=True)

            whole_V, whole_evals = whole_eigenspectrum(edge_index, num_nodes) # evals = eigenvalues
            reg_V, reg_evals = regularized_eigenspectrum(edge_index, num_nodes,)

            torch.save(
                {
                    "whole_V": whole_V,
                    "whole_evals": whole_evals,
                    "reg_V": reg_V,
                    "reg_evals": reg_evals,
                },
                out_path,
            )
            print("done")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        default="all",
        choices=(*FAMILIES, "all"),
        help="Graph family to process (default: all)",
    )
    parser.add_argument(
        "--noise-type",
        default="clean",
        choices=(*NOISE_TYPES, "all"),
        dest="noise_type",
        help="Noise type filter (default: clean)",
    )
    parser.add_argument(
        "--n-eigenvectors",
        type=int,
        default=20,
        dest="n_eigenvectors",
        help="Candidate pool size for regularized_eigenspectrum (default: 20)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/cache/synthetic"),
        help="Dataset root directory (default: data/cache/synthetic)",
    )
    args = parser.parse_args()

    families = FAMILIES if args.family == "all" else (args.family,)
    noise_types = NOISE_TYPES if args.noise_type == "all" else (args.noise_type,)

    precompute(
        root=args.root,
        families=families,
        noise_types=noise_types,
    )


if __name__ == "__main__":
    main()
