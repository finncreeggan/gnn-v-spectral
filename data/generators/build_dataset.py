from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np

from data.generators.characterize import compute_all_graph_stats
from data.generators.io import (
    format_base_graph_id,
    format_noise_code,
    make_graph_id,
    make_metadata_row,
    make_output_paths,
    save_graph_edgelist,
    save_labels,
    write_metadata_csv,
)
from data.generators.lfr import LFRConfig, generate_lfr
from data.generators.perturbations import build_noise_chain
from data.generators.sbm import SBMConfig, generate_sbm


GeneratorFn = Callable[[Any, int], tuple[nx.Graph, np.ndarray, dict[str, Any]]]

DEFAULT_DATASET_ROOT = Path("data/cache/synthetic_benchmark")
DEFAULT_NUM_BASE_GRAPHS = 5
DEFAULT_NOISE_FRACS = [i / 100 for i in range(5, 50, 5)]


def _relative_to_root(path: Path, root: Path) -> str:
    """
    Store metadata paths relative to the dataset root so the zip artifact is portable.
    """
    return path.relative_to(root).as_posix()


def _filter_metadata_for_csv(
    metadata: dict[str, Any],
    stats: dict[str, Any],
) -> dict[str, Any]:
    """
    Remove keys that should not overwrite row fields or graph-instance stats.

    We keep family-specific configuration fields like p_in, p_out, tau1, tau2, mu,
    community_sizes, etc., and also keep perturbation-specific fields like
    num_edges_original, num_edges_removed, removed_edge_fraction, perturbation_seed.

    We drop:
    - row-level fields that are passed directly to make_metadata_row
    - graph-instance stats fields computed by characterize.py
    - the nested raw config dict, which is noisy in the metadata CSV
    """
    row_level_keys = {
        "graph_id",
        "family",
        "base_graph_id",
        "seed",
        "noise_type",
        "noise_code",
        "noise_frac",
        "edge_path",
        "label_path",
    }

    keys_to_drop = row_level_keys | set(stats.keys()) | {"config"}

    return {
        key: value
        for key, value in metadata.items()
        if key not in keys_to_drop
    }


def _save_graph_instance(
    *,
    dataset_root: Path,
    family: str,
    noise_type: str,
    noise_frac: float,
    base_graph_id: str,
    seed: int,
    G: nx.Graph,
    labels: np.ndarray,
    extra_metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Save one graph instance and return its metadata row.
    """
    stats = compute_all_graph_stats(G, labels)

    noise_code = format_noise_code(noise_frac)
    graph_id = make_graph_id(base_graph_id, noise_code, noise_type, family)
    paths = make_output_paths(dataset_root, family, noise_type, graph_id)

    save_graph_edgelist(G, labels, paths["edge_path"])
    save_labels(labels, paths["label_path"])

    row = make_metadata_row(
        graph_id=graph_id,
        family=family,
        base_graph_id=base_graph_id,
        seed=seed,
        noise_type=noise_type,
        noise_code=noise_code,
        noise_frac=noise_frac,
        edge_path=_relative_to_root(paths["edge_path"], dataset_root),
        label_path=_relative_to_root(paths["label_path"], dataset_root),
        stats=stats,
        family_metadata=_filter_metadata_for_csv(extra_metadata, stats),
    )

    return row


def build_family_dataset(
    *,
    family: str,
    config: Any,
    generator_fn: GeneratorFn,
    dataset_root: Path,
    num_base_graphs: int,
    noise_fracs: list[float],
) -> Path:
    """
    Build the full clean + noisy dataset for one graph family and write its metadata CSV.
    """
    rows: list[dict[str, Any]] = []

    for index in range(1, num_base_graphs + 1):
        requested_seed = index - 1
        base_graph_id = format_base_graph_id(index)

        print(f"[{family}] generating clean base graph {base_graph_id} (requested seed={requested_seed})")
        G_clean, labels, clean_metadata = generator_fn(config, requested_seed)

        actual_seed = int(clean_metadata["seed"])

        clean_row = _save_graph_instance(
            dataset_root=dataset_root,
            family=family,
            noise_type="clean",
            noise_frac=0.0,
            base_graph_id=base_graph_id,
            seed=actual_seed,
            G=G_clean,
            labels=labels,
            extra_metadata=clean_metadata,
        )
        rows.append(clean_row)

        print(f"[{family}] building random perturbation chain for {base_graph_id}")
        random_chain = build_noise_chain(
            G_clean=G_clean,
            noise_type="random",
            noise_fracs=noise_fracs,
            seed=actual_seed,
        )

        for noise_frac, G_perturbed, perturb_metadata in random_chain:
            combined_metadata = {**clean_metadata, **perturb_metadata}

            row = _save_graph_instance(
                dataset_root=dataset_root,
                family=family,
                noise_type="random",
                noise_frac=noise_frac,
                base_graph_id=base_graph_id,
                seed=actual_seed,
                G=G_perturbed,
                labels=labels,
                extra_metadata=combined_metadata,
            )
            rows.append(row)

        print(f"[{family}] building targeted_betweenness perturbation chain for {base_graph_id}")
        targeted_chain = build_noise_chain(
            G_clean=G_clean,
            noise_type="targeted_betweenness",
            noise_fracs=noise_fracs,
        )

        for noise_frac, G_perturbed, perturb_metadata in targeted_chain:
            combined_metadata = {**clean_metadata, **perturb_metadata}

            row = _save_graph_instance(
                dataset_root=dataset_root,
                family=family,
                noise_type="targeted_betweenness",
                noise_frac=noise_frac,
                base_graph_id=base_graph_id,
                seed=actual_seed,
                G=G_perturbed,
                labels=labels,
                extra_metadata=combined_metadata,
            )
            rows.append(row)

    metadata_path = dataset_root / "metadata" / f"graph_index_{family}.csv"
    write_metadata_csv(rows, metadata_path)

    print(f"[{family}] wrote {len(rows)} metadata rows to {metadata_path}")
    return metadata_path


def build_dataset(
    *,
    dataset_root: Path,
    num_base_graphs: int,
    noise_fracs: list[float],
) -> dict[str, Path]:
    """
    Build the full first-pass synthetic benchmark dataset.
    """
    dataset_root.mkdir(parents=True, exist_ok=True)

    sbm_config = SBMConfig()
    lfr_config = LFRConfig()

    sbm_metadata_path = build_family_dataset(
        family="sbm",
        config=sbm_config,
        generator_fn=generate_sbm,
        dataset_root=dataset_root,
        num_base_graphs=num_base_graphs,
        noise_fracs=noise_fracs,
    )

    lfr_metadata_path = build_family_dataset(
        family="lfr",
        config=lfr_config,
        generator_fn=generate_lfr,
        dataset_root=dataset_root,
        num_base_graphs=num_base_graphs,
        noise_fracs=noise_fracs,
    )

    return {
        "sbm_metadata_path": sbm_metadata_path,
        "lfr_metadata_path": lfr_metadata_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the synthetic SBM/LFR benchmark dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory where the generated dataset artifact will be written.",
    )
    parser.add_argument(
        "--num-base-graphs",
        type=int,
        default=DEFAULT_NUM_BASE_GRAPHS,
        help="Number of clean base graphs to generate per family.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Build a very small dataset for quick validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke_test:
        num_base_graphs = 1
        noise_fracs = [0.05, 0.10]
        print("Running in smoke-test mode")
    else:
        num_base_graphs = args.num_base_graphs
        noise_fracs = DEFAULT_NOISE_FRACS

    paths = build_dataset(
        dataset_root=args.dataset_root,
        num_base_graphs=num_base_graphs,
        noise_fracs=noise_fracs,
    )

    print("\nFinished building synthetic benchmark dataset")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()