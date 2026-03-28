"""
Build master experiment tables from the graph index CSVs delivered by Jamie.

Reads graph_index_sbm.csv and graph_index_lfr.csv from the metadata folder,
resolves local file paths for edges, labels, and the three eigenspectrum
assets, and writes enriched metadata tables that serve as the single source
of truth for downstream experiment runners.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── directory layout beneath data/synthetic_benchmark/ ──────────────────────
NOISE_TYPE_DIRS = {
    "clean": "clean",
    "random": "random",
    "targeted_betweenness": "targeted_betweenness",
}

FAMILIES = ("sbm", "lfr")


def _resolve_asset_paths(
    row: pd.Series,
    data_root: Path,
) -> pd.Series:
    """Attach absolute file paths for edges, labels, and spectra to one row."""
    family = row["family"]
    noise_dir = NOISE_TYPE_DIRS[row["structural_noise_type"]]
    graph_id = row["graph_id"]

    family_root = data_root / family / noise_dir

    row["edge_path"] = str(family_root / "edges" / f"{graph_id}.csv")
    row["label_path"] = str(family_root / "labels" / f"{graph_id}_labels.npy")
    row["whole_spectrum_path"] = str(
        family_root / "spectra" / "whole" / f"{graph_id}_whole_spectrum.csv"
    )
    row["kcut_spectrum_path"] = str(
        family_root / "spectra" / "kcut" / f"{graph_id}_kcut_spectrum.csv"
    )
    row["regularized_spectrum_path"] = str(
        family_root / "spectra" / "regularized" / f"{graph_id}_regularized_spectrum.csv"
    )
    return row


def build_structural_noise_table(
    data_root: str | Path,
    metadata_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Build the unified metadata table for the structural-noise experiment.

    Parameters
    ----------
    data_root : path
        Root of the ``data/synthetic_benchmark/`` tree.
    metadata_dir : path, optional
        Folder containing ``graph_index_sbm.csv`` and ``graph_index_lfr.csv``.
        Defaults to ``data_root / "metadata"``.

    Returns
    -------
    pd.DataFrame
        One row per graph instance with resolved asset paths.
    """
    data_root = Path(data_root)
    metadata_dir = Path(metadata_dir) if metadata_dir else data_root / "metadata"

    frames = []
    for family in FAMILIES:
        index_path = metadata_dir / f"graph_index_{family}.csv"
        if not index_path.exists():
            logger.warning("Index file not found: %s – skipping %s", index_path, family)
            continue
        df = pd.read_csv(index_path)
        df["family"] = family
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No graph index CSVs found in {metadata_dir}. "
            "Ensure Jamie's dataset has been unzipped."
        )

    table = pd.concat(frames, ignore_index=True)
    table = table.apply(_resolve_asset_paths, axis=1, data_root=data_root)

    logger.info(
        "Built structural-noise metadata table: %d rows (%s)",
        len(table),
        ", ".join(f"{f}={len(table[table.family==f])}" for f in FAMILIES),
    )
    return table


def build_feature_experiment_table(
    structural_table: pd.DataFrame,
    features_root: str | Path,
    structural_noise_codes: tuple[str, ...] = ("015", "030", "045"),
    informativeness_codes: tuple[str, ...] = (
        "100", "090", "080", "070", "060",
        "050", "040", "030", "020", "010", "000",
    ),
) -> pd.DataFrame:
    """Build the experiment table for the feature-informativeness experiment.

    Filters the structural-noise table to the selected noise levels, then
    cross-joins with the 11 feature-informativeness levels and resolves
    feature-matrix paths.

    Parameters
    ----------
    structural_table : pd.DataFrame
        Output of :func:`build_structural_noise_table`.
    features_root : path
        Root of the ``data/synthetic_benchmark/features/`` tree.
    structural_noise_codes : tuple of str
        Which structural-noise codes to include (default: low/med/high).
    informativeness_codes : tuple of str
        Feature-informativeness code values (11 levels, endpoints inclusive).

    Returns
    -------
    pd.DataFrame
        One row per graph x feature-informativeness level.
    """
    features_root = Path(features_root)

    # Filter to selected structural noise levels (exclude clean graphs)
    selected = structural_table[
        structural_table["structural_noise_code"].astype(str).isin(structural_noise_codes)
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No rows matched structural_noise_codes={structural_noise_codes}. "
            "Check that the metadata table has the expected code values."
        )

    rows = []
    for _, graph_row in selected.iterrows():
        for code in informativeness_codes:
            frac = int(code) / 100.0
            new_row = graph_row.copy()
            new_row["feature_informativeness_code"] = code
            new_row["feature_informativeness_frac"] = frac
            new_row["feature_noise_frac"] = 1.0 - frac

            family = graph_row["family"]
            noise_dir = NOISE_TYPE_DIRS[graph_row["structural_noise_type"]]
            graph_id = graph_row["graph_id"]

            feature_filename = (
                f"{graph_id}_feature_informativeness_{code}_features.npy"
            )
            new_row["feature_path"] = str(
                features_root / family / noise_dir / feature_filename
            )
            rows.append(new_row)

    feature_table = pd.DataFrame(rows).reset_index(drop=True)

    logger.info(
        "Built feature-informativeness table: %d rows "
        "(%d graphs x %d informativeness levels)",
        len(feature_table),
        len(selected),
        len(informativeness_codes),
    )
    return feature_table


def save_metadata_tables(
    data_root: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Convenience function: build both tables and write them to disk.

    Returns the paths to the saved structural-noise and feature tables.
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir) if output_dir else data_root / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)

    structural = build_structural_noise_table(data_root)
    structural_path = output_dir / "structural_noise_experiment_table.csv"
    structural.to_csv(structural_path, index=False)
    logger.info("Saved structural-noise table to %s", structural_path)

    features_root = data_root / "features"
    feature = build_feature_experiment_table(structural, features_root)
    feature_path = output_dir / "feature_informativeness_experiment_table.csv"
    feature.to_csv(feature_path, index=False)
    logger.info("Saved feature-informativeness table to %s", feature_path)

    return structural_path, feature_path


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/synthetic_benchmark")
    save_metadata_tables(root)
