"""
Build master experiment tables from the graph index CSVs.

Reads graph_index_sbm.csv and graph_index_lfr.csv from the metadata folder,
adds the spectra_path column needed by load_graph_data, and writes enriched
metadata tables that serve as the single source of truth for downstream
experiment runners.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── directory layout beneath data/cache/synthetic/ ───────────────────────────
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
    """Attach resolved file paths for edges, labels, and spectra to one row."""
    family = row["family"]
    noise_dir = NOISE_TYPE_DIRS[row["noise_type"]]
    graph_id = row["graph_id"]

    relative_root = Path(family) / noise_dir

    row["edge_path"] = str(relative_root / "edges" / f"{graph_id}.csv")
    row["label_path"] = str(relative_root / "labels" / f"{graph_id}_labels.npy")
    row["spectra_path"] = str(relative_root / "spectra" / f"{graph_id}.pt")
    return row


def build_structural_noise_table(
    data_root: str | Path,
    metadata_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Build the unified metadata table for the structural-noise experiment.

    Also saves enriched graph_index CSVs (with spectra_path column) so that
    load_graph_data can find the precomputed eigenspectra.

    Parameters
    ----------
    data_root : path
        Root of the ``data/cache/synthetic/`` tree.
    metadata_dir : path, optional
        Folder containing ``graph_index_sbm.csv`` and ``graph_index_lfr.csv``.
        Defaults to ``data_root / "metadata"``.
    output_dir : path, optional
        Where to write enriched CSVs.  Defaults to ``data_root / "metadata"``.

    Returns
    -------
    pd.DataFrame
        One row per graph instance with resolved asset paths and
        structural_noise_* column names.
    """
    data_root = Path(data_root)
    metadata_dir = Path(metadata_dir) if metadata_dir else data_root / "metadata"
    output_dir = Path(output_dir) if output_dir else data_root / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)

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
            "Ensure the dataset has been unzipped."
        )

    table = pd.concat(frames, ignore_index=True)

    # Resolve asset paths (edge_path, label_path, spectra_path)
    table = table.apply(_resolve_asset_paths, axis=1, data_root=data_root)

    # Save enriched graph_index CSVs so load_graph_data can read spectra_path
    for family in FAMILIES:
        family_df = table[table["family"] == family]
        if family_df.empty:
            continue
        enriched_path = output_dir / f"graph_index_{family}.csv"
        family_df.to_csv(enriched_path, index=False)
        logger.info("Saved enriched graph index: %s (%d rows)", enriched_path, len(family_df))

    # Rename columns for the experiment table
    table = table.rename(columns={
        "noise_type": "structural_noise_type",
        "noise_code": "structural_noise_code",
        "noise_frac": "structural_noise_frac",
    })

    logger.info(
        "Built structural-noise metadata table: %d rows (%s)",
        len(table),
        ", ".join(f"{f}={len(table[table.family==f])}" for f in FAMILIES),
    )
    return table


def build_feature_experiment_table(
    structural_table: pd.DataFrame,
    features_root: str | Path,
    structural_noise_codes: tuple[str | int, ...] = ("015", "030", "045"),
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
        Root of the ``data/cache/synthetic/features/`` tree.
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
    # Handle both int (15) and zero-padded string ("015") codes
    str_codes = {str(c) for c in structural_noise_codes}
    int_codes = set()
    for c in structural_noise_codes:
        try:
            int_codes.add(int(c))
        except (ValueError, TypeError):
            pass
    selected = structural_table[
        structural_table["structural_noise_code"].astype(str).isin(str_codes)
        | structural_table["structural_noise_code"].isin(int_codes)
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
                Path("features") / family / noise_dir / feature_filename
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

    structural = build_structural_noise_table(data_root, output_dir=output_dir)
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
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/cache/synthetic")
    save_metadata_tables(root)
