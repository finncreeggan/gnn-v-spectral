from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASET_ROOT = Path("data/cache/synthetic")
DEFAULT_NUM_BASE_GRAPHS = 5
DEFAULT_NOISE_FRACS = [i / 100 for i in range(5, 50, 5)]
EXPECTED_EDGE_COLUMNS = ["src", "dst", "same_comm", "comm_pair"]
EXPECTED_NOISE_TYPES = ["clean", "random", "targeted_betweenness"]


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def warn(warnings: list[str], message: str) -> None:
    warnings.append(message)


def load_metadata(dataset_root: Path, family: str) -> pd.DataFrame:
    path = dataset_root / "metadata" / f"graph_index_{family}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {path}")
    return pd.read_csv(path)


def canonical_edge_set(df: pd.DataFrame) -> set[tuple[int, int]]:
    return {
        (min(int(src), int(dst)), max(int(src), int(dst)))
        for src, dst in zip(df["src"], df["dst"])
    }


def load_labels(path: Path) -> np.ndarray:
    labels = np.load(path)
    return np.asarray(labels, dtype=np.int64)


def validate_edge_file(
    *,
    edge_df: pd.DataFrame,
    labels: np.ndarray,
    row: pd.Series,
    errors: list[str],
) -> None:
    graph_id = row["graph_id"]
    n_nodes = int(row["n_nodes"])
    num_edges = int(row["num_edges"])

    if list(edge_df.columns) != EXPECTED_EDGE_COLUMNS:
        fail(
            errors,
            f"{graph_id}: edge CSV columns are {list(edge_df.columns)}, expected {EXPECTED_EDGE_COLUMNS}.",
        )
        return

    if len(edge_df) != num_edges:
        fail(
            errors,
            f"{graph_id}: metadata num_edges={num_edges}, but edge CSV has {len(edge_df)} rows.",
        )

    src = edge_df["src"].to_numpy(dtype=np.int64)
    dst = edge_df["dst"].to_numpy(dtype=np.int64)

    if np.any(src == dst):
        fail(errors, f"{graph_id}: found self-loops in edge CSV.")

    if np.any(src > dst):
        fail(errors, f"{graph_id}: edge CSV should store undirected edges with src < dst.")

    if np.any(src < 0) or np.any(dst < 0) or np.any(src >= n_nodes) or np.any(dst >= n_nodes):
        fail(errors, f"{graph_id}: edge CSV contains node ids outside [0, {n_nodes - 1}].")

    if edge_df.duplicated(subset=["src", "dst"]).any():
        fail(errors, f"{graph_id}: duplicate edges found in edge CSV.")

    computed_same = (labels[src] == labels[dst]).astype(np.int64)
    saved_same = edge_df["same_comm"].to_numpy(dtype=np.int64)
    if not np.array_equal(computed_same, saved_same):
        fail(errors, f"{graph_id}: same_comm column does not match labels.")

    computed_pairs = np.array(
        [
            f"{min(int(labels[u]), int(labels[v]))}_{max(int(labels[u]), int(labels[v]))}"
            for u, v in zip(src, dst)
        ],
        dtype=object,
    )
    saved_pairs = edge_df["comm_pair"].astype(str).to_numpy()
    if not np.array_equal(computed_pairs, saved_pairs):
        fail(errors, f"{graph_id}: comm_pair column does not match labels.")

def normalize_noise_code(value: object) -> str:
    """
    Normalize noise_code values read from CSV so that 0, 0.0, '0', and '000'
    all become '000'.
    """
    if pd.isna(value):
        raise ValueError("noise_code is missing.")

    if isinstance(value, str):
        value = value.strip()
        if value == "":
            raise ValueError("noise_code is empty.")
        return f"{int(float(value)):03d}"

    return f"{int(float(value)):03d}"

def validate_family(
    *,
    family: str,
    dataset_root: Path,
    df: pd.DataFrame,
    num_base_graphs: int,
    noise_fracs: list[float],
    errors: list[str],
    warnings: list[str],
) -> None:
    required_columns = {
        "graph_id",
        "family",
        "base_graph_id",
        "seed",
        "noise_type",
        "noise_code",
        "noise_frac",
        "edge_path",
        "label_path",
        "n_nodes",
        "num_edges",
        "avg_degree",
        "num_communities",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        fail(errors, f"[{family}] metadata missing required columns: {sorted(missing_columns)}")
        return

    if df["graph_id"].duplicated().any():
        dupes = df.loc[df["graph_id"].duplicated(), "graph_id"].tolist()
        fail(errors, f"[{family}] duplicate graph_id values found: {dupes[:5]}")

    expected_total_rows = num_base_graphs * (1 + 2 * len(noise_fracs))
    if len(df) != expected_total_rows:
        fail(
            errors,
            f"[{family}] expected {expected_total_rows} metadata rows, found {len(df)}.",
        )

    edge_cache: dict[str, pd.DataFrame] = {}
    label_cache: dict[str, np.ndarray] = {}
    edge_set_cache: dict[str, set[tuple[int, int]]] = {}

    for _, row in df.iterrows():
        graph_id = str(row["graph_id"])
        row_family = str(row["family"])
        noise_type = str(row["noise_type"])
        noise_frac = float(row["noise_frac"])
        edge_rel = Path(str(row["edge_path"]))
        label_rel = Path(str(row["label_path"]))
        edge_path = dataset_root / edge_rel
        label_path = dataset_root / label_rel

        if row_family != family:
            fail(errors, f"{graph_id}: family column is {row_family}, expected {family}.")

        if noise_type not in EXPECTED_NOISE_TYPES:
            fail(errors, f"{graph_id}: unexpected noise_type={noise_type}.")

        
        normalized_noise_code = normalize_noise_code(row["noise_code"])

        if noise_type == "clean":
            if noise_frac != 0.0:
                fail(errors, f"{graph_id}: clean graph should have noise_frac=0.0, got {noise_frac}.")
            if normalized_noise_code != "000":
                fail(errors, f"{graph_id}: clean graph should have noise_code='000'.")
        else:
            expected_noise_code = f"{int(round(noise_frac * 100)):03d}"
            if normalized_noise_code != expected_noise_code:
                fail(
                    errors,
                    f"{graph_id}: noise_code={normalized_noise_code}, expected {expected_noise_code}.",
                )

        if edge_rel.is_absolute() or label_rel.is_absolute():
            warn(warnings, f"{graph_id}: metadata paths are absolute, relative paths are preferred.")

        if not edge_path.exists():
            fail(errors, f"{graph_id}: missing edge file {edge_path}.")
            continue

        if not label_path.exists():
            fail(errors, f"{graph_id}: missing label file {label_path}.")
            continue

        labels = load_labels(label_path)
        label_cache[graph_id] = labels

        if labels.ndim != 1:
            fail(errors, f"{graph_id}: labels file is not 1D.")
            continue

        if len(labels) != int(row["n_nodes"]):
            fail(
                errors,
                f"{graph_id}: metadata n_nodes={int(row['n_nodes'])}, but labels length is {len(labels)}.",
            )

        edge_df = pd.read_csv(edge_path)
        edge_cache[graph_id] = edge_df
        edge_set_cache[graph_id] = canonical_edge_set(edge_df)

        validate_edge_file(edge_df=edge_df, labels=labels, row=row, errors=errors)

    base_groups = df.groupby("base_graph_id")
    if len(base_groups) != num_base_graphs:
        fail(
            errors,
            f"[{family}] expected {num_base_graphs} base_graph_id values, found {len(base_groups)}.",
        )

    for base_graph_id, group in base_groups:
        clean_rows = group[group["noise_type"] == "clean"]
        random_rows = group[group["noise_type"] == "random"].sort_values("noise_frac")
        targeted_rows = group[group["noise_type"] == "targeted_betweenness"].sort_values("noise_frac")

        if len(clean_rows) != 1:
            fail(errors, f"[{family}/{base_graph_id}] expected exactly 1 clean row, found {len(clean_rows)}.")

        if len(random_rows) != len(noise_fracs):
            fail(
                errors,
                f"[{family}/{base_graph_id}] expected {len(noise_fracs)} random rows, found {len(random_rows)}.",
            )

        if len(targeted_rows) != len(noise_fracs):
            fail(
                errors,
                f"[{family}/{base_graph_id}] expected {len(noise_fracs)} targeted rows, found {len(targeted_rows)}.",
            )

        if len(clean_rows) != 1:
            continue

        clean_row = clean_rows.iloc[0]
        clean_graph_id = str(clean_row["graph_id"])
        clean_num_edges = int(clean_row["num_edges"])
        clean_labels = label_cache.get(clean_graph_id)
        clean_edges = edge_set_cache.get(clean_graph_id)

        if clean_labels is None or clean_edges is None:
            continue

        for noisy_rows, noise_type in [(random_rows, "random"), (targeted_rows, "targeted_betweenness")]:
            prev_edges = clean_edges
            prev_num_edges = clean_num_edges

            for _, noisy_row in noisy_rows.iterrows():
                graph_id = str(noisy_row["graph_id"])
                labels = label_cache.get(graph_id)
                edges = edge_set_cache.get(graph_id)

                if labels is None or edges is None:
                    continue

                if not np.array_equal(labels, clean_labels):
                    fail(errors, f"[{family}/{base_graph_id}] labels changed for {graph_id}.")

                if not edges.issubset(prev_edges):
                    fail(
                        errors,
                        f"[{family}/{base_graph_id}] edge set for {graph_id} is not a subset of the previous graph in the {noise_type} chain.",
                    )

                if int(noisy_row["num_edges"]) > prev_num_edges:
                    fail(
                        errors,
                        f"[{family}/{base_graph_id}] num_edges increased in the {noise_type} chain at {graph_id}.",
                    )

                num_edges_original = noisy_row.get("num_edges_original", np.nan)
                if pd.notna(num_edges_original) and int(num_edges_original) != clean_num_edges:
                    fail(
                        errors,
                        f"[{family}/{base_graph_id}] {graph_id} has num_edges_original={num_edges_original}, expected {clean_num_edges}.",
                    )

                num_edges_removed = noisy_row.get("num_edges_removed", np.nan)
                if pd.notna(num_edges_removed):
                    actual_removed = clean_num_edges - int(noisy_row["num_edges"])
                    if int(num_edges_removed) != actual_removed:
                        fail(
                            errors,
                            f"[{family}/{base_graph_id}] {graph_id} metadata num_edges_removed={num_edges_removed}, actual removed={actual_removed}.",
                        )

                removed_edge_fraction = noisy_row.get("removed_edge_fraction", np.nan)
                if pd.notna(removed_edge_fraction):
                    expected_fraction = (clean_num_edges - int(noisy_row["num_edges"])) / clean_num_edges
                    if abs(float(removed_edge_fraction) - expected_fraction) > 1e-12:
                        fail(
                            errors,
                            f"[{family}/{base_graph_id}] {graph_id} removed_edge_fraction mismatch.",
                        )

                prev_edges = edges
                prev_num_edges = int(noisy_row["num_edges"])


def print_summary(*, family: str, df: pd.DataFrame) -> None:
    clean_df = df[df["noise_type"] == "clean"].copy()

    print(f"\n[{family}] summary")
    print(f"rows: {len(df)}")
    print(f"base graphs: {clean_df['base_graph_id'].nunique()}")
    print("noise counts:")
    print(df["noise_type"].value_counts().sort_index().to_string())

    if not clean_df.empty:
        print("\nclean graph summary:")
        print(
            f"  n_nodes mean ± std: "
            f"{clean_df['n_nodes'].mean():.2f} ± {clean_df['n_nodes'].std(ddof=0):.2f}"
        )
        print(
            f"  avg_degree mean ± std: "
            f"{clean_df['avg_degree'].mean():.3f} ± {clean_df['avg_degree'].std(ddof=0):.3f}"
        )
        print(
            f"  num_communities min/mean/max: "
            f"{int(clean_df['num_communities'].min())} / "
            f"{clean_df['num_communities'].mean():.2f} / "
            f"{int(clean_df['num_communities'].max())}"
        )

    high_noise = df[df["noise_frac"] == max(DEFAULT_NOISE_FRACS)]
    if not high_noise.empty and {"largest_cc_fraction", "num_connected_components"} <= set(df.columns):
        print("\nhigh-noise structural summary (noise_frac = 0.45):")
        grouped = (
            high_noise.groupby("noise_type")[["largest_cc_fraction", "num_connected_components"]]
            .mean()
            .round(4)
        )
        print(grouped.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the generated synthetic benchmark dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory of the generated dataset artifact.",
    )
    parser.add_argument(
        "--num-base-graphs",
        type=int,
        default=DEFAULT_NUM_BASE_GRAPHS,
        help="Expected number of clean base graphs per family.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    noise_fracs = DEFAULT_NOISE_FRACS

    errors: list[str] = []
    warnings: list[str] = []

    try:
        sbm_df = load_metadata(dataset_root, "sbm")
        lfr_df = load_metadata(dataset_root, "lfr")
    except FileNotFoundError as exc:
        print(f"Validation failed: {exc}")
        sys.exit(1)

    validate_family(
        family="sbm",
        dataset_root=dataset_root,
        df=sbm_df,
        num_base_graphs=args.num_base_graphs,
        noise_fracs=noise_fracs,
        errors=errors,
        warnings=warnings,
    )

    validate_family(
        family="lfr",
        dataset_root=dataset_root,
        df=lfr_df,
        num_base_graphs=args.num_base_graphs,
        noise_fracs=noise_fracs,
        errors=errors,
        warnings=warnings,
    )

    print_summary(family="sbm", df=sbm_df)
    print_summary(family="lfr", df=lfr_df)

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for message in warnings[:20]:
            print(f"  - {message}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more")

    if errors:
        print(f"\nValidation FAILED with {len(errors)} error(s):")
        for message in errors[:30]:
            print(f"  - {message}")
        if len(errors) > 30:
            print(f"  ... and {len(errors) - 30} more")
        sys.exit(1)

    print("\nValidation PASSED.")


if __name__ == "__main__":
    main()