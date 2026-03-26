"""Utility script for calibrating LFR generator defaults."""
from __future__ import annotations

from dataclasses import replace
from statistics import mean, pstdev

from data.generators.lfr import LFRConfig, generate_lfr


def realized_avg_degree(num_nodes: int, num_edges: int) -> float:
    return 2 * num_edges / num_nodes


def calibrate_lfr_average_degree() -> None:
    base_config = LFRConfig()

    candidate_average_degrees = [16, 18, 20, 22, 24, 25]
    seeds = [0, 1, 2, 3, 4]

    results = []

    for avg_deg in candidate_average_degrees:
        realized_degrees = []
        num_communities = []
        edge_counts = []
        successful_seeds = []
        failed_seeds = []

        for seed in seeds:
            config = replace(base_config, average_degree=avg_deg)

            try:
                G, labels, metadata = generate_lfr(config, seed=seed)
                rd = realized_avg_degree(G.number_of_nodes(), G.number_of_edges())

                realized_degrees.append(rd)
                num_communities.append(metadata["num_communities"])
                edge_counts.append(G.number_of_edges())
                successful_seeds.append(seed)

            except Exception as exc:
                failed_seeds.append((seed, str(exc)))

        row = {
            "average_degree_input": avg_deg,
            "num_successes": len(successful_seeds),
            "num_failures": len(failed_seeds),
            "success_rate": len(successful_seeds) / len(seeds),
            "mean_realized_avg_degree": mean(realized_degrees) if realized_degrees else None,
            "std_realized_avg_degree": pstdev(realized_degrees) if len(realized_degrees) > 1 else 0.0 if realized_degrees else None,
            "mean_num_communities": mean(num_communities) if num_communities else None,
            "mean_num_edges": mean(edge_counts) if edge_counts else None,
            "successful_seeds": successful_seeds,
            "failed_seeds": failed_seeds,
        }
        results.append(row)

    print("\nLFR calibration results\n")
    for row in results:
        print("-" * 80)
        print(f"average_degree input:      {row['average_degree_input']}")
        print(f"successes / failures:      {row['num_successes']} / {row['num_failures']}")
        print(f"success rate:              {row['success_rate']:.2f}")
        print(f"mean realized avg degree:  {row['mean_realized_avg_degree']}")
        print(f"std realized avg degree:   {row['std_realized_avg_degree']}")
        print(f"mean number of communities:{row['mean_num_communities']}")
        print(f"mean number of edges:      {row['mean_num_edges']}")
        print(f"successful seeds:          {row['successful_seeds']}")
        if row["failed_seeds"]:
            print("failed seeds:")
            for seed, err in row["failed_seeds"]:
                print(f"  seed={seed}: {err}")


if __name__ == "__main__":
    calibrate_lfr_average_degree()