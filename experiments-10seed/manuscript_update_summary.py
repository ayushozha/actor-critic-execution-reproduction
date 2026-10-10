"""Manuscript values computed solely in the original scientific runtime."""
from __future__ import print_function

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execution_study import HERE, digest, identity, utc, write_json


def main():
    data = json.loads((HERE / "analysis-summary.json").read_text(encoding="utf-8"))
    rows = {(row["scenario"], row["algorithm"], row["regime"]): row["metrics"]["objective"] for row in data["scenario_group_results"]}
    comparisons = []
    for row in data["primary_and_regime_comparisons"]:
        copy = dict(row)
        copy["treatment_mean"] = float(np.mean(row["treatment_seed_values"]))
        copy["comparator_mean"] = float(np.mean(row["comparator_seed_values"]))
        copy["interval_includes_zero"] = row["ci95_lower"] <= 0 <= row["ci95_upper"]
        comparisons.append(copy)
    signs = []
    for algorithm in ["ddpg", "td3"]:
        for scenario in ["high_impact", "low_impact", "concave_impact", "convex_impact", "thin_liquidity", "joint_shift", "high_liquidity"]:
            nominal = rows[(scenario, algorithm, "nominal")]["mean"]
            randomized = rows[(scenario, algorithm, "randomized")]["mean"]
            difference = randomized - nominal
            signs.append({"algorithm": algorithm, "scenario": scenario, "nominal_mean": nominal, "randomized_mean": randomized,
                          "randomized_minus_nominal": difference, "lower_mean_regime": "randomized" if difference < 0 else "nominal" if difference > 0 else "equal"})
    nominal_groups = [{"algorithm": algorithm, "regime": regime, "objective": stat} for (scenario, algorithm, regime), stat in rows.items() if scenario == "nominal"]
    output = {"computed_utc": utc(), "runtime": identity(), "source_sha256": digest(HERE / "analysis-summary.json"),
              "training_seed_count": 10, "training_runs": 40, "training_transitions": 800000, "unique_episode_evaluations": 344064,
              "primary_and_regime_comparisons": comparisons, "cross_algorithm_secondary": data["cross_algorithm_secondary"],
              "nominal_learned_objectives": nominal_groups,
              "nominal_learned_mean_range": [float(np.min([row["objective"]["mean"] for row in nominal_groups])), float(np.max([row["objective"]["mean"] for row in nominal_groups]))],
              "nominal_td3_randomized_baseline_contrasts": [row for row in data["scenario_baseline_contrasts"] if row["scenario"] == "nominal" and row["treatment"] == "td3_randomized"],
              "nonnominal_regime_directions": signs,
              "joint_shift_learned_objectives": [{"algorithm": algorithm, "regime": regime, "objective": stat} for (scenario, algorithm, regime), stat in rows.items() if scenario == "joint_shift"],
              "nominal_and_joint_baselines": [row for row in data["baseline_results"] if row["scenario"] in ["nominal", "joint_shift"]]}
    write_json(HERE / "manuscript-update-summary.json", output)
    print(json.dumps({"primary_means": comparisons, "nominal_range": output["nominal_learned_mean_range"], "regime_directions": signs}, indent=2), flush=True)


if __name__ == "__main__":
    main()
