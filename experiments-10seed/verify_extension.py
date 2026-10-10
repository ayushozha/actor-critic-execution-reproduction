"""Independent extension integrity checks in the original scientific runtime."""
from __future__ import print_function

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from execution_study import HERE, ROOT, digest, identity, utc, write_json


def main():
    plan = json.loads((HERE / "extension-plan.json").read_text(encoding="utf-8"))
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    parent = ROOT / "experiments"
    original_cfg = json.loads((parent / "config.json").read_text(encoding="utf-8"))
    assert [key for key in cfg if cfg[key] != original_cfg[key]] == ["training_seeds"]
    assert cfg["training_seeds"] == [11, 23, 37, 53, 71, 89, 107, 127, 149, 173]
    assert digest(HERE / "execution_study.py") == digest(parent / "execution_study.py")
    for row in plan["parent_inputs"]:
        assert digest(ROOT / row["path"]) == row["sha256"], row["path"]
    for row in plan["preserved_run_copies"]:
        assert digest(ROOT / row["copy"]) == row["sha256"], row["copy"]
    arrays_checked = 0
    for source in sorted((parent / "evaluation").glob("*.npz")):
        with np.load(str(source)) as a, np.load(str(HERE / "evaluation" / source.name)) as b:
            assert sorted(a.files) == sorted(b.files)
            for key in a.files:
                assert np.array_equal(a[key], b[key]), (source.name, key)
                arrays_checked += 1
    original_rows = json.loads((parent / "evaluation" / "seed-means.json").read_text(encoding="utf-8"))
    combined_rows = json.loads((HERE / "evaluation" / "seed-means.json").read_text(encoding="utf-8"))
    keys = lambda row: (row["scenario"], row["algorithm"], row["regime"], row["seed"])
    combined = {keys(row): row for row in combined_rows}
    assert all(combined[keys(row)] == row for row in original_rows)
    assert len(combined_rows) == 42 * 8
    summary = json.loads((HERE / "analysis-summary.json").read_text(encoding="utf-8"))
    assert len(summary["training_runs"]) == 40 and summary["total_training_transitions"] == 800000
    assert summary["total_episode_evaluations"] == 344064 and summary["completion_failures"] == 0
    nonnominal = [scenario["name"] for scenario in cfg["scenarios"] if scenario["name"] != "nominal"]
    differences = []
    for seed in cfg["training_seeds"]:
        treatment = [combined[(s, "td3", "randomized", seed)]["mean_objective"] for s in nonnominal]
        comparator = [combined[(s, "td3", "nominal", seed)]["mean_objective"] for s in nonnominal]
        differences.append(float(np.mean(treatment) - np.mean(comparator)))
    primary = [row for row in summary["primary_and_regime_comparisons"] if row["primary"]][0]
    mean = float(np.mean(differences))
    half = 2.2621571628540993 * float(np.std(differences, ddof=1)) / math.sqrt(10)
    assert abs(mean - primary["mean"]) < 1e-14
    assert abs(mean - half - primary["ci95_lower"]) < 1e-14
    assert abs(mean + half - primary["ci95_upper"]) < 1e-14
    assert primary["seed_count"] == 10 and primary["ci_method"] == "paired training-seed t interval df=9"
    with (HERE / "evaluation" / "raw-episodes.csv").open(newline="") as stream:
        raw_count = sum(1 for row in csv.DictReader(stream))
    assert raw_count == 344064
    report = {"verified_utc": utc(), "result": "PASS", "runtime": identity(), "parent_files_preserved": len(plan["parent_inputs"]),
              "original_run_copy_files_verified": len(plan["preserved_run_copies"]), "original_npz_arrays_equal": arrays_checked,
              "original_seed_mean_rows_equal": len(original_rows), "raw_episode_rows": raw_count, "training_runs": 40, "additional_runs": 20,
              "training_transitions": 800000, "evaluation_policy_instances": 42, "evaluation_scenarios": 8, "episodes_per_policy_scenario": 1024,
              "changed_configuration_fields": ["training_seeds"], "learning_code_unchanged": True, "original_common_paths_and_policy_outputs_equal": True,
              "original_evidence_unmodified": True, "fixed_extension_after_initial_analysis": True, "optional_stopping_performed": False,
              "primary_contrast": primary, "primary_interval_includes_zero": primary["ci95_lower"] <= 0 <= primary["ci95_upper"],
              "independent_primary_recomputation": "PASS", "analysis_summary_sha256": digest(HERE / "analysis-summary.json")}
    write_json(HERE / "extension-verification.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
