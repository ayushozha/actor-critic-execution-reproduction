"""Prespecified scientific summary; executed by the historical runtime, not modern NumPy."""
from __future__ import print_function

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# The isolated embedded runtime otherwise imports the parent experiment module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from execution_study import HERE, ROOT, digest, identity, utc, write_json

T9 = 2.2621571628540993
METRICS = ["impact_cost", "holding_risk_penalty", "price_slippage", "implementation_shortfall", "objective", "maximum_trade"]


def interval(values):
    a = np.asarray(values, dtype=float)
    assert len(a) == 10 and np.all(np.isfinite(a))
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    half = T9 * sd / math.sqrt(len(a))
    return {"mean": mean, "seed_sd": sd, "ci95_lower": mean - half, "ci95_upper": mean + half,
            "seed_values": a.tolist(), "seed_count": len(a), "ci_method": "paired training-seed t interval df=9"}


def main():
    started, clock = utc(), time.time()
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    seeds = cfg["training_seeds"]
    groups = [(algorithm, regime) for algorithm in cfg["algorithms"] for regime in cfg["regimes"]]
    scenarios = [s["name"] for s in cfg["scenarios"]]
    data, counts, ids = {}, {}, {}
    source = HERE / "evaluation" / "raw-episodes.csv"
    with source.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["scenario"], row["algorithm"], row["regime"], int(row["training_seed"]))
            if key not in data:
                data[key] = {metric: [] for metric in METRICS}
                ids[key] = []
            ids[key].append(int(row["episode"]))
            assert float(row["terminal_inventory"]) == 0.0
            for metric in METRICS:
                data[key][metric].append(float(row[metric]))
    means = {}
    for key, metrics in data.items():
        assert sorted(ids[key]) == list(range(cfg["evaluation_episodes_per_scenario"]))
        means[key] = {metric: float(np.mean(values)) for metric, values in metrics.items()}
    expected = (len(groups) * len(seeds) + 2) * len(scenarios)
    assert len(means) == expected
    saved_means = json.loads((HERE / "evaluation" / "seed-means.json").read_text(encoding="utf-8"))
    mapped = {"impact_cost": "mean_impact", "holding_risk_penalty": "mean_risk", "price_slippage": "mean_slippage",
              "implementation_shortfall": "mean_shortfall", "objective": "mean_objective", "maximum_trade": "mean_maximum_trade"}
    for row in saved_means:
        key = (row["scenario"], row["algorithm"], row["regime"], row["seed"])
        for metric, saved_field in mapped.items():
            assert abs(means[key][metric] - row[saved_field]) < 1e-12
    rows, baselines, contrasts, schedule_rows = [], [], [], []
    for scenario in scenarios:
        for baseline in ["twap", "ac_nominal"]:
            baseline_key = (scenario, baseline, "none", -1)
            baselines.append({"scenario": scenario, "method": baseline, "metrics": means[baseline_key],
                              "episodes": cfg["evaluation_episodes_per_scenario"], "trained_seed_count": 0})
        for algorithm, regime in groups:
            stats = {metric: interval([means[(scenario, algorithm, regime, seed)][metric] for seed in seeds]) for metric in METRICS}
            rows.append({"scenario": scenario, "algorithm": algorithm, "regime": regime, "metrics": stats})
            schedules = []
            for seed in seeds:
                path = HERE / "evaluation" / "{}_{}_{}_seed{}.npz".format(scenario, algorithm, regime, seed)
                with np.load(str(path)) as arrays:
                    assert arrays["trades"].shape == (cfg["evaluation_episodes_per_scenario"], cfg["steps"])
                    for array_name, metric in [("impact", "impact_cost"), ("risk", "holding_risk_penalty"),
                                              ("slippage", "price_slippage"), ("shortfall", "implementation_shortfall"), ("objective", "objective")]:
                        assert np.max(np.abs(arrays[array_name] - np.asarray(data[(scenario, algorithm, regime, seed)][metric]))) < 1e-12
                    schedules.append(arrays["trades"].mean(axis=0))
            matrix = np.asarray(schedules)
            schedule_rows.append({"scenario": scenario, "algorithm": algorithm, "regime": regime,
                                  "mean_trade": matrix.mean(axis=0).tolist(), "seed_sd_trade": matrix.std(axis=0, ddof=1).tolist(),
                                  "mean_remaining_inventory": (1.0 - np.cumsum(matrix.mean(axis=0))).tolist()})
            for baseline in ["twap", "ac_nominal"]:
                baseline_value = means[(scenario, baseline, "none", -1)]["objective"]
                diffs = [means[(scenario, algorithm, regime, seed)]["objective"] - baseline_value for seed in seeds]
                stat = interval(diffs)
                stat.update(scenario=scenario, treatment="{}_{}".format(algorithm, regime), comparator=baseline,
                            metric="objective", relative_percent=100 * stat["mean"] / baseline_value,
                            relative_percent_denominator=baseline_value)
                contrasts.append(stat)
    nonnominal = [name for name in scenarios if name != "nominal"]
    primary = []
    for algorithm in cfg["algorithms"]:
        nominal_seed = [float(np.mean([means[(s, algorithm, "nominal", seed)]["objective"] for s in nonnominal])) for seed in seeds]
        randomized_seed = [float(np.mean([means[(s, algorithm, "randomized", seed)]["objective"] for s in nonnominal])) for seed in seeds]
        stat = interval(np.asarray(randomized_seed) - np.asarray(nominal_seed))
        denom = float(np.mean(nominal_seed))
        stat.update(comparison="{}_randomized_minus_nominal".format(algorithm), primary=algorithm == "td3",
                    scenario_weighting="equal over seven prespecified nonnominal scenarios",
                    comparator_seed_values=nominal_seed, treatment_seed_values=randomized_seed,
                    relative_percent=100 * stat["mean"] / denom, relative_percent_denominator=denom,
                    negative_difference_means="lower risk-adjusted simulation objective")
        primary.append(stat)
    cross_algorithm = []
    for regime in cfg["regimes"]:
        diffs = [np.mean([means[(s, "td3", regime, seed)]["objective"] - means[(s, "ddpg", regime, seed)]["objective"] for s in nonnominal]) for seed in seeds]
        stat = interval(diffs)
        stat.update(comparison="td3_minus_ddpg_{}".format(regime), scenario_weighting="equal seven nonnominal scenarios")
        cross_algorithm.append(stat)
    training_records, curves = [], []
    for algorithm, regime in groups:
        per_seed = []
        for seed in seeds:
            run = HERE / "runs" / "{}_{}_seed{}".format(algorithm, regime, seed)
            record = json.loads((run / "execution-record.json").read_text(encoding="utf-8"))
            assert record["result"] == "completed" and record["transitions"] == cfg["training_transitions"]
            assert record["critic_updates"] == cfg["training_transitions"] - cfg["warmup_transitions"]
            assert record["actor_updates"] == record["critic_updates"] // (cfg["td3_policy_delay"] if algorithm == "td3" else 1)
            assert record["code_sha256"] == digest(HERE / "execution_study.py")
            for artifact in record["artifacts"]:
                assert digest(ROOT / artifact["path"]) == artifact["sha256"]
            with (run / "training-episodes.csv").open(newline="") as stream:
                episode_rows = list(csv.DictReader(stream))
            assert len(episode_rows) == cfg["training_transitions"] // cfg["steps"]
            training_records.append({"algorithm": algorithm, "regime": regime, "seed": seed,
                                     "elapsed_seconds": record["elapsed_seconds"], "started_utc": record["started_utc"],
                                     "finished_utc": record["finished_utc"], "transitions": record["transitions"],
                                     "critic_updates": record["critic_updates"], "actor_updates": record["actor_updates"]})
            values = np.asarray([float(row["objective"]) for row in episode_rows])
            per_seed.append(values.reshape(-1, 50).mean(axis=1))
        matrix = np.asarray(per_seed)
        curves.append({"algorithm": algorithm, "regime": regime, "transitions_at_block_end": (np.arange(1, matrix.shape[1] + 1) * 50 * cfg["steps"]).tolist(),
                       "mean_training_objective": matrix.mean(axis=0).tolist(), "seed_sd_training_objective": matrix.std(axis=0, ddof=1).tolist(),
                       "each_block_episodes": 50, "checkpoint_selection_performed": False})
    report = {"created_utc": utc(), "runtime": identity(), "method": "fixed final-checkpoint analysis in historical scientific runtime",
              "training_seeds": seeds, "total_training_transitions": sum(row["transitions"] for row in training_records),
              "total_episode_evaluations": sum(len(v["objective"]) for v in data.values()), "raw_result_integrity": "PASS",
              "completion_failures": 0, "primary_and_regime_comparisons": primary, "cross_algorithm_secondary": cross_algorithm,
              "scenario_group_results": rows, "baseline_results": baselines, "scenario_baseline_contrasts": contrasts,
              "schedule_plot_series": schedule_rows, "learning_curve_plot_series": curves, "training_runs": training_records,
              "ci_scope": "Seed-level intervals quantify ten independent trained-policy seeds conditional on the fixed common Monte Carlo evaluation paths. They do not include finite test-path uncertainty or establish live-market performance.",
              "multiple_comparison_policy": "The original primary TD3 contrast retained for a fixed ten-seed extension after the initial five-seed analysis; remaining scenario/algorithm comparisons descriptive, no multiplicity-adjusted significance claim.",
              "design_limitations": ["Small network and 20k interactions, no exact published benchmark replication", "Uncalibrated simulation parameter choices",
                                     "Temporary impact only; no permanent/transient impact, queues, spreads, fees, or empirical equity data",
                                     "Holding-risk term covers independent price shocks rather than total unconditional shortfall variance",
                                     "One fixed common set of 1024 evaluation paths; ten training seeds", "No hindsight or nonlinear exact-optimum oracle"]}
    write_json(HERE / "analysis-summary.json", report)
    analysis_record = {"started_utc": started, "finished_utc": utc(), "elapsed_seconds": time.time() - clock, "result": "completed",
                       "runtime": identity(), "code_sha256": digest(__file__), "configuration_sha256": digest(HERE / "config.json"),
                       "inputs": [{"path": str(source.relative_to(ROOT)), "sha256": digest(source)},
                                  {"path": "experiments/evaluation/seed-means.json", "sha256": digest(HERE / "evaluation" / "seed-means.json")}],
                       "outputs": [{"path": "experiments/analysis-summary.json", "sha256": digest(HERE / "analysis-summary.json")}],
                       "new_training_or_tuning_performed": False}
    write_json(HERE / "analysis-execution-record.json", analysis_record)
    print(json.dumps(primary, indent=2), flush=True)


if __name__ == "__main__":
    main()
