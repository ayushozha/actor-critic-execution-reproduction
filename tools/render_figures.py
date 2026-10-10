"""Present-day administrative rendering of saved historical-runtime scientific results.

No training, scientific resampling, confidence calculation, or scientific statistic
is performed here. Display geometry and labels alone are constructed from saved
analysis values. The original experiment and analysis artifacts remain unchanged.
"""
import datetime
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path('C:\\Users\\ayush\\Desktop\\Research Papers\\Deep Actorâ€“Critic Methods for Optimal Trade Execution Under Uncertain Market Impact')
OUT = ROOT / "figures" / "revision-10seed"
SOURCE = ROOT / "experiments-10seed" / "analysis-summary.json"
GROUPS = [("ddpg", "nominal"), ("ddpg", "randomized"), ("td3", "nominal"), ("td3", "randomized")]
LABELS = {("ddpg", "nominal"): "DDPG nominal", ("ddpg", "randomized"): "DDPG randomized",
          ("td3", "nominal"): "TD3 nominal", ("td3", "randomized"): "TD3 randomized"}
COLORS = ["#4477AA", "#66CCEE", "#228833", "#AA3377"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(figure, name):
    figure.savefig(OUT / (name + ".pdf"), bbox_inches="tight", metadata={"Creator": "Administrative Matplotlib rendering of saved scientific values"})
    figure.savefig(OUT / (name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    OUT.mkdir(exist_ok=True)
    source_hash = sha(SOURCE)
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16, "axes.labelsize": 17,
                         "axes.titlesize": 17, "legend.fontsize": 15, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    rows = {(row["scenario"], row["algorithm"], row["regime"]): row for row in data["scenario_group_results"]}
    scenarios = [row["scenario"] for row in data["scenario_group_results"] if (row["algorithm"], row["regime"]) == GROUPS[0]]
    baseline = {(row["scenario"], row["method"]): row for row in data["baseline_results"]}
    figure, axis = plt.subplots(figsize=(11.2, 5.4))
    width = .125
    for index, group in enumerate(GROUPS):
        stats = [rows[(scenario,) + group]["metrics"]["objective"] for scenario in scenarios]
        means = [stat["mean"] for stat in stats]
        # Endpoint subtraction is plotting geometry; saved confidence limits are unchanged.
        lower = [stat["mean"] - stat["ci95_lower"] for stat in stats]
        upper = [stat["ci95_upper"] - stat["mean"] for stat in stats]
        locations = [k + (index - 2.5) * width for k in range(len(scenarios))]
        axis.bar(locations, means, width=width, color=COLORS[index], label=LABELS[group],
                 yerr=[lower, upper], capsize=2, error_kw={"elinewidth": .8, "ecolor": "#333333"})
    for index, (method, color, label, hatch) in enumerate([("twap", "#BBBBBB", "TWAP", "//"), ("ac_nominal", "#555555", "Fixed quadratic schedule", "")], 4):
        locations = [k + (index - 2.5) * width for k in range(len(scenarios))]
        means = [baseline[(scenario, method)]["metrics"]["objective"] for scenario in scenarios]
        axis.bar(locations, means, width=width, color=color, label=label, hatch=hatch)
    axis.set_xticks(list(range(len(scenarios))))
    axis.set_xticklabels(["Nominal", "High\nimpact", "Low\nimpact", "Concave\nimpact", "Convex\nimpact", "Thin\nliquidity", "Joint\nshift", "High\nliquidity"])
    axis.set_ylim(0, 1.1)
    axis.set_yticks([0, .25, .5, .75, 1])
    axis.set_ylabel("Risk-adjusted cost (normalized)")
    axis.set_xlabel("Prespecified evaluation scenario")
    axis.grid(axis="y", color="#DDDDDD", linewidth=.6)
    axis.set_axisbelow(True)
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.18), frameon=False)
    figure.tight_layout()
    save(figure, "scenario_objective")

    schedules = {(row["scenario"], row["algorithm"], row["regime"]): row for row in data["schedule_plot_series"]}
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "axes.titlesize": 15, "legend.fontsize": 13})
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharex=True, sharey=True)
    for axis, scenario, title in zip(axes, ["nominal", "joint_shift"], ["Nominal setting", "Joint impact and liquidity shift"]):
        for group, color in zip(GROUPS, COLORS):
            remaining = [1.0] + schedules[(scenario,) + group]["mean_remaining_inventory"]
            time_positions = [k / float(len(remaining) - 1) for k in range(len(remaining))]
            axis.plot(time_positions, remaining, color=color, linewidth=1.7, label=LABELS[group],
                      linestyle="--" if group[1] == "nominal" else "-", drawstyle="steps-post")
        axis.set_title(title)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xticks([0, .2, .4, .6, .8, 1])
        axis.set_yticks([0, .2, .4, .6, .8, 1])
        axis.set_xlabel("Elapsed horizon fraction")
        axis.grid(color="#DDDDDD", linewidth=.6)
    axes[0].set_ylabel("Mean remaining inventory")
    axes[1].legend(frameon=False, loc="upper right")
    figure.tight_layout()
    save(figure, "inventory_schedules")

    curves = {(row["algorithm"], row["regime"]): row for row in data["learning_curve_plot_series"]}
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11, "legend.fontsize": 9})
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    for group, color in zip(GROUPS, COLORS):
        curve = curves[group]
        axis.plot(curve["transitions_at_block_end"], curve["mean_training_objective"], color=color,
                  linewidth=1.7, label=LABELS[group], linestyle="--" if group[1] == "nominal" else "-")
    axis.set_xlim(0, 20000)
    axis.set_ylim(0, 1)
    axis.set_xticks([0, 5000, 10000, 15000, 20000])
    axis.set_yticks([0, .25, .5, .75, 1])
    axis.set_xlabel("Training environment transitions")
    axis.set_ylabel("Mean training risk-adjusted cost")
    axis.grid(color="#DDDDDD", linewidth=.6)
    axis.legend(ncol=2, frameon=False, loc="upper right")
    figure.tight_layout()
    save(figure, "training_objective")
    assert sha(SOURCE) == source_hash
    manifest = {"started_utc": started, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "role": "present-day administrative figure rendering only", "scientific_statistics_recomputed": False,
                "new_training_or_trials": False, "input": {"path": str(SOURCE.relative_to(ROOT)), "sha256": source_hash},
                "renderer": {"python": sys.version, "matplotlib": matplotlib.__version__,
                             "code_path": "build/reviewer-revision/render_figures.py", "code_sha256": sha(Path(__file__))},
                "figures": [{"name": "scenario_objective", "source_fields": ["scenario_group_results.*.metrics.objective", "baseline_results.*.metrics.objective"],
                             "description": "Eight scenarios, four learned policy groups and two baselines. Learned bars carry saved seed-level 95% intervals."},
                            {"name": "inventory_schedules", "source_fields": ["schedule_plot_series.*.mean_remaining_inventory"],
                             "description": "Four learned policy groups only; saved nominal and joint-shift mean inventory curves."},
                            {"name": "training_objective", "source_fields": ["learning_curve_plot_series.*.transitions_at_block_end", "learning_curve_plot_series.*.mean_training_objective"],
                             "description": "Four learned groups, saved 50-episode block means. No checkpoint selection."}],
                "outputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path), "bytes": path.stat().st_size}
                            for path in sorted(OUT.iterdir()) if path.suffix in [".pdf", ".png"]]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"input_sha256": source_hash, "outputs": manifest["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
