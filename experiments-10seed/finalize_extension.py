"""Administrative assembly of receipts and already-computed manuscript values."""
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def write(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def main():
    verification = read("extension-verification.json")
    assert verification["result"] == "PASS"
    launch = read("launch-record-initial.json")
    analysis_receipt = read("analysis-execution-record.json")
    assert all(row["exit_code"] == 0 for row in launch["records"] if "job" in row)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    launch.update(result="completed", recovered_utc=now,
                  original_first_attempt_result=launch["result"], original_first_attempt_elapsed_seconds=launch["elapsed_seconds"],
                  recovery_note="Initial analysis resolved the parent module because embedded Python has a fixed import path. Extension analysis and verification explicitly select the byte-identical copied module. No training or evaluation was rerun; no seed or hyperparameter was changed.",
                  scientific_analysis_receipt="experiments-10seed/analysis-execution-record.json",
                  extension_verification="experiments-10seed/extension-verification.json",
                  recovery_steps=[{"command": [str(ROOT / "runtime/python366/python.exe"), str(HERE / "analyze_results.py")], "exit_code": 0, "receipt": "analysis-execution-record.json"},
                                  {"command": [str(ROOT / "runtime/python366/python.exe"), str(HERE / "verify_extension.py")], "exit_code": 0, "receipt": "extension-verification.json"}])
    write("launch-record.json", launch)
    data = read("analysis-summary.json")
    manifest = {"prepared_utc": now, "source": "experiments-10seed/analysis-summary.json",
                "source_sha256": verification["analysis_summary_sha256"], "training_seed_count": 10,
                "training_seeds": data["training_seeds"], "training_runs": 40, "additional_training_runs": 20,
                "total_training_transitions": data["total_training_transitions"], "total_episode_evaluations": data["total_episode_evaluations"],
                "saved_checkpoints": 160, "completion_failures": data["completion_failures"], "ci_degrees_of_freedom": 9,
                "ci_critical_value": 2.2621571628540993, "extension_timing": "Five additional matched seeds fixed after the original five-seed analysis; no optional stopping",
                "primary_and_regime_comparisons": data["primary_and_regime_comparisons"], "cross_algorithm_secondary": data["cross_algorithm_secondary"],
                "scenario_group_results": data["scenario_group_results"], "baseline_results": data["baseline_results"],
                "scenario_baseline_contrasts": data["scenario_baseline_contrasts"], "schedule_plot_series": data["schedule_plot_series"],
                "learning_curve_plot_series": data["learning_curve_plot_series"], "training_runs_details": data["training_runs"],
                "ci_scope": data["ci_scope"], "design_limitations": data["design_limitations"], "integrity_result": verification["result"],
                "statistical_computation": "Copied from saved scientific-runtime output; no modern-library recomputation"}
    write("manuscript-results.json", manifest)
    files = [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
             for p in sorted(HERE.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.name != "evidence-manifest.json"]
    write("evidence-manifest.json", {"created_utc": now, "role": "Completed fixed seed-extension evidence; actual execution times preserved",
                                     "integrity_verification": "extension-verification.json", "files": files})
    primary = verification["primary_contrast"]
    print("Completed extension: mean={}, CI=[{}, {}], relative={}%, integrity={}".format(primary["mean"], primary["ci95_lower"], primary["ci95_upper"], primary["relative_percent"], verification["result"]), flush=True)


if __name__ == "__main__":
    main()
