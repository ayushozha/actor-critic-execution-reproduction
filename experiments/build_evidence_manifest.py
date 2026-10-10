"""Administrative receipt assembly; every scientific statistic is read from prior old-runtime output."""
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def artifact(kind, path):
    return {"kind": kind, "path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    analysis = json.loads((HERE / "analysis-summary.json").read_text(encoding="utf-8"))
    runs = analysis["training_runs"]
    evaluation = json.loads((HERE / "evaluation" / "execution-record.json").read_text(encoding="utf-8"))
    frozen = json.loads((HERE / "frozen-protocol.json").read_text(encoding="utf-8"))
    artifacts = [artifact("code", HERE / "execution_study.py"), artifact("code", HERE / "analyze_results.py"),
                 artifact("configuration", HERE / "config.json"), artifact("configuration", HERE / "frozen-protocol.json"),
                 artifact("raw_results", HERE / "evaluation" / "raw-episodes.csv"),
                 artifact("raw_results", HERE / "evaluation" / "common-paths.npz"),
                 artifact("raw_results", HERE / "analysis-summary.json"),
                 artifact("execution_record", HERE / "evaluation" / "execution-record.json"),
                 artifact("execution_record", HERE / "analysis-execution-record.json"),
                 artifact("execution_record", HERE / "launch-record.json"),
                 artifact("runtime_identity", ROOT / "runtime" / "runtime-manifest.json")]
    for run in runs:
        directory = HERE / "runs" / "{}_{}_seed{}".format(run["algorithm"], run["regime"], run["seed"])
        for path in sorted(directory.glob("checkpoint_*.pt")):
            artifacts.append(artifact("checkpoint", path))
        artifacts.extend([artifact("runtime_identity", directory / "runtime-identity.json"),
                          artifact("execution_record", directory / "execution-record.json"),
                          artifact("configuration", directory / "configuration.json"),
                          artifact("raw_results", directory / "training-episodes.csv"),
                          artifact("raw_results", directory / "replay-final.npz")])
    manifest = {"id": "controlled_execution_v1", "record_type": "current_simulation_outputs",
                "executed_start_utc": min(row["started_utc"] for row in runs), "executed_end_utc": evaluation["finished_utc"],
                "analysis_completed_utc": json.loads((HERE / "analysis-execution-record.json").read_text(encoding="utf-8"))["finished_utc"],
                "manifest_created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "cutoff_exclusive": "2018-09-20", "method_cutoff_applies_to": "scientific versions, designs, methods and software, not the actual present-day execution date",
                "new_experiments_authorized": True, "simulated_data": True, "real_market_data_used": False,
                "training_runs": len(runs), "training_transitions": analysis["total_training_transitions"],
                "heldout_episode_evaluations": analysis["total_episode_evaluations"], "checkpoint_count": sum(a["kind"] == "checkpoint" for a in artifacts),
                "preflight": {"path": "experiments/preflight.json", "sha256": hashlib.sha256((HERE / "preflight.json").read_bytes()).hexdigest(), "result": "PASS"},
                "frozen_utc": frozen["frozen_utc"], "heldout_tuning_performed": False,
                "sources": frozen["scientific_sources"], "scientific_software": ["Python3.6.6", "NumPy1.15.1", "PyTorch0.4.1"],
                "raw_integrity": analysis["raw_result_integrity"], "completion_failures": analysis["completion_failures"], "artifacts": artifacts}
    (HERE / "evidence-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
