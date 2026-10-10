"""Administrative preparation; never changes the original scientific package."""
import datetime
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARENT = ROOT / "experiments"
ADDED = [89, 107, 127, 149, 173]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (HERE / "extension-plan.json").exists():
        raise RuntimeError("Extension already prepared; refusing to overwrite its plan")
    cfg = json.loads((PARENT / "config.json").read_text(encoding="utf-8"))
    original_seeds = list(cfg["training_seeds"])
    cfg["training_seeds"] = original_seeds + ADDED
    shutil.copy2(PARENT / "execution_study.py", HERE / "execution_study.py")
    analysis = (PARENT / "analyze_results.py").read_text(encoding="utf-8")
    analysis = analysis.replace("import math\n", "import math\nimport sys\n")
    analysis = analysis.replace("from execution_study import HERE", "# Resolve the copied module despite the embedded runtime's fixed import path.\nsys.path.insert(0, str(Path(__file__).resolve().parent))\nfrom execution_study import HERE")
    analysis = analysis.replace("T4 = 2.7764451051977987", "T9 = 2.2621571628540993")
    analysis = analysis.replace("assert len(a) == 5", "assert len(a) == 10")
    analysis = analysis.replace("half = T4 *", "half = T9 *")
    analysis = analysis.replace("interval df=4", "interval df=9")
    analysis = analysis.replace("five independent trained-policy seeds", "ten independent trained-policy seeds")
    analysis = analysis.replace("1024 evaluation paths; five training seeds", "1024 evaluation paths; ten training seeds")
    analysis = analysis.replace("One prespecified primary TD3 contrast", "The original primary TD3 contrast retained for a fixed ten-seed extension after the initial five-seed analysis")
    (HERE / "analyze_results.py").write_text(analysis, encoding="utf-8")
    (HERE / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    parent_inputs = [PARENT / name for name in ["config.json", "PROTOCOL.md", "frozen-protocol.json", "execution_study.py", "analyze_results.py", "analysis-summary.json", "analysis-execution-record.json", "evaluation/common-paths.npz", "evaluation/raw-episodes.csv", "evaluation/seed-means.json", "evaluation/execution-record.json"]]
    parent_inputs += sorted((PARENT / "runs").rglob("*"))
    parent_inputs = [p for p in parent_inputs if p.is_file()]
    records = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in parent_inputs]
    shutil.copytree(PARENT / "runs", HERE / "runs")
    copied = []
    for source in sorted((PARENT / "runs").rglob("*")):
        if source.is_file():
            target = HERE / "runs" / source.relative_to(PARENT / "runs")
            assert digest(source) == digest(target)
            copied.append({"original": str(source.relative_to(ROOT)), "copy": str(target.relative_to(ROOT)), "sha256": digest(target)})
    plan = {"extension_id": "execution-impact-2018-v1-fixed-10seed-extension", "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "original_training_seeds": original_seeds, "additional_training_seeds": ADDED, "combined_training_seeds": cfg["training_seeds"],
            "decision_timing": "Fixed extension declared after inspecting the original five-seed results; not an original ten-seed preregistration",
            "stopping_rule": "Run all five additional seeds in all four groups; include every completed run; no optional stopping or outcome-based seed selection",
            "changed_configuration_fields": ["training_seeds"], "scientific_learning_code_unchanged": digest(PARENT / "execution_study.py") == digest(HERE / "execution_study.py"),
            "evaluation_design_unchanged": True, "expected_new_training_runs": 20, "expected_total_training_runs": 40,
            "parent_inputs": records, "preserved_run_copies": copied}
    (HERE / "extension-plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print("Prepared fixed extension, copied and verified {} parent run files".format(len(copied)), flush=True)


if __name__ == "__main__":
    main()
