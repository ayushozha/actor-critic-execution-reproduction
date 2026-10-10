"""Administrative launch orchestration. Scientific children use the isolated 2018 runtime."""
import concurrent.futures
import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PYTHON = ROOT / "runtime" / "python366" / "python.exe"


def run_job(job):
    algorithm, regime, seed = job
    command = [str(PYTHON), str(HERE / "execution_study.py"), "train", "--algorithm", algorithm, "--regime", regime, "--seed", str(seed)]
    log_path = HERE / "launch-logs" / "{}_{}_seed{}.txt".format(*job)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    return {"job": job, "command": command, "exit_code": result.returncode, "log": str(log_path.relative_to(ROOT))}


def main():
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    jobs = [(algorithm, regime, seed) for seed in cfg["training_seeds"] for algorithm in cfg["algorithms"] for regime in cfg["regimes"]]
    (HERE / "launch-logs").mkdir(exist_ok=True)
    records = []
    clock = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print("Completed {} exit={} elapsed={:.1f}s ({}/{})".format(record["job"], record["exit_code"], time.time() - clock, len(records), len(jobs)), flush=True)
    if not all(record["exit_code"] == 0 for record in records):
        result = "failed_training"
    else:
        command = [str(PYTHON), str(HERE / "execution_study.py"), "evaluate"]
        evaluation = subprocess.run(command)
        records.append({"command": command, "exit_code": evaluation.returncode})
        result = "completed" if evaluation.returncode == 0 else "failed_evaluation"
    report = {"started_utc": started, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "result": result, "elapsed_seconds": time.time() - clock, "parallel_child_limit": 4,
              "administrative_python": __import__("sys").version,
              "scientific_python": str(PYTHON), "records": records,
              "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE / "launch-record.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if result != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
