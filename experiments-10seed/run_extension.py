"""Administrative fixed-extension launcher; scientific children use the old runtime."""
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
    path = HERE / "launch-logs" / "{}_{}_seed{}.txt".format(*job)
    with path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    return {"job": job, "command": command, "exit_code": result.returncode, "log": str(path.relative_to(ROOT))}


def main():
    plan = json.loads((HERE / "extension-plan.json").read_text(encoding="utf-8"))
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    subprocess.run([str(PYTHON), str(HERE / "execution_study.py"), "freeze"], check=True, stdout=subprocess.DEVNULL)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    clock = time.time()
    (HERE / "launch-logs").mkdir(exist_ok=False)
    jobs = [(algorithm, regime, seed) for seed in plan["additional_training_seeds"] for algorithm in cfg["algorithms"] for regime in cfg["regimes"]]
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print("Completed {} exit={} elapsed={:.1f}s ({}/{})".format(record["job"], record["exit_code"], time.time() - clock, len(records), len(jobs)), flush=True)
    result = "failed_training"
    if all(record["exit_code"] == 0 for record in records):
        result = "completed"
        for script, args in [("execution_study.py", ["evaluate"]), ("analyze_results.py", []), ("verify_extension.py", [])]:
            command = [str(PYTHON), str(HERE / script)] + args
            child = subprocess.run(command)
            records.append({"command": command, "exit_code": child.returncode})
            if child.returncode:
                result = "failed_" + script
                break
    report = {"started_utc": started, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "elapsed_seconds": time.time() - clock,
              "result": result, "parallel_child_limit": 4, "administrative_python": __import__("sys").version, "scientific_python": str(PYTHON),
              "records": records, "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE / "launch-record.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if result != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
