"""Administrative downloader: isolate historical scientific binaries; no pip or system mutation."""
import datetime
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime"
DOWNLOADS = OUT / "downloads"
BIN = OUT / "python366"
PACKAGES = BIN / "packages"


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save_url(url, target):
    if not target.exists():
        with requests.get(url, timeout=(30, 120), stream=True) as response:
            response.raise_for_status()
            with target.open("wb") as stream:
                for block in response.iter_content(1024 * 1024):
                    stream.write(block)
    raw = target.read_bytes()
    return {"url": url, "path": str(target.relative_to(ROOT)), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), "retrieved_or_checked_utc": stamp()}


def main():
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    PACKAGES.mkdir(parents=True, exist_ok=True)
    records = []
    python_page = save_url("https://www.python.org/downloads/release/python-366/", DOWNLOADS / "python-release.html")
    records.append(python_page)
    python_zip = DOWNLOADS / "python-3.6.6-embed-amd64.zip"
    record = save_url("https://www.python.org/ftp/python/3.6.6/python-3.6.6-embed-amd64.zip", python_zip)
    assert hashlib.md5(python_zip.read_bytes()).hexdigest() == "7148ec14edfdc13f42e06a14d617c921", "Python official checksum mismatch"
    record.update(name="Python", version="3.6.6", public_date="2018-06-27",
                  date_evidence_url=python_page["url"], expected_md5="7148ec14edfdc13f42e06a14d617c921")
    records.append(record)
    with zipfile.ZipFile(str(python_zip)) as archive:
        archive.extractall(str(BIN))
    for package, version in [("numpy", "1.15.1"), ("torch", "0.4.1")]:
        meta_file = DOWNLOADS / (package + "-pypi.json")
        if package == "numpy":
            meta_record = save_url("https://pypi.org/pypi/{}/{}/json".format(package, version), meta_file)
            records.append(meta_record)
            metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        else:
            metadata = {"urls": []}
        matches = [item for item in metadata["urls"] if "cp36" in item["filename"] and item["filename"].endswith("win_amd64.whl")]
        if package == "torch" and not matches:
            release_file = DOWNLOADS / "torch-release.json"
            release_record = save_url("https://api.github.com/repos/pytorch/pytorch/releases/tags/v0.4.1", release_file)
            records.append(release_record)
            release = json.loads(release_file.read_text(encoding="utf-8"))
            records.append(save_url("https://pytorch.org/get-started/previous-versions/", DOWNLOADS / "torch-previous-versions.html"))
            matches = [{"filename": "torch-0.4.1-cp36-cp36m-win_amd64.whl",
                        "url": "https://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-win_amd64.whl",
                        "upload_time_iso_8601": release["published_at"], "digests": {}}]
        if len(matches) != 1:
            raise RuntimeError("Expected one Windows Python3.6 wheel: {} {}".format(package, matches))
        item = matches[0]
        wheel = DOWNLOADS / item["filename"]
        record = save_url(item["url"], wheel)
        expected = item["digests"].get("sha256")
        if expected:
            assert record["sha256"] == expected, "PyPI wheel checksum mismatch"
        record.update(name=package, version=version, public_date=item["upload_time_iso_8601"][:10],
                      date_evidence_url=(release_record["url"] if package == "torch" and not expected else meta_record["url"]), expected_sha256=expected)
        records.append(record)
        with zipfile.ZipFile(str(wheel)) as archive:
            archive.extractall(str(PACKAGES))
        print("Downloaded and extracted {} {}".format(package, version), flush=True)
    (BIN / "python36._pth").write_text("python36.zip\n.\npackages\n../../experiments\n", encoding="ascii")
    command = [str(BIN / "python.exe"), "-c", "import sys,numpy,torch; print(sys.version); print(numpy.__version__); print(torch.__version__); print(torch.tensor([1.,2.]).sum().item())"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    manifest = {"created_utc": stamp(), "role": "present-day administrative acquisition of pre-cutoff scientific runtime", "records": records,
                "smoke_command": command, "smoke_exit_code": result.returncode, "smoke_stdout": result.stdout, "smoke_stderr": result.stderr,
                "system_packages_modified": False, "scientific_dependencies": ["Python 3.6.6", "NumPy 1.15.1", "PyTorch 0.4.1"]}
    (OUT / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
