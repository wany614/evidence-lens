"""Verify that platform grades descend from anonymous downloads made after publication."""
import argparse
from datetime import datetime
import os
from pathlib import Path
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require


def native(store, pin, member):
    directory = store / "runs" / pin["run_id"]
    completion = load(directory / "completion.json")
    manifest_path = directory / "manifest.json"
    require(sha(manifest_path) == completion["manifest_sha256"] == pin["manifest_sha256"], "manifest differs")
    entries = {f["path"]: f for f in load(manifest_path)["files"]}
    for name in (member, pin["member"]):
        file = directory / name
        require(file.stat().st_size == entries[name]["bytes"] and sha(file) == entries[name]["sha256"],
                "different native member " + name)
    require(entries[pin["member"]]["sha256"] == pin["sha256"], "source identity differs")
    return load(directory / member)


def time_value(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def verify(args):
    pins = load(args.config)["sources"]
    release = native(args.store, pins["publication"], pins["publication"]["member"])
    public_url = "https://github.com/wany614/evidence-lens/releases/tag/" + release["tag"]
    require(release["draft"] is False and release["url"] == public_url, "release canonical URL differs")
    published = time_value(release["published_at"])
    clocks = {}
    for role in ("mac_download", "mac_grade", "linux_grade", "windows_grade"):
        run = native(args.store, pins[role], "run.json")
        start, end = run["started_at_utc"], run["ended_at_utc"]
        require(run["execution_status"] == "succeeded" and published <= time_value(start) <= time_value(end),
                "an acceptance or download predates publication")
        clocks[role] = {"run_id": run["run_id"], "start_utc": start, "finish_utc": end}
    windows_grade = pins["windows_grade"]
    require(windows_grade["run_id"] == pins["windows_download"]["run_id"], "Windows grade and HTTP record differ")
    mac_download = pins["mac_download"]
    for role in ("mac_grade", "linux_grade"):
        cfg = native(args.store, pins[role], "config.resolved.yaml")
        require(cfg["sources"]["download"] == mac_download
                and time_value(clocks[role]["start_utc"]) >= time_value(clocks["mac_download"]["finish_utc"]),
                "grade lacks the completed public download as its parent")
    windows = native(args.store, windows_grade, "config.resolved.yaml")
    require(windows["phase"] == "public" and windows["input_release_manifest_sha256"] == release["manifest_sha256"]
            and windows["commit"] == release["code_commit"], "Windows grade reports a different phase or release")
    http = {}
    for role in ("mac_download", "windows_download"):
        doc = native(args.store, pins[role], pins[role]["member"])
        requests = doc["anonymous_requests"]
        prefix = public_url.replace("/tag/", "/download/") + "/"
        require(doc["code_commit"] == release["code_commit"] and doc["tag"] == release["tag"]
                and all(row["anonymous"] and row["http_status"] == 200 and row["url"] == prefix + row["name"]
                        for row in requests), "public request URL or status differs")
        http[role] = [row["name"] for row in requests]
    transport = load(args.transport)
    intake = native(args.store, pins["windows_intake"], "config.resolved.yaml")
    require(sha(args.transport) == intake["extra_sha256"]["return_receipt"] and transport["run"] == windows_grade["run_id"]
            and transport["acceptance_mode"] == "public" and transport["certificate_verification"] == "CERT_REQUIRED"
            and transport["hostname_verification"] is True, "different trust evidence")
    failed = native(args.store, pins["previous_failure"], pins["previous_failure"]["member"])
    require(failed["native_run_id"] == transport["retry_of_run_id"]
            and failed["native_manifest_sha256"] == transport["retry_of_manifest_sha256"], "wrong predecessor")
    save(args.out / "chronology.json", {
        "schema": "el-public-replay-chronology-1", "run_id": os.environ["EL_RUN_ID"], "tag": release["tag"],
        "release_url": public_url, "code_commit": release["code_commit"], "manifest_sha256": release["manifest_sha256"],
        "published_at": release["published_at"], "worker_utc": clocks, "anonymous_assets": http,
        "linux_source": "Mac public download mounted in a fresh network-off arm64 container",
        "windows_first_attempt": failed["native_run_id"],
        "windows_trust": {k: transport[k] for k in ("ssl_cert_file_sha256", "certificate_verification", "hostname_verification",
                                                     "additional_ca_source", "public_http_client", "system_native_transfer_client")},
        "source_receipts": pins,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("store", "config", "out", "transport"):
        parser.add_argument("--" + key, type=Path, required=True)
    verify(parser.parse_args())
