"""Transfer just the three preregistered opaque ETH3D archives and account for all response bodies."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import urllib.request


def download(registration, scene, out, ledger):
    spec = json.loads(registration.read_bytes())["external"]
    (row,) = [r for r in spec["roster"] if r["id"] == scene]
    cap = spec["download_cap_bytes"]
    url = row["url"]
    basename = scene + "_dslr_undistorted.7z"
    if url != "https://www.eth3d.net/data/" + basename or not os.environ.get("EL_RUN_ID"):
        raise ValueError("need a preregistered official opaque source and native recorder")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=True, parents=True)
    started, cpu = time.perf_counter(), time.process_time()
    hasher = hashlib.sha256()
    actual = 0
    chunk = 4 << 20
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as head:
        url_final = head.geturl()
        if url_final != url or int(head.headers["Content-Length"]) <= 0:
            raise ValueError("different archive endpoint or missing exact transport length")
        expected = int(head.headers["Content-Length"])
    with ledger.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lock.seek(0)
        prior = [json.loads(line) for line in lock if line.strip()]
        reserved = sum(x["reserved_bytes"] for x in prior)
        if reserved + expected > cap:
            raise ValueError("aggregate 5 GB external archive budget exhausted before GET")
        lock.write(
            json.dumps({"run_id": os.environ["EL_RUN_ID"], "scene": scene, "reserved_bytes": expected}) + "\n"
        )
        lock.flush()
        os.fsync(lock.fileno())
    with urllib.request.urlopen(url, timeout=90) as reply, (out / basename).open("xb") as dest:
        if reply.geturl() != url_final or int(reply.headers["Content-Length"]) != expected:
            raise ValueError("HEAD/GET archive identity differs")
        while actual < expected:
            data = reply.read(min(chunk, expected - actual))
            if not data:
                raise ValueError("truncated download")
            hasher.update(data)
            dest.write(data)
            actual += len(data)
    journal = {
        "schema": "el-many-opaque-acquisition-1",
        "run_id": os.environ["EL_RUN_ID"],
        "scene_id": scene,
        "split": row["split"],
        "url": url,
        "catalog": spec["urls"]["catalog"],
        "rights": spec["license"],
        "file": basename,
        "sha256": hasher.hexdigest(),
        "bytes": actual,
        "http_head_requests": 1,
        "http_get_requests": 1,
        "prior_reserved_body_bytes": reserved,
        "aggregate_download_cap": cap,
        "opaque_original_photo_archive": True,
        "photograph_decodes": 0,
        "individual_H_photo_opens": 0,
        "geometric_truth_reads": 0,
        "wall_s": time.perf_counter() - started,
        "cpu_s": time.process_time() - cpu,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (out / "download.json").write_text(json.dumps(journal, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("registration", "out", "ledger"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--scene", required=True)
    a = p.parse_args()
    download(a.registration, a.scene, a.out, a.ledger)
