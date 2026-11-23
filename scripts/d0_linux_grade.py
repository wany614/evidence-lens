"""Run the tagged public CPU verifier in a disposable, uncredentialed Linux container."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

TAG = "v0.3.0-rc.1"
COMMIT = "a65b1caea258d991573bddb7a65f2bc4f9945fd7"
CANDIDATE = "7e767f053f9cca09072de1259433f65c0ee550d8c14a9d968579ad1cb9623669"
SOURCES = ("verify_stage3_public.py", "verify_vision_public.py", "vision_release.py",
           "vision_public_smoke.py", "vision_release_native.py", "vision_control_attachment.py")


def write(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf8")


def inside(source, out, budget):
    interfaces = [p for p in Path("/sys/class/net").iterdir() if p.name != "lo"]
    if not interfaces:
        raise RuntimeError("Linux network accounting is unavailable")

    def counters():
        return {key: sum(int((p / "statistics" / key).read_text()) for p in interfaces)
                for key in ("rx_bytes", "tx_bytes")}

    start, before = time.monotonic(), counters()
    record = {"schema": "el-d0-linux-resources-1", "distribution": platform.freedesktop_os_release()["PRETTY_NAME"],
              "kernel": platform.release(), "machine": platform.machine(), "python": platform.python_version(),
              "download_budget_bytes": budget, "release_version": TAG, "commit": COMMIT,
              "candidate_sha256": CANDIDATE,
              "verifier_sources": {n: hashlib.sha256((source / n).read_bytes()).hexdigest() for n in SOURCES}}
    minimum = shutil.disk_usage("/tmp").free
    peak = Path("/sys/fs/cgroup/memory.peak")
    try:
        cmd = [sys.executable, "-I", "-B", str(source / SOURCES[0]), "--commit", COMMIT,
               "--candidate-sha256", CANDIDATE, "--output", str(out / "acceptance")]
        with subprocess.Popen(cmd) as proc:
            while proc.poll() is None:
                minimum = min(minimum, shutil.disk_usage("/tmp").free)
                if counters()["rx_bytes"] - before["rx_bytes"] >= budget:
                    proc.kill()
                    raise RuntimeError("Public installation exceeded the registered download budget")
                time.sleep(0.25)
            if proc.returncode:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
    finally:
        after = counters()
        record.update(elapsed_seconds=time.monotonic() - start,
                      network_bytes={k: after[k] - before[k] for k in before},
                      tmpfs_capacity_bytes=shutil.disk_usage("/tmp").total,
                      sampled_minimum_tmpfs_free_bytes=minimum,
                      container_peak_memory_bytes=int(peak.read_text()) if peak.is_file() else None)
        write(out / "linux-resources.json", record)


def run(a):
    out = a.output.resolve()
    out.mkdir(exist_ok=True)
    if set(p.name for p in a.verifier.iterdir()) != set(SOURCES):
        raise ValueError("Supply the six public verifier source files from the registered Git tag")
    binary = shutil.which("docker")
    if not binary:
        raise RuntimeError("Docker CLI is unavailable")
    image = json.loads(subprocess.check_output([binary, "image", "inspect", a.image], text=True))[0]
    if image["Os"] != "linux" or image["Id"] != a.image:
        raise ValueError("Supply the immutable local Linux image ID")
    start = time.monotonic()
    floor = 30 * 2**30
    minimum = shutil.disk_usage(out).free
    if minimum <= floor:
        raise RuntimeError("Host volume is already at the registered physical floor")
    cmd = [binary, "run", "--rm", "--pull=never", "--read-only", "--cap-drop=ALL",
           "--security-opt=no-new-privileges", "--cpus=4", "--memory=5g",
           "--tmpfs", "/tmp:rw,exec,nosuid,size=3g,mode=1777",
           "--volume", f"{a.verifier.resolve()}:/verifier:ro",
           "--volume", f"{Path(__file__).resolve()}:/d0.py:ro",
           "--volume", f"{out}:/out:rw", a.image, "python", "-I", "-B", "/d0.py",
           "inside", "--verifier", "/verifier", "--output", "/out", "--network-budget", str(a.network_budget)]
    cid = "el-d0-" + os.urandom(8).hex()
    cmd[2:2] = ["--name", cid]
    try:
        with subprocess.Popen(cmd) as proc:
            while proc.poll() is None:
                free = shutil.disk_usage(out).free
                minimum = min(free, minimum)
                if free < floor:
                    subprocess.run([binary, "kill", cid], check=False, stdout=subprocess.DEVNULL)
                    raise RuntimeError("Stopped disposable Linux verifier at the host volume floor")
                time.sleep(0.25)
            if proc.returncode:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
    finally:
        write(out / "host.json", {"schema": "el-d0-linux-host-1", "image_id": image["Id"],
                                  "image_repo_digests": image.get("RepoDigests", []),
                                  "image_architecture": image["Architecture"], "host_machine": platform.machine(),
                                  "host_system": platform.system(), "elapsed_seconds": time.monotonic() - start,
                                  "minimum_free_bytes": minimum, "free_floor_bytes": floor})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "inside"])
    parser.add_argument("--verifier", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", help="immutable local Docker image ID, including sha256:")
    parser.add_argument("--network-budget", type=int, default=3 * 2**28)
    args = parser.parse_args()
    if args.action == "inside":
        inside(args.verifier, args.output, args.network_budget)
    else:
        run(args)
