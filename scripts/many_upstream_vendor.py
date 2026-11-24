"""Seal the immutable official gsplat example and issue a data-free Windows environment inventory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request
import zipfile

from dual_records import selected, stage
from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens_bundle.common import require


COMMIT = "937e29912570c372bed6747a5c9bf85fed877bae"
BASELINE = "6e8e837afa0818cdf8c100b494672b46c10bdbd22e522f388983543f0a5b0f23"
UPSTREAM = "https://raw.githubusercontent.com/nerfstudio-project/gsplat/"
FILES = ("LICENSE", "examples/simple_trainer.py", "examples/utils.py", "examples/datasets/colmap.py",
         "examples/datasets/normalize.py", "examples/requirements.txt", "gsplat/strategy/mcmc.py",
         "gsplat/strategy/ops.py", "gsplat/rendering.py")


def build(s):
    pins = {"baseline": selected(s, "a-baseline"), "anonymous": selected(s, "a-anonymous-readback")}
    require(pins["baseline"]["sha256"] == BASELINE, "wrong A baseline")
    cmd = ["{python}", "-B", "{source}/scripts/many_upstream_vendor.py", "worker", "--config", "{config}", "--out", "{out}"]
    return stage(s, "b-upstream-vendor", "archive_verify", pins, {"repo": "nerfstudio-project/gsplat",
                 "commit": COMMIT, "files": FILES, "scope": "official program, license and runtime inventory only"},
                 cmd, "payload/vendor.json", 900_000)


def worker(cfg, out):
    require(cfg["commit"] == COMMIT and cfg["sources"]["baseline"]["sha256"] == BASELINE, "wrong source revision or science baseline")
    native = out / "official"
    entries = {}
    for member in cfg["files"]:
        url = UPSTREAM + COMMIT + "/" + member
        request = urllib.request.Request(url, headers={"User-Agent": "EvidenceLens-scientific-source-archive"})
        with urllib.request.urlopen(request, timeout=35) as response:
            raw = response.read()
        p = native / member
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        entries[member] = {"path": "official/" + member, "sha256": sha(p), "bytes": len(raw), "url": url}
    routine = (Path(__file__).parent / "many_upstream_inventory.py").read_bytes()
    source = (native/"gsplat/strategy/mcmc.py").read_bytes()
    run_id = os.environ["EL_RUN_ID"]
    ticket = {"schema": "el-many-gsplat-inventory-ticket-1", "run_id": run_id, "commit": COMMIT,
              "allowed_datasets": [], "mcmc_canonical_lf_sha256": hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest(),
              "worker_sha256": hashlib.sha256(routine).hexdigest()}
    packet = out / "inventory-gpu.zip"
    with zipfile.ZipFile(packet, "x", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("ticket.json", encoded(ticket))
        z.writestr("run_gpu.py", routine)
    save(out / "vendor.json", {"schema": "el-many-official-gsplat-1", "run_id": run_id, "baseline_sha256": BASELINE,
         "repository": "https://github.com/nerfstudio-project/gsplat", "tag": "v1.5.3", "commit": COMMIT,
         "files": entries, "license": "Apache-2.0", "inventory": {"path": packet.name, "bytes": packet.stat().st_size,
         "sha256": sha(packet)}, "fits": 0, "photographs": 0})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("record", "worker", "publish"))
    for k in ("store", "config", "out", "exchange"):
        p.add_argument("--" + k, type=Path)
    a = p.parse_args()
    if a.action == "worker":
        worker(load(a.config), a.out)
    else:
        pin = build(a.store.resolve())
        if a.action == "publish":
            from dual_records import resolve
            from make_handoff import publish
            source = resolve((a.store, a.store/"history"), pin)
            receipt = load(source)
            name = receipt["inventory"]["path"]
            for n, raw in ((name, (source.parent/name).read_bytes()), ("dispatch.json", source.read_bytes()), ("pin.json", encoded(pin))):
                publish(a.exchange/n, raw)
        print(json.dumps(pin))
