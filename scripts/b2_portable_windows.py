"""Admit a Windows portable-package replay, verify its historical inputs, and compare the returned arrays."""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from b2_reconstruct_history import original, verify_external
from b2_records import stage
from evidence_lens._bundle import RUN_ID, load, save, sha
from evidence_lens._verify import require, verify_bundle, verify_dependencies
from evidence_lens.formal_inputs import verified_file
from t0_reconstruct_history import missing_parents
from t0_reconstruct_intake import receive_verified_zip
from t0_records import bind, resolve


NATIVE_STAGES = (
    "01-create-venv", "02-install", "03-pip-check", "04-pip-list", "05-environment",
    "06-validate", "07-score", "08-report", "09-compare",
)


def picture_bytes(html):
    images = []
    for raw in re.findall(rb"data:image/png;base64,([A-Za-z0-9+/=]+)", html):
        with Image.open(io.BytesIO(base64.b64decode(raw, validate=True))) as pic:
            require(pic.mode == "RGB" and not pic.info, "expected the original untagged opaque PNG")
            images.append((pic.size, hashlib.sha256(pic.tobytes()).hexdigest()))
    return images


def readback(roots, pins, receipt, out):
    windows = pins["windows"]
    w = resolve(roots, windows)
    claimed = load(w)
    win = w.parent
    info = load(win / "release-verification.json")
    scores = load(win / "scores/scores.json")
    opened = resolve(roots, pins["portable"])
    require(claimed == receipt["comparison"] and info["mac_producer_run"] == pins["portable"]["run_id"]
            and info["zip"] == sha(opened) == pins["portable"]["sha256"]
            and info["zip_bytes"] == opened.stat().st_size
            and info["bundle_sha256"] == scores["bundle_sha256"] == claimed["bundle_sha256"]
            and info["total_members"] == 50 and scores["calibration_profile"] is None,
            "Windows replay differs from the portable release")
    rows = []
    with zipfile.ZipFile(opened) as package:
        manifest = json.loads(package.read("manifest.json"))
        require(len(package.namelist()) == 50 and set(package.namelist()) == {*manifest, "manifest.json"},
                "incomplete portable example")
        wheel, = (name for name in manifest if name.endswith(".whl"))
        require(manifest["prepared/bundle.json"]["sha256"] == info["bundle_sha256"]
                and manifest[wheel]["sha256"] == info["wheel_sha256"]
                and hashlib.sha256(package.read("manifest.json")).hexdigest() == info["manifest_sha256"],
                "Windows measured different inner inputs")
        for name, entry in manifest.items():
            raw = package.read(name)
            require(len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"],
                    "changed portable member " + name)
        for spec in scores["views"]:
            v = spec["id"]
            native = package.read("expected/" + v + ".npz")
            path = verified_file(win / "scores", spec["arrays"]["path"], spec["arrays"])
            with np.load(io.BytesIO(native), allow_pickle=False) as old, np.load(path, allow_pickle=False) as new:
                require(set(old.files) == set(new.files) == {"risk", "valid", "state", "unknown_reason"},
                        "unexpected deployment arrays")
                delta = {}
                for field in old.files:
                    x,y = old[field],new[field]
                    require(x.shape == y.shape == (256,256) and x.dtype == y.dtype, "type or shape mismatch")
                    if field == "risk":
                        require(x.dtype == np.float32, "expected FP32 risk")
                        x,y = x.view(np.uint32),y.view(np.uint32)
                    delta[field] = int(np.count_nonzero(x != y))
                state = new["state"]
                measured = {"canvas":int(state.size), "scoreable":int(new["valid"].sum()),
                            "high":int((state == 1).sum()), "low":int((state == 0).sum()), "unknown":int((state == 255).sum())}
                require(spec["counts"] == measured, "recorded per-target counts differ from the output")
                rows.append({"target": v, "counts": measured, "value_differences": delta,
                             "expected_npz_sha256": hashlib.sha256(native).hexdigest(),
                             "windows_npz_sha256": sha(path)})
        expected_html = package.read("prepared-report.html")
        original_images = picture_bytes(expected_html)
    returned_html = (win / "report.html").read_bytes()
    win_images = picture_bytes(returned_html)
    require(len(original_images) == len(win_images) == 14 and original_images == win_images
            and hashlib.sha256(expected_html).hexdigest() == claimed["reference_html_sha256"]
            and scores["counts"] == {k:sum(r["counts"][k] for r in rows) for k in rows[0]["counts"]}
            and [r["target"] for r in rows] == ["70","109","13"]
            and all(not any(r["value_differences"].values()) for r in rows) and receipt["status"] == "succeeded"
            and sha(win / "report.html") == claimed["windows_html_sha256"], "portable readback differs")
    environment = load(win / "environment-cpu.json")
    run = load(win.parent / "run.json")
    stages = json.loads((win / "stages.json").read_bytes())
    require(isinstance(stages, list) and tuple(s["stage"] for s in stages) == NATIVE_STAGES
            and all(s["exit_code"] == 0 for s in stages),
            "unexpected, missing or unsuccessful portable replay stage")
    with (win.parent / "telemetry.csv").open(newline="") as f:
        rss = max(int(row["tree_rss_bytes"]) for row in csv.DictReader(f) if row["tree_rss_bytes"])
    require(environment["offline_install"] and not any(environment["optional_importable"].values())
            and environment["python"] == receipt["cpu_environment"]["python"]
            and stages == receipt["timing"]["stages"]
            and run["execution_status"] == receipt["status"]
            and run["elapsed_seconds_monotonic"] == receipt["timing"]["recorder_elapsed_seconds"]
            and rss == receipt["timing"]["sampled_tree_rss_peak_bytes"],
            "Windows replay did not use the declared CPU-only installation")
    save(out / "windows-cpu.json", {
        "schema": "el-b2-windows-cpu-independent-1", "run_id": os.environ["EL_RUN_ID"],
        "portable": pins["portable"], "windows": windows, "history": pins["history"], "rows": rows,
        "counts": scores["counts"], "python": receipt["cpu_environment"]["python"],
        "distributions": environment["distributions"], "release_members": info["total_members"],
        "upstream_archive_sha256": pins["portable"]["sha256"], "return_zip_bytes": receipt["bytes"],
        "windows_measured_archive_sha256": info["zip"], "reference_html_hash_recomputed": True,
        "return_zip_sha256": receipt["sha256"], "wheel_sha256": info["wheel_sha256"],
        "html": {"same_bytes": expected_html == returned_html, "embedded_images": len(original_images),
                 "images_with_different_decoded_rgb_or_dimensions": sum(a != b for a,b in zip(original_images, win_images, strict=True)),
                 "windows_sha256": claimed["windows_html_sha256"], "reference_sha256": claimed["reference_html_sha256"]},
        "stages_seconds": {s["stage"]:s["wall_seconds"] for s in stages},
        "native_stages": [{k: s[k] for k in ("stage", "exit_code", "wall_seconds")} for s in stages],
        "windows_recorder_elapsed_s": receipt["timing"]["recorder_elapsed_seconds"],
        "windows_sampled_tree_rss_peak_bytes": receipt["timing"]["sampled_tree_rss_peak_bytes"],
        "target_photographs": 0, "passed": True,
    })


def main(a):
    s, old, prior = a.store.resolve(), a.history.resolve(), a.first.resolve()
    if a.action == "history":
        pins = load(a.config)["sources"]
        ext = missing_parents(s, [pins["portable"]])
        result = verify_external(prior, old, ext)
        save(a.out / "history.json", {"schema":"el-b2-portable-historical-audit-1", "portable":pins["portable"], **result})
        return
    receipt = load(a.receipt)
    if a.action == "compare":
        return readback((s, old), load(a.config)["sources"], receipt, a.out)
    require(RUN_ID.fullmatch(receipt["run"]) and receipt["package"] == receipt["run"] + ".zip",
            "unexpected return filename")
    archive = a.receipt.parent / receipt["package"]
    require(sha(archive) == receipt["sha256"] and archive.stat().st_size == receipt["bytes"], "changed Windows return")
    with zipfile.ZipFile(archive) as archive_file:
        meta = json.loads(archive_file.read("manifest.json"))
        require(len(meta["parents"]) == 1 and meta["run_id"] == receipt["run"], "unexpected Windows parents")
        src = meta["parents"][0]
    portable = bind(s, src["run_id"], "payload/evidence-lens-assets-v0.4.0b1.zip")
    require(portable["manifest_sha256"] == src["manifest_sha256"]
            and portable["sha256"] == a.portable_sha256, "Windows used a different publication candidate")
    args = ["--store", str(s), "--history", str(old), "--first", str(prior), "--config", "{config}", "--out", "{out}"]
    proof = stage(s, old, "portable-windows-history", "archive_verify", {"portable":portable},
                  {"scope":"Deep cross-store verification of the one Windows CPU replay input"},
                  ["{python}", "{source}/scripts/b2_portable_windows.py","history",*args], "payload/history.json", 180000)
    history = load(resolve((s,old),proof))
    anchors = {}
    for rid, digest in history["historical_parents"].items():
        _, folder = original(prior, old, rid, digest)
        v = verify_bundle(folder)
        require(v["manifest_sha256"] == history["checked_manifests"][rid] == digest, "historical root changed")
        anchors[rid] = v
    directory = s / "runs" / receipt["run"]
    if not directory.is_dir():
        require(a.action == "receive", "admit the original package before independent verification")
        incoming = receive_verified_zip(archive, s, receipt["sha256"], Path("configs/storage.json"), anchors, proof)
        require(incoming["status"] in ("accepted","identical") and incoming["manifest_sha256"] == receipt["manifest_sha256"],
                "Windows package failed verified admission")
    win = bind(s, receipt["run"], "payload/comparison.json")
    verified = verify_bundle(directory)
    require(verified["manifest_sha256"] == win["manifest_sha256"] == receipt["manifest_sha256"],
            "local Windows original differs from its returned manifest")
    verify_dependencies(verified, s, dict(anchors))
    details = {"platform":"Windows clean native CPU replay, followed by independent Mac numerical readback"}
    previous = []
    for rid in a.corrects:
        require(RUN_ID.fullmatch(rid), "invalid corrected run identity")
        failed = s / "runs" / rid
        if not failed.is_dir():
            failed = s / "quarantine" / rid
        failed_seal = verify_bundle(failed)
        failed_run = failed_seal["run"]
        require(failed_run["trial_id"] in {"b2-portable-windows-independent-v2", "b2-portable-windows-independent-v3"}
                and failed_run["execution_status"] == "failed", "expected an unsuccessful Windows engineering check")
        previous.append({"run_id": rid, "manifest_sha256": failed_seal["manifest_sha256"],
                         "run_sha256": sha(failed / "run.json"), "trial": failed_run["trial_id"]})
    if previous:
        details["previous_engineering_attempts"] = previous
    return stage(s, old, "portable-windows-independent-v5", "archive_verify", {"portable":portable,"windows":win,"history":proof},
                 details,
                 ["{python}","{source}/scripts/b2_portable_windows.py","compare",*args,"--receipt","{input:sidecar}"],
                 "payload/windows-cpu.json", 20000, {"sidecar": a.receipt}, a.retry_of)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["history","compare","receive","verify"])
    for field in ("store","first","history","receipt","config","out"):
        p.add_argument("--"+field,type=Path)
    p.add_argument("--portable-sha256")
    p.add_argument("--retry-of")
    p.add_argument("--corrects", action="append", default=[])
    print(json.dumps(main(p.parse_args()),sort_keys=True))
