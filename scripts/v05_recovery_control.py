"""Register a single-camera release renderer check and evaluate its returned prediction against the original."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np

from dual_records import bind, resolve, selected, stage
from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens._verify import require
from t0_reconstruct_intake import receive_verified_zip
from v05_bell_checkpoint import NATIVE
from v05_bell_receive import history_roots


CAMERAS = {"run_id": "20261124T041057887060Z-ed115219ffb44fa7",
           "manifest_sha256": "8caeeca92b65eeab9372ea1322d2b3c69b8c1272492b8631b239789acd15afb8",
           "member": "payload/source_cameras.json", "sha256": "e99df6b5b685aa6750f46dab17c161fb00225a83ad06be4a1cc8f403b515ea3a"}
ORIGINAL = {"run_id": "20261124T042109162295Z-4868a909bf114eb6",
            "manifest_sha256": "cd4ab5558ba5d3a751671b4f5c3cffb60e422bd909bfa2391a87ba4d94695b66",
            "member": "payload/step-5000/s00.npz", "sha256": "8e46274e2c28381242886b35a8c17fc26f1a8f08f243ff701c1e428e6d41568c"}
LIMITS = {"rgb": 1e-6, "alpha": 1e-6, "z": 1e-5}
ROOT = Path(__file__).resolve().parents[1]
MODULES = {"evidence_lens": ["__init__", "features", "scoring_io", "multisource", "multisource_io"],
           "evidence_lens_asset": ["__init__", "gaussian", "prepare", "render", "permissions", "colmap"],
           "evidence_lens_bundle": ["__init__", "bundle", "common", "grids"]}


def command(action, pins):
    cmd = ["{python}", "-B", "{source}/scripts/v05_recovery_control.py", action, "--out", "{out}", "--config", "{config}"]
    for key in pins:
        cmd += ["--input", key, "{input:" + key + "}"]
    return cmd


def ticket(s):
    pins = {"entry": selected(s, "a-grid-registration"), "prior": selected(s, "a-upstream-gaussian-contract"),
            "checkpoint": NATIVE, "cameras": CAMERAS}
    details = {"source_camera": "s00", "raw_and_native_activations": "all five byte-for-byte",
               "holdback": ORIGINAL, "maximum_absolute_discrepancy": LIMITS, "require_identical_finite_mask": True,
               "registered_gaussian_count": 16251, "training": False, "photograph_reads": 0, "private_originals": True}
    return stage(s, "a-recovery-render-control-ticket", "dataset_build", pins, details,
                 command("ticket-worker", pins), "payload/dispatch.json", 12_000_000)


def pack(cfg, paths, out):
    source = load(paths["cameras"])
    require(source["schema"] == "el-camera-1" and source["pixel_center"] == "integer" and source["unit"] == "m"
            and set(source["views"]) == {"s00", "s03"} and cfg["sources"]["checkpoint"] == NATIVE
            and sha(paths["checkpoint"]) == NATIVE["sha256"], "wrong original checkpoint, camera or image domain")
    cam = source["views"]["s00"]
    physical = {k: np.asarray(cam[k], dtype=np.float64).tolist() for k in ("K", "c2w", "clip_m")}
    physical.update(width=cam["width"], height=cam["height"])
    raw = {"camera.json": encoded(physical), "checkpoint.pt": paths["checkpoint"].read_bytes(),
           "run_gpu.py": (ROOT / "scripts/v05_recovery_gpu.py").read_bytes()}
    for pkg, modules in MODULES.items():
        for module in modules:
            name = f"{pkg}/{module}.py"
            raw["runtime/" + name] = (ROOT / "src" / name).read_bytes()
    frozen = {"runtime/evidence_lens_asset/gaussian.py": "2355addcfcb78381afbbbededfbb6408fc1921f240e49a32ebc132fbb0da6d90",
              "runtime/evidence_lens_asset/render.py": "ae819e6dda86b898594537172297f8c1351049c185aef9894e63f991e45deff1"}
    hashes = {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}
    require(all(hashes[k] == h for k, h in frozen.items()), "actual release decoder or renderer changed")
    authority = {"schema": "el-v05-recovery-render-ticket-1", "run_id": os.environ["EL_RUN_ID"],
                 "source_view": "s00", "gaussians": cfg["registered_gaussian_count"], "cap": 20000,
                 "files": hashes, "checkpoint": NATIVE, "source_cameras": CAMERAS,
                 "maximum_absolute_discrepancy": LIMITS, "allowed_photographs": [], "allowed_target_views": []}
    raw["ticket.json"] = encoded(authority)
    archive = out / "recovery-gpu.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, content in sorted(raw.items()):
            z.writestr(name, content)
    save(out / "dispatch.json", {"schema": "el-v05-recovery-render-dispatch-1", "run_id": os.environ["EL_RUN_ID"],
         "archive": archive.name, "archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size,
         "gpu_ticket_sha256": hashlib.sha256(raw["ticket.json"]).hexdigest(), "checkpoint": NATIVE, "source_cameras": CAMERAS,
         "operator_source": frozen, "registered_holdback": ORIGINAL, "maximum_absolute_discrepancy": LIMITS,
         "windows_command": "python -I -B run_gpu.py --out EMPTY_RECORDER_PAYLOAD", "release_of_raw_data": False})


def intake(s, side):
    receipt = load(side)
    archive = side.parent / receipt["package"]
    rid, manifest = receipt["run_id"], receipt["manifest_sha256"]
    require(receipt["package"] == rid + ".zip" and receipt["status"] == "succeeded" and receipt["worker_exit_code"] == 0
            and archive.stat().st_size == receipt["bytes"] and sha(archive) == receipt["sha256"],
            "native return, disposition or receipt size/hash differs")
    ticket_pin = selected(s, "a-recovery-render-control-ticket")
    dispatch = load(resolve((s, s/"history"), ticket_pin))
    with zipfile.ZipFile(archive) as z:
        info, inputs = (json.loads(z.read(n)) for n in ("run.json", "inputs.json"))
        require(info["run_id"] == rid and info["experiment_id"] == "dual-track-a" and info["trial_id"] == "a-recovery-gpu"
                and info["run_kind"] == "render" and info["execution_status"] == "succeeded", "unexpected returned experiment")
        shas = {row["sha256"] for item in inputs["inputs"] for row in item["files"]}
        require(dispatch["archive_sha256"] in shas and any(p["run_id"] == ticket_pin["run_id"] and
                p["manifest_sha256"] == ticket_pin["manifest_sha256"] for p in inputs["parents"]), "unbound original dispatch")
    anchors = history_roots(s, inputs["parents"])
    imported = receive_verified_zip(archive, s, receipt["sha256"], ROOT/"configs/storage.json", anchors, ticket_pin)
    require(imported["status"] in ("accepted", "identical") and imported["run_id"] == rid
            and imported["manifest_sha256"] == manifest, "native recorded archive verification failed")
    pins = {"dispatch": ticket_pin, "native": bind(s, rid, "payload/recovery.json"),
            "prediction": bind(s, rid, "payload/s00.npz"), "original": ORIGINAL}
    details = {"native_archive_sha256": receipt["sha256"], "native_archive_bytes": receipt["bytes"], "limits": LIMITS}
    return stage(s, "a-recovery-render-agreement", "evaluation", pins, details, command("compare-worker", pins),
                 "payload/agreement.json", 18_000, extra_files={"transport": side})


def compare(cfg, paths, out):
    receipt, dispatch = load(paths["native"]), load(paths["dispatch"])
    require(receipt["schema"] == "el-v05-recovery-render-1" and receipt["run_id"] == cfg["sources"]["native"]["run_id"]
            and receipt["ticket_run"] == dispatch["run_id"] and receipt["checkpoint_sha256"] == NATIVE["sha256"]
            and receipt["prediction"]["sha256"] == sha(paths["prediction"]) and dispatch["registered_holdback"] == ORIGINAL
            and sha(paths["original"]) == ORIGINAL["sha256"] and receipt["source_view"] == "s00"
            and receipt["gaussians"] == 16251 and not receipt["previous_prediction_available_to_gpu"]
            and receipt["source_photo_reads"] == receipt["target_reference_reads"] == receipt["new_optimization_steps"] == 0,
            "incorrect engineering control exposure or provenance")
    require(len(receipt["parameters"]) == 5 and all(row["raw_exact"] and row["activation_exact"]
            for row in receipt["parameters"].values()), "raw or activated native tensors differ")
    results = {}
    with np.load(paths["prediction"], allow_pickle=False) as current, np.load(paths["original"], allow_pickle=False) as expected:
        require(set(current.files) == set(expected.files) == set(LIMITS), "native frame field roster changed")
        for key, tolerance in LIMITS.items():
            a, b = current[key], expected[key]
            require(a.dtype == b.dtype == np.float32 and a.shape == b.shape, "native frame shape or dtype differs")
            m, n = np.isfinite(a), np.isfinite(b)
            require(np.any(m & n) and (key == "z" or (m.all() and n.all())), "no finite evidence on common calibrated frame")
            exact_mask = np.array_equal(m, n)
            x, y = a[m & n], b[m & n]
            max_error = float(np.abs(x.astype(np.float64) - y).max())
            results[key] = {"shape": list(a.shape), "finite_mask_exact": exact_mask, "common_finite": int(x.size),
                            "finite_value_bytes_exact": x.tobytes() == y.tobytes(),
                            "max_absolute_error": max_error, "limit": tolerance, "passed": bool(exact_mask and max_error <= tolerance)}
    save(out / "agreement.json", {"schema": "el-v05-release-recovery-agreement-1", "run_id": os.environ["EL_RUN_ID"],
         "passed": all(r["passed"] for r in results.values()), "source_view": "s00", "original_predicted_frame": ORIGINAL,
         "inputs": cfg["sources"], "raw_and_activated_parameters": receipt["parameters"], "gaussians": receipt["gaussians"],
         "native_render": receipt["runtime"], "clip_m": receipt["clip_m"], "recipe": receipt["model"], "predictions": results,
         "photographs_opened": 0, "old_experiment_reruns": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("ticket", "receive", "ticket-worker", "compare-worker"))
    for key in ("store", "config", "out", "receipt"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--input", action="append", nargs=2, default=[])
    args = parser.parse_args()
    if args.action.endswith("-worker"):
        fn = pack if args.action == "ticket-worker" else compare
        fn(load(args.config), {k: Path(v) for k, v in args.input}, args.out)
    else:
        result = ticket(args.store.resolve()) if args.action == "ticket" else intake(args.store.resolve(), args.receipt)
        print(json.dumps(result))
