"""Seal an example against recorded CPU scores and grade an independently installed release."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .l1 import KEYS, load_package
from .scoring_io import TARGETS, file_hash, new_output, write_json


ABS_TOL = 1e-6
EXCLUDED = ("sklearn", "torch", "bpy", "gsplat", "sharp", "pyarrow")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def entry(path):
    require(path.is_file() and not path.is_symlink(), f"missing regular demo file {path.name}")
    return {"sha256": file_hash(path), "bytes": path.stat().st_size}


def public_inputs(bundle):
    model = load_package(bundle / "model")
    model_files = {"model/model.json", *("model/" + n for n in model["index"]["files"])}
    pred = read(bundle / "example/predictions/prediction.json")
    roles = {"input_png": "example/input.png", "cameras": "example/cameras.json",
             "prediction_metadata": "example/predictions/prediction.json"}
    failed = pred.get("failed_views", {})
    for vid in ("source", *TARGETS):
        if vid not in failed:
            roles["prediction/" + vid] = f"example/predictions/{vid}.npz"
    return model, roles, {name: entry(bundle / name) for name in sorted(model_files | set(roles.values()))}


def seal(bundle, score_dir, expected_receipt, views):
    require(views and len(views) == len(set(views)) and set(views) <= set(TARGETS), "invalid demo target selection")
    meta_file = score_dir / "l1_score.json"
    require(entry(meta_file)["sha256"] == expected_receipt, "recorded score receipt digest mismatch")
    rec = read(meta_file)
    model, roles, inputs = public_inputs(bundle)
    require(rec.get("schema") == "el-l1-score-1" and rec.get("status") == "scored" and rec.get("run_id")
            and rec["model_id"] == model["index"]["model_id"] and rec["model_files"] == model["file_hashes"]
            and rec["methods"] == KEYS and rec["inputs"] == {role: inputs[name] for role, name in roles.items()},
            "demo inputs or frozen model differ from the recorded score")
    expected = {}
    for vid in views:
        src = score_dir / (vid + ".npz")
        expected[vid] = entry(src)
        require(rec["views"][vid]["status"] == "scored" and rec["views"][vid]["sha256"] == expected[vid]["sha256"],
                "demo expectation must be an originally recorded scored target")
    manifest = {"schema": "el-clean-demo-1", "model_id": rec["model_id"], "score_run_id": rec["run_id"],
                "producer_run_id": rec["producer_run_id"], "original_score_receipt_sha256": expected_receipt,
                "model_index_sha256": inputs["model/model.json"]["sha256"], "input_files": inputs,
                "expected_views": expected, "methods": KEYS, "maximum_absolute_score_difference": ABS_TOL}
    dest = bundle / "expected"
    if dest.exists():
        require(all(entry(dest / (v + ".npz")) == record for v, record in expected.items()), "existing expectation conflicts")
    else:
        with new_output(dest) as tmp:
            for vid, record in expected.items():
                target = tmp / (vid + ".npz")
                shutil.copyfile(score_dir / (vid + ".npz"), target)
                require(entry(target) == record, "score bytes changed during public staging")
    path = bundle / "demo.json"
    with tempfile.TemporaryDirectory(dir=bundle) as work:
        temp = Path(work) / "demo.json"
        write_json(temp, manifest)
        try:
            os.link(temp, path)
        except FileExistsError:
            require(entry(path) == entry(temp), "existing public demo manifest conflicts")
    return {"manifest_sha256": file_hash(path), "graded_views": list(expected)}


def compare(expected, actual):
    rows = {}
    with np.load(expected, allow_pickle=False) as ref, np.load(actual, allow_pickle=False) as run:
        for method, key in KEYS.items():
            a, b = ref[key], run[key]
            require(a.shape == b.shape and a.dtype == b.dtype == np.float32, "score type or shape differs")
            af, bf = np.isfinite(a), np.isfinite(b)
            valid = all(src[key + "_valid"].dtype == np.bool_ and np.array_equal(src[key + "_valid"], mask)
                        for src, mask in ((ref, af), (run, bf)))
            same = bool(np.array_equal(af, bf) and valid and np.all(np.isnan(a[~af])) and np.all(np.isnan(b[~bf])))
            av, bv = ref[key + "_state"], run[key + "_state"]
            states = bool(av.dtype == bv.dtype == np.uint8 and np.array_equal(av, bv))
            errors = np.abs(a[af & bf].astype(np.float64) - b[af & bf])
            gap = float(errors.max()) if errors.size else None
            rows[method] = {"pixels": a.size, "finite_expected": int(af.sum()), "finite_actual": int(bf.sum()),
                            "validity_equal": same, "states_equal": states, "max_absolute_score_difference": gap,
                            "passed": same and states and (gap is None or gap <= ABS_TOL)}
        reason = "l0_reason" not in ref and "l0_reason" not in run or (
            "l0_reason" in ref and "l0_reason" in run and np.array_equal(ref["l0_reason"], run["l0_reason"]))
    return {"methods": rows, "l0_reason_codes_equal": bool(reason), "passed": bool(reason and all(m["passed"] for m in rows.values()))}


def clean_worker(bundle, output, index_sha):
    from .l1 import run
    require(sys.prefix != sys.base_prefix and sys.flags.isolated, "clean grading requires an isolated virtual environment")
    installed = [name for name in EXCLUDED if importlib.util.find_spec(name) is not None]
    require(not installed, f"clean grading found optional training/render packages: {installed}")
    distribution = importlib.metadata.distribution("evidence-lens")
    origin = Path(__file__).resolve()
    require(origin.is_relative_to(Path(sys.prefix).resolve()) and distribution.version, "install the release distribution in the grading venv")
    run(bundle / "example/input.png", bundle / "example/cameras.json", bundle / "example/predictions", bundle / "model",
        output / "scores", run_id="clean-public-demo", model_sha256=index_sha)
    require(not any(name in sys.modules for name in (*EXCLUDED, "evidence_lens.labels", "evidence_lens.evaluation")),
            "deployment loaded training or answer modules")
    write_json(output / "runtime.json", {"isolated": True, "optional_training_render_packages_present": installed,
               "python": sys.version.split()[0], "packages": {name: importlib.metadata.version(name) for name in
               ("evidence-lens", "numpy", "Pillow")}})


def grade(bundle, output, manifest_sha256, *, python=None):
    manifest_file = bundle / "demo.json"
    require(entry(manifest_file)["sha256"] == manifest_sha256, "public demo manifest hash mismatch")
    m = read(manifest_file)
    model, _, inputs = public_inputs(bundle)
    require(m["schema"] == "el-clean-demo-1" and m["methods"] == KEYS and m["maximum_absolute_score_difference"] == ABS_TOL
            and m["model_id"] == model["index"]["model_id"] and m["model_index_sha256"] == inputs["model/model.json"]["sha256"]
            and m["input_files"] == inputs and m["expected_views"] and set(m["expected_views"]) <= set(TARGETS),
            "invalid or modified public model/demo inputs")
    for vid, spec in m["expected_views"].items():
        require(entry(bundle / "expected" / (vid + ".npz")) == spec, f"changed expected {vid}")
    with new_output(output) as out, tempfile.TemporaryDirectory(prefix="el-clean-demo-inputs-") as temp:
        allowed = Path(temp)
        for name, info in inputs.items():
            dst = allowed / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundle / name, dst)
            require(entry(dst) == info, "public source changed during isolated staging")
        exe = str(Path(python or sys.executable).absolute())
        p = subprocess.run([exe, "-I", "-B", "-m", "evidence_lens.l1_demo", "_score", "--bundle", str(allowed),
                            "--output", str(out.absolute()), "--model-sha256", m["model_index_sha256"]],
                           cwd=allowed, text=True, capture_output=True)
        report = {"schema": "el-clean-demo-grade-1", "source_demo_sha256": manifest_sha256, "model_id": m["model_id"],
                  "model_index_sha256": m["model_index_sha256"], "expected_score_run_id": m["score_run_id"],
                  "absolute_tolerance": ABS_TOL, "worker_exit_code": p.returncode}
        if p.returncode:
            report.update(passed=False, error=(p.stderr or p.stdout).strip())
        else:
            measured = {vid: compare(bundle / "expected" / (vid + ".npz"), out / "scores" / (vid + ".npz")) for vid in m["expected_views"]}
            finite_l1 = sum(v["methods"]["L1-full"]["finite_expected"] for v in measured.values())
            report.update(passed=all(v["passed"] for v in measured.values()), views=measured,
                          finite_l1_example_pixels=finite_l1, numerical_l1_parity_exercised=finite_l1 > 0, runtime=read(out / "runtime.json"))
        write_json(out / "grade.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("seal")
    make.add_argument("--scores", type=Path, required=True)
    make.add_argument("--score-receipt-sha256", required=True)
    make.add_argument("--view", action="append", choices=TARGETS)
    check = commands.add_parser("grade")
    check.add_argument("--manifest-sha256", required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--python", type=Path, help="clean venv containing the installed release; defaults to this process's venv")
    work = commands.add_parser("_score", help=argparse.SUPPRESS)
    work.add_argument("--output", type=Path, required=True)
    work.add_argument("--model-sha256", required=True)
    for cmd in (make, check, work):
        cmd.add_argument("--bundle", type=Path, required=True)
    a = parser.parse_args(argv)
    if a.command == "_score":
        clean_worker(a.bundle, a.output, a.model_sha256)
        return 0
    if a.command == "seal":
        result = seal(a.bundle, a.scores, a.score_receipt_sha256, a.view or ["v01"])
    else:
        result = grade(a.bundle, a.output, a.manifest_sha256, python=a.python)
    print(json.dumps(result))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
