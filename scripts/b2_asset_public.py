"""Assemble only the authored geometry, validated CPU runtime, deployment outputs and numerical checks."""

import argparse
import csv
import io
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
from bundle_build import environment, run
from bundle_original import check, put, sha
import tempfile
import numpy as np


def load(p):
    return json.loads(p.read_bytes())


def pack(path, files):
    with ZipFile(path, "w") as z:
        for name, raw in sorted(files.items()):
            i = ZipInfo(name, (2000, 1, 1, 0, 0, 0))
            i.compress_type = ZIP_DEFLATED
            z.writestr(i, raw)
    with ZipFile(path) as z:
        check(z.testzip() is None, "ZIP checksum failure")


def table(records):
    s = io.StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=list(records[0]))
    w.writeheader()
    w.writerows(records)
    return s.getvalue().encode()


def work(a):
    req = load(a.pins)
    win = load(a.windows / "parity.json")
    ind = load(a.metrics / "readback.json")
    check(
        ind["passed"]
        and ind["rows"] == 30
        and ind["windows_parity_sha256"] == sha((a.windows / "parity.json").read_bytes()),
        "unverified CUDA parity",
    )
    cli = load(a.cpu)
    parser = load(a.parser)
    source_meta = load(a.fixture / "fixture.json")
    empirical = load(a.external / "grade.json")
    original = load(a.uncalibrated / "grade.json")
    check(
        cli["wheel_sha256"]
        == original["wheel_sha256"]
        == empirical["wheel_sha256"]
        == sha(a.wheel.read_bytes()),
        "different distributed wheel",
    )
    check(
        original["bundle_sha256"]
        == empirical["bundle_sha256"]
        == sha((a.windows / "bundles/graphdeco-sh3/binary/bundle.json").read_bytes()),
        "wrong prepared example",
    )
    check(
        empirical["calibration"]["status"] == "outside_calibration_scope"
        and original["calibration"] is None
        and original["counts"] == empirical["counts"],
        "different reported domain",
    )
    for row in load(a.uncalibrated / "scores/scores.json")["views"]:
        q = row["arrays"]
        one = a.uncalibrated / "scores" / q["path"]
        two = a.external / "scores" / q["path"]
        with np.load(one, allow_pickle=False) as x, np.load(two, allow_pickle=False) as y:
            check(
                np.array_equal(x["risk"].view(np.uint32), y["risk"].view(np.uint32))
                and all(np.array_equal(x[k], y[k]) for k in ("valid", "state", "unknown_reason")),
                "wrong-scope profile altered original arrays",
            )
    outcomes = {}
    for cond, gfile, pfile in (("M2", a.fixed2, a.profile2), ("M4", a.fixed4, a.profile4)):
        g = load(gfile)
        p = load(pfile)
        cal = g["calibration"]
        check(
            g["passed"]
            and cal["profile_sha256"] == sha(pfile.read_bytes())
            and cal["status"] == p["status"]
            and not cal["mismatches"],
            "scope or profile differs",
        )
        check(
            len(g["original_case_rows"]) == 6
            and all(
                r["different_states"]
                == r["different_float32_words_including_NaN"]
                == r["different_validity"]
                == 0
                for r in g["original_case_rows"]
            ),
            "default score changed",
        )
        outcomes[cond] = {
            "status": cal["status"],
            "threshold": cal["threshold"],
            "applied": cal["applied"],
            "original_threshold": cal["original_threshold"],
            "original_counts": g["counts"],
            "profile_sha256": sha(pfile.read_bytes()),
            "scope_sha256": cal["scope_sha256"],
            "holdout_evaluated": cal["evaluated_on_holdout"],
            "holdout_target_met": cal["holdout_target_met"],
            "recorded_original_comparisons": len(g["original_case_rows"]),
        }
    files = {
        a.wheel.name: a.wheel.read_bytes(),
        "profiles/M2.json": a.profile2.read_bytes(),
        "profiles/M4.json": a.profile4.read_bytes(),
        "prepared-report.html": (a.uncalibrated / "report.html").read_bytes(),
        "outside-scope-report.html": (a.external / "report.html").read_bytes(),
        "render-parity.csv": (a.metrics / "render-parity.csv").read_bytes(),
        "LICENSE": a.license.read_bytes(),
        "NOTICE": a.notice.read_bytes(),
        "licenses/example.txt": a.example_license.read_bytes(),
        "licenses/model.txt": a.model_license.read_bytes(),
        "licenses/research.txt": a.record_license.read_bytes(),
        "licenses/CC-BY-4.0.txt": a.ccby.read_bytes(),
    }
    for n, v in source_meta["files"].items():
        raw = (a.fixture / n).read_bytes()
        check(sha(raw) == v["sha256"], "changed authored input " + n)
        files["authored/" + n] = raw
    files["authored/fixture.json"] = (a.fixture / "fixture.json").read_bytes()
    dest = a.windows / "bundles/graphdeco-sh3/binary"
    for f in dest.rglob("*"):
        if f.is_file():
            files["prepared/" + f.relative_to(dest).as_posix()] = f.read_bytes()
    for n in ("graphdeco-sh0", "graphdeco-sh3", "project-npz"):
        files["comparisons/" + n + ".png"] = (a.windows / (n + "-comparison.png")).read_bytes()
    scores = load(a.uncalibrated / "scores/scores.json")
    for r in scores["views"]:
        v = r["id"]
        files["expected/" + v + ".npz"] = (a.uncalibrated / "scores" / r["arrays"]["path"]).read_bytes()
    files["README.txt"] = (
        """Evidence Lens original helical bead study | 0.4.0b1\nInstall the included wheel in a fresh Python 3.11+ environment with NumPy/Pillow.\n\npython -m pip install ./"""
        + a.wheel.name
        + """\nevidence-lens validate --bundle prepared\nevidence-lens score --bundle prepared --out scores\nevidence-lens report --scores scores --out new-report.html\n\nCPU inspection from the two authorized sources:\nevidence-lens inspect --colmap authored/colmap/text --encoding text --images authored/images --gaussians authored/graphdeco-sh3.ply --gaussian-profile graphdeco-v1 --plan authored/plan.json --out inspected\n\nUse render-asset in place of inspect on CUDA gsplat 1.5.3 to produce a new bundle. Alternative assets are authored/project.npz (evidence-lens-npz-v1) and authored/graphdeco-sh0.ply (graphdeco-v1). Inputs are native 256 x 256. The two supplied C6 profiles have no feasible registered operating point; both belong to fixed DTU layouts. The original Gaussian study is outside both domains. The artifact includes only this original study's source and predicted pictures.\n"""
    ).encode()
    fees = []
    for r in parser["cases"]:
        fees.append(
            {
                "stage": "CPU inspection",
                "representation": r["asset"],
                "format": r["encoding"],
                "seconds": r["elapsed_seconds"],
            }
        )
    for r in win["timings"]:
        fees.append(
            {
                "stage": r["stage"],
                "representation": r["case"],
                "format": "native" if r["stage"] == "render_native_reference" else r["stage"].split("_")[-1],
                "seconds": r["seconds"],
            }
        )
    fees.append(
        {
            "stage": "CPU scoring, within worker",
            "representation": "graphdeco-sh3",
            "format": "prepared",
            "seconds": scores["cost"]["worker_to_receipt_seconds"],
        }
    )
    fees.append(
        {
            "stage": "Windows recorder, including interpreter and worker",
            "representation": "all three assets",
            "format": "text+binary",
            "seconds": load(a.return_file)["recorder_seconds"],
        }
    )
    files["measured-stage-times.csv"] = table(fees)
    checks = {
        "schema": "el-b2-public-asset-acceptance-1",
        "run_id": os.environ["EL_RUN_ID"],
        "origins": req,
        "positive_CUDA_imports": 6,
        "native_camera_comparisons": ind["rows"],
        "channels": ind["channels"],
        "camera_max_abs": ind["camera_max_abs"],
        "source_ids": scores["source_order"],
        "target_ids": scores["target_order"],
        "default_threshold": scores["model"]["threshold"],
        "original_example_counts": scores["counts"],
        "external_profile_status": empirical["calibration"]["status"],
        "in_scope_set48": outcomes,
        "upstream_reader_sha256": parser["official_COLMAP_reader_sha256"],
        "positive_CPU_imports": len(cli["positive"]),
        "negative_CPU": cli["negative"],
        "device": win["device"],
        "gsplat": win["gsplat"],
        "torch": win["torch"],
        "cuda": win["cuda"],
        "cuda_allocator_peak_bytes": {"allocated": win["peak_allocated"], "reserved": win["peak_reserved"]},
        "wheel_sha256": sha(a.wheel.read_bytes()),
        "fixture_sha256": ind["fixture_sha256"],
        "protocol_sha256": ind["protocol_sha256"],
        "renderer_parity_limits": {"RGB": 0.0001, "alpha": 0.0001, "ED_m": 0.0001, "camera": 1e-8},
        "target_error_labels": 0,
    }
    with ZipFile(a.wheel) as w, ZipFile(a.ticket) as z:
        mods = [n for n in w.namelist() if n.startswith("evidence_lens_asset/") and n.endswith(".py")]
        check(mods and all(w.read(n) == z.read("runtime/" + n) for n in mods), "CUDA ticket and wheel differ")
        checks["windows_asset_module_sha256"] = {n: sha(w.read(n)) for n in sorted(mods)}
    out = a.out
    put(out / "verification.json", checks)
    files["verification.json"] = (out / "verification.json").read_bytes()
    for n in ("render-parity.csv", "measured-stage-times.csv"):
        (out / n).write_bytes(files[n])
    index = {n: {"bytes": len(b), "sha256": sha(b)} for n, b in sorted(files.items())}
    files["manifest.json"] = json.dumps(index, sort_keys=True, indent=2).encode() + b"\n"
    zip_path = out / "evidence-lens-assets-v0.4.0b1.zip"
    pack(zip_path, files)
    with tempfile.TemporaryDirectory(dir=out.parent) as t:
        base = Path(t)
        case = base / "demo"
        case.mkdir()
        with ZipFile(zip_path) as z:
            z.extractall(case)
        exe, packages = environment(base, case / a.wheel.name)
        run([exe, "validate", "--bundle", case / "prepared", "--bundle-sha256", empirical["bundle_sha256"]])
        root = base / "scores"
        run([exe, "score", "--bundle", case / "prepared", "--out", root])
        for v in load(root / "scores.json")["views"]:
            with (
                np.load(root / v["arrays"]["path"], allow_pickle=False) as x,
                np.load(case / "expected" / (v["id"] + ".npz"), allow_pickle=False) as y,
            ):
                check(
                    np.array_equal(x["risk"].view(np.uint32), y["risk"].view(np.uint32))
                    and all(np.array_equal(x[k], y[k]) for k in ("valid", "state", "unknown_reason")),
                    "clean portable replay differs",
                )
        page = base / "example.html"
        run([exe, "report", "--scores", root, "--out", page])
    put(
        out / "portable.json",
        {
            "schema": "el-b2-public-asset-distribution-1",
            "run_id": os.environ["EL_RUN_ID"],
            "file": zip_path.name,
            "bytes": zip_path.stat().st_size,
            "sha256": sha(zip_path.read_bytes()),
            "verification_sha256": sha(files["verification.json"]),
            "manifest_sha256": sha(files["manifest.json"]),
            "members": len(files),
            "clean_replay_pixels": scores["counts"]["canvas"],
            "clean_replay_states_different": 0,
            "clean_replay_float32_words_different": 0,
            "clean_distributions": packages,
            "example_html_sha256": sha(files["prepared-report.html"]),
            "outside_html_sha256": sha(files["outside-scope-report.html"]),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in (
        "pins",
        "windows",
        "metrics",
        "cpu",
        "parser",
        "fixture",
        "external",
        "uncalibrated",
        "fixed2",
        "fixed4",
        "profile2",
        "profile4",
        "wheel",
        "ticket",
        "return-file",
        "license",
        "notice",
        "example-license",
        "model-license",
        "record-license",
        "ccby",
        "out",
    ):
        p.add_argument("--" + n, type=Path, required=True)
    work(p.parse_args())
