"""Check a locally received Stage C candidate and run its complete two-condition example in a fresh CPU environment."""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from vision_release import entries, read, require, safe_name, sha, sig, write

TAG = "v0.3.0-rc.1"
SHARED = {"LICENSE", "NOTICE", "licenses/CC-BY-4.0.txt"}


def extract(path, spec, dest):
    require(
        sig(path) == {k: spec[k] for k in ("bytes", "sha256")} and path.name == spec["name"],
        "candidate ZIP differs from its independent recorded digest: " + path.name,
    )
    dest.mkdir()
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        require(
            names == sorted(set(names))
            and len(names) == spec["members"]
            and len({s.casefold() for s in names}) == len(names)
            and sum(r.file_size for r in z.infolist()) < 2 * 1024**3,
            "invalid or excessive ZIP inventory",
        )
        for r in z.infolist():
            n = safe_name(r.filename)
            p = dest / n
            require(r.external_attr >> 16 == 0o100644 and not r.is_dir(), "unexpected member type")
            p.parent.mkdir(parents=True, exist_ok=True)
            with z.open(r) as f, p.open("xb") as out:
                shutil.copyfileobj(f, out, 2**20)
    idx = dest / spec["index"]
    require(sha(idx) == spec["index_sha256"], "candidate index differs: " + spec["name"])
    doc = read(idx)
    files = doc["files"]
    require(set(names) == set(files) | SHARED | {spec["index"]}, "unexpected or missing package member")
    for n, s in files.items():
        require(
            sig(dest / safe_name(n)) == {k: s[k] for k in ("bytes", "sha256")},
            "changed candidate member: " + n,
        )
    return doc


def verify(args):
    from verify_vision_public import install

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), "use a fresh validation folder")
    phase = "independent local candidate"
    candidate = args.candidate.resolve()
    try:
        require(sha(candidate) == args.candidate_sha256, "independent candidate SHA differs")
        c = read(candidate)
        require(
            c["schema"] == "el-stage3-candidate-1"
            and c["version"] == TAG
            and c["commit"] == args.commit
            and set(c["assets"]) == {"cpu", "numerical", "example"},
            "wrong publication candidate",
        )
        with tempfile.TemporaryDirectory(prefix="stage3-cpu-acceptance-") as td:
            temp = Path(td)
            parts = {}
            phase = "verify all recipient ZIPs"
            for name, s in c["assets"].items():
                parts[name] = extract(candidate.parent / safe_name(s["name"]), s, temp / name)
            require(
                all(
                    d["commit"] == args.commit
                    and d["version"] == TAG
                    and d["method_sha256"] == c["method_sha256"]
                    for d in parts.values()
                )
                and parts["cpu"]["models"] == c["models"]
                and parts["cpu"]["expected_example_sha256"] == c["assets"]["example"]["index_sha256"]
                and parts["cpu"]["numerical_sha256"] == c["assets"]["numerical"]["index_sha256"]
                and parts["cpu"]["heldout"] == parts["numerical"]["heldout"],
                "public ZIPs refer to different release sources",
            )
            require(
                all(len({sha(temp / n / f) for n in parts}) == 1 for f in SHARED),
                "original attribution/license mismatch across archives",
            )
            phase = "install original CPU dependencies"
            exe = install(temp, temp / "cpu/requirements.txt", args.wheelhouse)
            shutil.copyfile(temp / "installation.json", out / "installation.json")
            phase = "score independently from native calibrated input and recompute reference error"
            work = temp / "fresh-work"
            work.mkdir()
            cmd = [
                str(exe),
                "-I",
                "-B",
                str(temp / "cpu/cpu.py"),
                "replay",
                "--release-sha256",
                c["assets"]["cpu"]["index_sha256"],
                "--example",
                str(temp / "example"),
                "--example-sha256",
                c["assets"]["example"]["index_sha256"],
                "--threads",
                str(args.threads),
                "--output",
                str(out / "replay"),
            ]
            with (out / "cpu-replay.log").open("xb") as log:
                result = subprocess.run(cmd, cwd=work, stdout=log, stderr=log)
            require(result.returncode == 0, "native CPU replay failed; inspect cpu-replay.log")
            grade = read(out / "replay/grade.json")
            require(
                grade["status"] == "passed"
                and grade["method_sha256"] == c["method_sha256"]
                and grade["models"] == c["models"]
                and grade["scheduled_comparisons"] == 144
                and grade["distinct_method_scope_views"] == 120
                and len(grade["reference_evaluations"]) == 12,
                "public CPU replay incomplete",
            )
            # Exercise the public reader's default single-view route, including visual files.
            phase = "render single-view risk and state PNGs"
            single = out / "default-v01"
            deployment = temp / "isolated-default-input"
            deployment.mkdir()
            original_case = temp / "example/CASE/M4"
            case_doc = read(original_case / "case.json")
            allowed = {
                "case.json",
                case_doc["prediction"]["path"],
                case_doc["source_cameras"]["path"],
                case_doc["render_cameras"]["path"],
                case_doc["renders"]["v01"]["path"],
                *(s["path"] for s in case_doc["source_images"].values()),
                *(case_doc["renders"][s]["path"] for s in case_doc["generator_source_ids"]),
            }
            for name in sorted(allowed):
                dest = deployment / safe_name(name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original_case / name, dest)
            with (out / "default-v01.log").open("xb") as log:
                res = subprocess.run(
                    [
                        str(exe),
                        "-I",
                        "-B",
                        str(temp / "cpu/cpu.py"),
                        "run",
                        "--release-sha256",
                        c["assets"]["cpu"]["index_sha256"],
                        "--case",
                        str(deployment),
                        "--case-sha256",
                        parts["example"]["case_sha256"]["M4"],
                        "--threads",
                        str(args.threads),
                        "--view",
                        "v01",
                        "--scope",
                        "all",
                        "--output",
                        str(single),
                    ],
                    cwd=work,
                    stdout=log,
                    stderr=log,
                )
            require(res.returncode == 0, "default risk/state visualization failed")
            score = read(single / "score.json")
            require(
                score["method"] == c["default_method"]
                and score["shape"] == [256, 256]
                and set(score["images"]) == {"risk", "state"},
                "default single-view output differs",
            )
            for k, p in score["images"].items():
                require(
                    sig(single / (k + ".png")) == {j: p[j] for j in ("bytes", "sha256")},
                    "broken default native map",
                )
            require(
                score["score"]["sha256"]
                == sha(out / "replay/scores" / c["default_method"] / "M4/all/v01.npz"),
                "physically isolated prediction-only route differs from complete native replay",
            )
        doc = {
            "schema": "el-stage3-local-candidate-verification-1",
            "status": "verified_full",
            "version": TAG,
            "commit": args.commit,
            "candidate_sha256": args.candidate_sha256,
            "platform": platform.platform(),
            "source_mode": "local pinned candidate",
            "assets": c["assets"],
            "models": c["models"],
            "method_sha256": c["method_sha256"],
            "default_method": c["default_method"],
            "grade_sha256": sha(out / "replay/grade.json"),
            "default_score_sha256": sha(out / "default-v01/score.json"),
            "native_pngs": score["images"],
            "isolated_deployment_files": sorted(allowed),
            "reference_evaluations": 12,
            "scheduled_comparisons": 144,
            "distinct_comparisons": 120,
            "max_abs_error": grade["max_abs_error"],
            "threshold_crossings": grade["threshold_crossings"],
            "installation_sha256": sha(out / "installation.json"),
        }
        write(out, "local-verification.json", entries(doc))
        return doc
    except Exception as exc:
        write(
            out,
            "failure.json",
            entries({"version": TAG, "phase": phase, "type": type(exc).__name__, "message": str(exc)}),
        )
        raise


if __name__ == "__main__":
    if sys.version_info < (3, 11):
        raise SystemExit("Stage C verification requires Python 3.11 or newer.")
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("candidate", "output"):
        parser.add_argument("--" + n, type=Path, required=True)
    for n in ("candidate-sha256", "commit"):
        parser.add_argument("--" + n, required=True)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    print(json.dumps(verify(parser.parse_args()), sort_keys=True))
