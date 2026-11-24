"""Restore only named public deployment assets and identify the frozen threshold profile."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


CPU = "f555956840870309217cc85fe9b339ef53762679bbd2138b7f149d8ea451a7b7"
EXAMPLE = "14204986e0d598adc613dd1a9ef2e5c1584f5be001d3541531d5d9e709707548"
INDEX = "150b04e702f5386b5808ac17545cd073e3d4d837ea789e6ae167168ac086084d"
METHODS = ("B-L1-transfer", "MS-L1", "MS-frozen", "MS-ft")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def put(path, data):
    save(path, (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def check(ok, what):
    if not ok:
        raise ValueError(what)


def restore(cpu_path, example_path, out):
    for path, expected in ((cpu_path, CPU), (example_path, EXAMPLE)):
        with path.open("rb") as source:
            check(hashlib.file_digest(source, "sha256").hexdigest() == expected, "published ZIP SHA differs")
    with ZipFile(cpu_path) as cpu, ZipFile(example_path) as ex:
        release = json.loads(cpu.read("release.json"))
        check(sha(cpu.read("release.json")) == INDEX, "release index differs")

        def fetch(archive, index, name):
            raw = archive.read(name)
            check(index[name]["sha256"] == sha(raw) and index[name]["bytes"] == len(raw), name)
            return raw

        files = release["files"]
        model_specs = {}
        for method in METHODS:
            original = release["models"][method]
            base = "models/" + method + "/"
            weight = (
                "L1-full.json"
                if method == "B-L1-transfer"
                else "coefficients.json"
                if method == "MS-L1"
                else "weights.safetensors"
            )
            pre = "L1-full.json" if method == "B-L1-transfer" else "preprocessing.json"
            feature = (
                "models/B-L1-transfer/feature_schema.json"
                if method == "B-L1-transfer"
                else "evidence/feature_schema.json"
            )
            check(files[base + weight]["sha256"] == original["weights_sha256"], "weight SHA mismatch")
            check(
                files[base + "model.json"]["sha256"] == original["model_index_sha256"], "model SHA mismatch"
            )
            model_specs[method] = {
                **original,
                "feature_schema_sha256": files[feature]["sha256"],
                "preprocessing_sha256": files[base + pre]["sha256"],
            }
        prof = {
            "schema": "el-bundle-profile-1",
            "id": "v0.3.0-rc.1/original-thresholds",
            "release_index_sha256": INDEX,
            "release_zip_sha256": CPU,
            "method_sha256": release["method_sha256"],
            "default": release["default_method"],
            "models": model_specs,
            "calibration_profile": None,
        }
        put(out / "assets/profile.json", prof)
        for name in ("model.json", "preprocessing.json", "coefficients.json"):
            save(out / "assets/MS-L1" / name, fetch(cpu, files, "models/MS-L1/" + name))
        # Actual deployment inputs alone. Ground truth and expected scores are never extracted to these cases.
        example = json.loads(ex.read("example.json"))
        origins = {}
        for cond in ("M2", "M4"):
            prefix = "CASE/" + cond + "/"
            original = fetch(ex, example["files"], prefix + "case.json")
            case = json.loads(original)
            objects = [case[k] for k in ("source_cameras", "render_cameras", "prediction")]
            objects += list(case["source_images"].values()) + list(case["renders"].values())
            for record in objects:
                data = fetch(ex, example["files"], prefix + record["path"])
                check(sha(data) == record["sha256"] and len(data) == record["bytes"], "case member differs")
                save(out / "cases" / cond / record["path"], data)
            save(out / "cases" / cond / "case.json", original)
            origins[cond] = sha(original)
        put(
            out / "original.json",
            {
                "schema": "el-bundle-original-assets-1",
                "example_zip_sha256": EXAMPLE,
                "cpu_zip_sha256": CPU,
                "release_index_sha256": INDEX,
                "cases": origins,
                "profile_sha256": sha((out / "assets/profile.json").read_bytes()),
                "model_asset_credit": "wany614",
                "linear_risk_license": "Apache-2.0",
                "example_media_license": "CC-BY-4.0",
                "release_commit": release["commit"],
            },
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("cpu", "example", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    restore(a.cpu, a.example, a.out)
