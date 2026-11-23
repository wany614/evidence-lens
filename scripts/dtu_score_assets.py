"""Restore the frozen CPU package and only the permitted 057 regression inputs from their published ZIPs."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

CPU = "f555956840870309217cc85fe9b339ef53762679bbd2138b7f149d8ea451a7b7"
EXAMPLE = "14204986e0d598adc613dd1a9ef2e5c1584f5be001d3541531d5d9e709707548"
EXAMPLE_INDEX = "b08efe193b2e93dd6e0f9e35360130745cb64295868bfc5b8edd30dc19ae67b8"
from dtu_score import METHODS, SCOPES, VIEWS, RELEASE_SHA, digest, require


def sha(b):
    return hashlib.sha256(b).hexdigest()


def artifact(folder, p):
    return {"path": p.relative_to(folder).as_posix(), "sha256": digest(p), "bytes": p.stat().st_size}


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def extract(archive, name, spec, folder, destination):
    rel = PurePosixPath(destination)
    require(
        not rel.is_absolute() and not any(n in ("", ".", "..") for n in rel.parts), "unsafe archive member"
    )
    b = archive.read(name)
    require(len(b) == spec["bytes"] and sha(b) == spec["sha256"], "member differs from signed index: " + name)
    p = folder / destination
    if p.exists():
        require(digest(p) == sha(b), "inconsistent repeat")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        f.write(b)


def restore(cpu_zip, example_zip, out):
    require(digest(cpu_zip) == CPU and digest(example_zip) == EXAMPLE, "unexpected distributed archives")
    require(
        not out.exists() or (out.is_dir() and not out.is_symlink() and not any(out.iterdir())),
        "nonempty output",
    )
    out.mkdir(parents=True, exist_ok=True)
    archive_bytes = {}
    for role, path, index, checksum in [
        ("cpu", cpu_zip, "release.json", RELEASE_SHA),
        ("example", example_zip, "example.json", EXAMPLE_INDEX),
    ]:
        with ZipFile(path) as z:
            names = z.namelist()
            require(len(names) == len(set(names)), "duplicate release member")
            b = z.read(index)
            require(sha(b) == checksum, "unexpected published index")
            book = json.loads(b)
            archive_bytes[role] = path.stat().st_size
            if role == "cpu":
                root = out / "cpu"
                root.mkdir()
                (root / index).write_bytes(b)
                for name, spec in book["files"].items():
                    extract(z, name, spec, root, name)
                continue

            def put(name, dest):
                extract(z, name, book["files"][name], out, dest)

            for c in ("M2", "M4"):
                prefix = "CASE/" + c + "/"
                m = json.loads(z.read(prefix + "case.json"))
                require(
                    m["condition"] == c and m["target_ids"] == list(VIEWS), "different public camera case"
                )
                names = {
                    "case.json",
                    *(m[k]["path"] for k in ("prediction", "source_cameras", "render_cameras")),
                    *(d["path"] for d in m["source_images"].values()),
                    *(d["path"] for d in m["renders"].values()),
                }
                for name in sorted(names):
                    put(prefix + name, "cases/" + c + "/" + name)
            metadata = json.loads(z.read("EXPECTED/expected-example.json"))
            regression = {
                "schema": "el-dtu-057-regression-roster-1",
                "family": book["family"],
                "method_sha256": book["method_sha256"],
                "release_sha256": RELEASE_SHA,
                "cases": {},
            }
            refdir = out / "reference"
            refdir.mkdir()
            for v in VIEWS:
                for ext in ("png", "npz"):
                    put("REFERENCE/reference/" + v + "." + ext, "reference/" + v + "." + ext)
            for c in ("M2", "M4"):
                byview = {}
                for v in VIEWS:
                    measured = next(k for k in metadata["cases"] if (k["view"], k["condition"]) == (v, c))
                    err = "EXPECTED/" + measured["error"]["path"]
                    ef = "measurements/" + c + "/" + v + ".npz"
                    put(err, ef)
                    scores = {}
                    for m in METHODS:
                        native = json.loads(z.read("EXPECTED/" + m + "/native-scores.json"))
                        entry = next(x for x in native["views"] if (x["condition"], x["view"]) == (c, v))[
                            "modes"
                        ][SCOPES[m]]["file"]
                        dst = "gold/" + c + "/" + v + "/" + m + ".npz"
                        put("EXPECTED/" + m + "/" + entry["path"], dst)
                        scores[m] = {"scope": SCOPES[m], "file": artifact(out, out / dst)}
                    byview[v] = {
                        "scores": scores,
                        "measurements": artifact(out, out / ef),
                        "counts": {k: measured[k] for k in ("V", "E", "Y")},
                        "rgb_png": artifact(out, refdir / (v + ".png")),
                        "original_fp32": artifact(out, refdir / (v + ".npz")),
                    }
                regression["cases"][c] = {
                    "case_sha256": digest(out / "cases" / c / "case.json"),
                    "views": byview,
                }
            write(out / "regression.json", regression)
    write(
        out / "release-assets.json",
        {
            "schema": "el-dtu-frozen-asset-import-1",
            "cpu_zip_sha256": CPU,
            "example_zip_sha256": EXAMPLE,
            "release_index_sha256": RELEASE_SHA,
            "example_index_sha256": EXAMPLE_INDEX,
            "archive_bytes": archive_bytes,
            "regression": artifact(out, out / "regression.json"),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("cpu-zip", "example-zip", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    restore(a.cpu_zip, a.example_zip, a.output)
