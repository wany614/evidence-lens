"""Create a licensed deployment-only M4 replay with separately sealed expected risk arrays."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from bundle_original import EXAMPLE, check, put, sha


def build(a):
    source = Path(__file__).with_name("bundle_grade.py")
    bodies = {"grade.py": source.read_bytes(), a.wheel.name: a.wheel.read_bytes()}
    manifest = json.loads((a.bundle / "bundle.json").read_bytes())
    producer = json.loads((a.bundle / "producer.json").read_bytes())
    names = [
        "bundle.json",
        "cameras.json",
        "producer.json",
        *[v["path"] for v in manifest["sources"].values()],
        *[v["file"]["path"] for v in producer["frames"].values() if v["status"] == "ready"],
    ]
    for n in names:
        bodies["bundle/" + n] = (a.bundle / n).read_bytes()
    with a.example.open("rb") as f:
        check(hashlib.file_digest(f, "sha256").hexdigest() == EXAMPLE, "wrong original example ZIP")
    with ZipFile(a.example) as z:
        meta = json.loads(z.read("example.json"))

        def original(n):
            raw = z.read(n)
            spec = meta["files"][n]
            check(len(raw) == spec["bytes"] and sha(raw) == spec["sha256"], "original differs: " + n)
            return raw

        pfx = "EXPECTED/MS-L1/"
        orig = json.loads(original(pfx + "native-scores.json"))
        result = {}
        for v in orig["views"]:
            if v["condition"] != "M4":
                continue
            name = v["view"]
            check(v["case_sha256"] == manifest["creation"]["original_case_sha256"], "wrong original Case")
            spec = v["modes"]["all"]["file"]
            raw = original(pfx + spec["path"])
            check(sha(raw) == spec["sha256"], "wrong original score")
            dest = f"{name}.npz"
            bodies["expected/" + dest] = raw
            result[name] = {"path": dest, "sha256": sha(raw), "bytes": len(raw)}
        check(list(result) == manifest["target_order"], "wrong public view order")
        expected = {
            "schema": "el-bundle-expected-risk-1",
            "method": "MS-L1",
            "condition": "M4",
            "view_order": list(result),
            "views": result,
            "bundle_sha256": sha(bodies["bundle/bundle.json"]),
            "model_index_sha256": orig["model_index_sha256"],
            "threshold": orig["threshold"],
            "source_example_zip_sha256": EXAMPLE,
            "original_case_sha256": manifest["creation"]["original_case_sha256"],
        }
        bodies["expected/expected.json"] = (json.dumps(expected, indent=2, sort_keys=True) + "\n").encode()
        for n in ("LICENSE", "NOTICE", "licenses/CC-BY-4.0.txt", "licenses/multisource-assets.json"):
            bodies[n] = z.read(n)
    bodies["README.txt"] = f"""Evidence Lens • original offset bell cage (wany614)
Public original scene photographs and renders: CC BY 4.0. Tool and linear model: Apache 2.0.
The numerical reference here contains the six previously published MS-L1 score arrays. No target truth is included.
Source release: https://github.com/wany614/evidence-lens/releases/tag/v0.3.0-rc.1

In a new Python 3.11+ environment from the extracted directory:
python -m pip install ./{a.wheel.name}
evidence-lens validate --bundle bundle --bundle-sha256 {expected["bundle_sha256"]}
evidence-lens score --bundle bundle --model MS-L1 --out scores
evidence-lens report --scores scores --out report.html
python grade.py --scores scores --expected expected --out grade.json
Open report.html locally. Keep scores and report paths new on a replay.
SHA256SUMS covers the original archive members. Python, NumPy and Pillow are needed for CPU deployment.
""".encode()
    bodies["SHA256SUMS"] = "".join(f"{sha(bodies[n])}  {n}\n" for n in sorted(bodies)).encode()
    path = a.out / "evidence-lens-bundle-v1-offset-bell-cage.zip"
    with ZipFile(path, "x") as z:
        for n, b in sorted(bodies.items()):
            entry = ZipInfo(n, date_time=(2000, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            z.writestr(entry, b)
    with ZipFile(path) as z:
        check(
            z.testzip() is None
            and set(z.namelist()) == set(bodies)
            and all(z.read(n) == b for n, b in bodies.items()),
            "archive readback failed",
        )
    put(
        a.out / "portable.json",
        {
            "schema": "el-bundle-portable-demo-1",
            "archive": {"name": path.name, "bytes": path.stat().st_size, "sha256": sha(path.read_bytes())},
            "bundle_sha256": expected["bundle_sha256"],
            "wheel_sha256": sha(bodies[a.wheel.name]),
            "files": {n: {"bytes": len(b), "sha256": sha(b)} for n, b in sorted(bodies.items())},
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("wheel", "bundle", "example", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    build(p.parse_args())
