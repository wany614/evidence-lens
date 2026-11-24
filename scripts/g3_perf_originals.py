"""Register original performance inputs using only released source images and color/depth/alpha predictions."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from b2_asset_fixture import camera_rig, native_values, source_picture
from evidence_lens_bundle import common
from evidence_lens_bundle.bundle import Bundle
from evidence_lens_bundle.common import encoded, json_bytes, member, parts, require, row, sha, write

ORIGINS = {
    "beads-M2": "d5b187fbd93eea875508d54d0a2ca67828c7a3d44ed07b13913ed805020a7ea5",
    "bell-M2": "2561eb1e68bb2a9567d722c3097ba25aa9ef85aeeda627d0ae3c0d2162c917aa",
    "bell-M4": "b8feb75ade71f39336f7927ebbb814e828b345932b4f651e0e1cb81137e109c2",
}


def prepare_ingress(beads, bell, project, output):
    """Copy only the deployment manifest's declared members from the three pinned bundles."""
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as z:
        for name, folder in (("beads-M2", beads), ("bell-M2", bell / "M2"), ("bell-M4", bell / "M4")):
            b = Bundle(folder, ORIGINS[name])
            recs = [b.meta["camera"], b.meta["producer"], *b.meta["sources"].values()]
            recs.extend(f["file"] for f in b.producer["frames"].values() if f["status"] == "ready")
            z.writestr(name + "/bundle.json", (folder / "bundle.json").read_bytes())
            for spec in recs:
                z.writestr(name + "/" + spec["path"], member(folder, spec))
        for k in ("profile.json", "MS-L1/model.json", "MS-L1/coefficients.json", "MS-L1/preprocessing.json"):
            z.write(project / "src/evidence_lens_bundle/assets" / k, "assets/" + k)


def build(archive, out):
    require(os.environ.get("EL_RUN_ID"), "record the engine input registration")
    out.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with ZipFile(archive) as z:
            for n in z.namelist():
                parts(n)
                z.extract(n, root)
        common.ASSETS = root / "assets"
        originals = {s: Bundle(root / s, origin) for s, origin in ORIGINS.items()}
        for b in originals.values():
            b.validate()
        b = originals["beads-M2"]
        dest = root / "beads-M4"
        shutil.copytree(b.root, dest)
        cameras = json_bytes(member(b.root, b.meta["camera"]))
        positions, _ = camera_rig()
        native = native_values()
        order = ["41", "70", "109", "5"]
        photos = dict(b.meta["sources"])
        duration = {}
        for s in ("70", "109"):
            cam = cameras["views"][s]
            require(
                np.allclose(positions[s]["K"], cam["K"], atol=1e-10)
                and np.allclose(positions[s]["c2w"], cam["c2w"], atol=1e-10),
                "authored source camera differs",
            )
            name = f"sources/new-{s}.png"
            start = time.perf_counter()
            source_picture(positions[s], native, dest / name)
            duration[s] = time.perf_counter() - start
            photos[s] = row(name, (dest / name).read_bytes())
        render_map = dict(b.producer["frames"])
        producer = {
            **b.producer,
            "source_order": order,
            "target_order": ["13"],
            "frames": render_map,
            "source_rgb_sha256": {s: v["sha256"] for s, v in photos.items()},
        }
        write(dest / "producer.json", producer)
        op = {
            "original_bundle": b.sha256,
            "source_order": order,
            "target_order": ["13"],
            "render_roles": "Previously rendered cameras 70 and 109 now have new legal analytic input photographs; held target 13 has no source photograph.",
        }
        write(
            dest / "bundle.json",
            {
                **b.meta,
                "condition": "M4",
                "sources": photos,
                "source_order": order,
                "target_order": ["13"],
                "producer": row("producer.json", (dest / "producer.json").read_bytes()),
                "creation": {
                    "run_id": os.environ["EL_RUN_ID"],
                    "code_sha256": sha(Path(__file__).read_bytes()),
                    "config_sha256": sha(encoded(op)),
                    "implementation": "original engineering four-source evidence admission",
                    "original_case_sha256": None,
                },
            },
        )
        b4 = Bundle(dest)
        b4.validate()
        workload = {**originals, "beads-M4": b4}
        idx = {
            "schema": "el-g3-originals-1",
            "run_id": os.environ["EL_RUN_ID"],
            "sequences": [
                {
                    "sequence_id": name,
                    "bundle_sha256": obj.sha256,
                    "map_sha256": obj.producer["checkpoint_sha256"],
                    "actual_target_ids": list(obj.targets),
                    "source_order": list(obj.order),
                }
                for name, obj in workload.items()
            ],
            "new_source_seconds": duration,
            "four_source_beads": op,
            "source_truth_for_any_scored_target": False,
            "risk_updates": 0,
        }
        write(root / "workloads.json", idx)
        with ZipFile(out / "workloads.zip", "x", compression=ZIP_DEFLATED) as z:
            z.write(root / "workloads.json", "workloads.json")
            for name in workload:
                for f in sorted((root / name).rglob("*")):
                    if f.is_file():
                        z.write(f, f.relative_to(root).as_posix())
        write(
            out / "workloads.json",
            {**idx, "archive": row("workloads.zip", (out / "workloads.zip").read_bytes())},
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("ingress", "build"))
    for n in ("beads", "bell", "project", "archive", "out"):
        p.add_argument("--" + n, type=Path)
    a = p.parse_args()
    if a.action == "ingress":
        prepare_ingress(a.beads, a.bell, a.project, a.archive)
    else:
        build(a.archive, a.out)
