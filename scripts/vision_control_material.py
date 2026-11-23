"""Export the disjoint inference and grading inputs for the native textured-plane control."""
import argparse
import shutil
from pathlib import Path

import numpy as np

from evidence_lens._bundle import load, save, sha
from evidence_lens.formal_inputs import require
from evidence_lens.scoring_io import ReadAudit, TARGETS, VIEWS, load_cameras, load_source_png
from evidence_lens.labels import ReferenceBundle


def signature(path):
    return {"bytes": path.stat().st_size, "sha256": sha(path)}


def export_case(args, config):
    source, out = args.source, args.output / "CASE"
    provenance = load(source / "provenance.json")
    meta = load(source / "predictions/prediction.json")
    require(provenance["run_id"] == meta["producer_run_id"] == args.producer
            and provenance["source_png_sha256"] == config["source_rgb_sha256"]
            and provenance["source_cameras_sha256"] == config["source_cameras_sha256"]
            and provenance["source_stride"] == config["resolution_stride"] == 1
            and provenance["hypothesized_plane_m"] == config["plane_z_m"] == 2.0
            and meta["adapter"] == config["adapter"], "control adapter or sources differ")
    cameras = load_cameras(source / "cameras.json", ReadAudit())
    require(all(camera.shape == (512, 512) for camera in cameras.values()), "expected native 512-square cameras")
    names = ["input.png", "cameras.json", "predictions/prediction.json",
             *["predictions/" + view + ".npz" for view in VIEWS]]
    require(len(names) == 10 and set(meta["views"]) == set(VIEWS), "incomplete control inputs")
    out.mkdir(parents=True)
    (out / "predictions").mkdir()
    for name in names:
        shutil.copyfile(source / name, out / name)
    members = {n: signature(out / n) for n in names}
    require(members["input.png"]["sha256"] == meta["input_sha256"]
            and members["cameras.json"]["sha256"] == meta["camera_sha256"]
            and all(members["predictions/" + v + ".npz"] == meta["views"][v] for v in VIEWS),
            "exported inference bytes differ from the recorded plane")
    save(out / "case.json", {"schema": "el-vision-public-adapter-case-1", "family": config["scene_family_id"],
         "condition": config["condition_id"], "generator": "authored-two-metre-plane",
         "width": 512, "height": 512, "files": members})
    save(args.output / "case-evidence.json", {"config_sha256": sha(args.config), "plane_run_id": args.producer,
         "reference_run_id": config["reference_run_id"], "reference_manifest_sha256": config["reference_manifest_sha256"],
         "original_source_sha256": config["source_rgb_sha256"], "original_cameras_sha256": config["source_cameras_sha256"],
         "plane_provenance_sha256": sha(source / "provenance.json"), "depth_probe_sha256": sha(source / "predictions/depth-probe.json"),
         "case_sha256": sha(out / "case.json"), "inference_files": members})


def export_reference(args, config):
    source, out = args.source, args.output / "REFERENCE"
    require(args.producer == config["reference_run_id"] and sha(source / "cameras.json") == config["source_cameras_sha256"],
            "control reference differs from the registered source")
    audit = ReadAudit()
    cameras = load_cameras(source / "cameras.json", audit)
    load_source_png(source / "input.png", cameras["source"], audit)
    reference = ReferenceBundle(source, audit)
    require(reference.meta["scene_family_id"] == config["scene_family_id"], "unrelated reference family")
    out.mkdir(parents=True)
    sources = {}
    for view in TARGETS:
        truth = reference.read(view, cameras[view])
        surface = np.isfinite(truth["z"]) & (truth["z"] > 0) & np.isfinite(truth["position"]).all(axis=-1)
        np.savez_compressed(out / (view + ".npz"), rgb=truth["rgb"], z=truth["z"], surface=surface)
        sources[view] = reference.meta["views"][view]["reference_sha256"]
    save(out / "reference.json", {"schema": "el-vision-public-control-reference-1", "family": config["scene_family_id"],
         "reference_run_id": args.producer, "reference_manifest_sha256": config["reference_manifest_sha256"],
         "original_reference_npz_sha256": sources, "original_cameras_sha256": config["source_cameras_sha256"],
         "files": {v + ".npz": signature(out / (v + ".npz")) for v in TARGETS}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("case", "reference"))
    for name in ("config", "source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--producer", required=True)
    arguments = parser.parse_args()
    config = load(arguments.config)
    require(config["schema"] == "el-public-native-plane-control-1", "unexpected public control registration")
    if arguments.action == "case":
        export_case(arguments, config)
    else:
        export_reference(arguments, config)
