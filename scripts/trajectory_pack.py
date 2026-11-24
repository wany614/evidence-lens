"""Make a portable six-sequence render ticket from two original, legally reusable development scenes."""
from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from evidence_lens_bundle.common import encoded, json_bytes, require, row, sha, write
from b2_asset_fixture import camera_rig, native_values, save_assets, source_picture


RUNTIME = {
    "evidence_lens": ("__init__.py", "multisource_io.py", "multisource.py", "features.py", "scoring_io.py"),
    "evidence_lens_asset": ("__init__.py", "gaussian.py", "colmap.py", "render.py", "prepare.py"),
    "evidence_lens_bundle": ("__init__.py", "bundle.py", "common.py", "assets/profile.json"),
}
BOUND_KEYS = ("width", "height", "K", "c2w", "projection", "distortion")
LAUNCHER = """import argparse, hashlib, json, sys
from pathlib import Path
root = Path(__file__).resolve().parent
ticket = json.loads((root / 'ticket.json').read_bytes())
for record in ticket['files']:
    path = root / record['path']
    if path.stat().st_size != record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('changed registered ticket member ' + record['path'])
sys.path.insert(0, str(root / 'runtime'))
from trajectory_render import execute
p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
execute(root, p.parse_args().out)
"""


def pose(center, subject, down_hint):
    forward = np.subtract(subject, center)
    forward /= np.linalg.norm(forward)
    right = np.cross(down_hint, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    out = np.eye(4)
    out[:3, :3] = np.column_stack((right, down, forward))
    out[:3, 3] = center
    return out.tolist()


def camera(template, center, subject, down):
    return {**{k: template[k] for k in BOUND_KEYS}, "c2w": pose(center, subject, down)}


def smooth_targets(scene, template):
    result = {}
    for i in range(120):
        angle = 2 * math.pi * i / 120
        co, si = math.cos(angle), math.sin(angle)
        if scene == "helical-beads":
            c = (0.42 * co, 0.14 * si, 0.045 * si)
            look, down = (0, 0, 2.4), (0, 1, 0)
        else:
            c = (0.44 * co, -1.92 + 0.075 * si, 0.86 + 0.11 * si)
            look, down = (0, 0, 0.43), (0, 0, -1)
        result[f"f{i:03d}"] = camera(template, c, look, down)
    return result


def make(a):
    require(os.environ.get("EL_RUN_ID"), "record ticket construction")
    project, case = a.project, a.bell
    assets = {}

    def add(name, data):
        if name in assets:
            require(assets[name] == data, "conflicting transport member")
        assets[name] = data
        return row(name, data)

    def sequence(name, scene, asset, scale, sources, cams, photos, origin, count=120):
        templates = smooth_targets(scene, cams[sources[0]])
        targets = list(templates)[:count]
        camera_doc = {
            "schema": "el-camera-1", "unit": "m", "pixel_center": "integer", "image_origin": "top_left",
            "axis_convention": "OpenCV x right, y down, z forward",
            "views": {**{v: cams[v] for v in sources}, **{v: templates[v] for v in targets}},
        }
        return {
            "id": name, "scene": scene, "asset": asset, "map_origin": origin, "meters_per_unit": scale,
            "gaussian_profile": "graphdeco-v1" if scale == .01 else "evidence-lens-npz-v1",
            "metric_evidence": "Original metric camera rig and authored scene coordinates",
            "source_order": sources, "target_order": targets,
            "camera": add(f"plans/{name}.json", encoded(camera_doc)), "photos": {s: photos[s] for s in sources},
        }

    for package, files in RUNTIME.items():
        for f in files:
            parent = a.resources if f.endswith(".json") else project
            add(f"runtime/{package}/{f}", (parent / "src" / package / f).read_bytes())
    add("runtime/trajectory_render.py", (project / "scripts/trajectory_render.py").read_bytes())
    add("run_gpu.py", LAUNCHER.encode())
    add("LICENSE", (a.resources / "LICENSE").read_bytes())
    add("CC-BY-4.0.txt", (a.resources / "licenses/CC-BY-4.0.txt").read_bytes())
    add("NOTICE", (a.resources / "NOTICE").read_bytes())
    add("ORIGINAL-EXAMPLE-NOTICE.txt",
        b"Original helical beads and offset bell cage, calibrated views, and trajectory renders: "
        b"wany614 / Evidence Lens, CC BY 4.0. Original software: Apache-2.0.\n")
    sequences = []
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        beads, _ = camera_rig()
        native = native_values()
        save_assets(temp, native)
        for name, x, y in (("inside-L", -0.11, 0.03), ("inside-R", .16, -.01), ("alternate-R", .21, .11)):
            beads[name] = camera(beads["41"], (x, y, .012), (0, 0, 2.4), (0, 1, 0))
        photos = {}
        for name in ("41", "5", "inside-L", "inside-R", "alternate-R"):
            source_picture(beads[name], native, temp / (name + ".png"))
            photos[name] = add("beads/sources/" + name + ".png", (temp / (name + ".png")).read_bytes())
        for tag, shape, order, n in (
            ("beads-M2", "sh3", ["41", "5"], 120),
            ("beads-M4", "sh3", ["41", "inside-L", "inside-R", "5"], 120),
            ("beads-map2-M4", "sh0", ["41", "inside-L", "inside-R", "5"], 24),
            ("beads-map2-source2-M4", "sh0", ["41", "inside-L", "alternate-R", "5"], 24),
        ):
            filename = "graphdeco-" + shape + ".ply"
            asset = add("beads/" + filename, (temp / filename).read_bytes())
            origin = {"run_id": os.environ["EL_RUN_ID"], "sha256": asset["sha256"], "asset": filename,
                      "recipe": "b2_asset_fixture.native_values / save_assets, seed 9312",
                      "source_code_sha256": sha((project / "scripts/b2_asset_fixture.py").read_bytes()),
                      "optimization_seconds": None, "reconstruction": "authored, no optimization"}
            sequences.append(sequence(tag, "helical-beads", asset, .01, order, beads, photos, origin, n))
    for condition, expected in (("M2", "4a08222890c42858c3d161208bc5c4d8347e019196c849e14777154d712de209"),
                                ("M4", "eedb19d170445f22273333b8c7f7e014c160fc606bc3f9b12f7f0ed0f84a7fac")):
        d = case / condition
        receipt = json_bytes((d / "reconstruction.json").read_bytes())
        raw = (d / "scene.npz").read_bytes()
        require(sha(raw) == receipt["checkpoint_sha256"] == expected, "original bell checkpoint differs")
        raw_cam = (d / "source_cameras.json").read_bytes()
        require(sha(raw_cam) == receipt["source_camera_sha256"], "bell source camera receipt differs")
        cams = {s: {k: c[k] for k in BOUND_KEYS} for s, c in json_bytes(raw_cam)["views"].items()}
        order = receipt["generator_source_ids"]
        photos = {}
        for s in order:
            raw_photo = (d / "sources" / (s + ".png")).read_bytes()
            require(sha(raw_photo) == receipt["source_rgb_sha256"][s], "bell source RGB receipt differs")
            photos[s] = add("bell/sources/" + s + ".png", raw_photo)
        asset = add("bell/" + condition + "/scene.npz", raw)
        origin = {k: receipt[k] for k in ("run_id", "checkpoint_sha256", "optimization_seconds", "stage_seconds",
                                            "gaussian_optimization_steps", "final_gaussian_count")}
        origin["reconstruction"] = "previously released development map"
        sequences.append(sequence("bell-" + condition, "offset-bell-cage", asset, 1., order, cams, photos, origin))
    ticket = {
        "schema": "el-static-trajectory-ticket-1", "run_id": os.environ["EL_RUN_ID"], "sequences": sequences,
        "files": [row(n, raw) for n, raw in sorted(assets.items())],
        "trajectory": "120 positions around a smooth closed elliptical camera path, endpoint excluded",
        "switch": "SH3 -> SH0 real asset, then replace one interior source with a new analytic observation",
        "allowed": "original development evidence only. Source PNGs only; predictions use gsplat 1.5.3.",
    }
    add("ticket.json", encoded(ticket))
    archive = a.out / "trajectory-gpu.zip"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED) as z:
        for n, raw in sorted(assets.items()):
            z.writestr(n, raw)
    write(a.out / "dispatch.json", {
        "schema": "el-static-trajectory-dispatch-1", "run_id": os.environ["EL_RUN_ID"],
        "artifact": row(archive.name, archive.read_bytes()), "ticket_sha256": sha(assets["ticket.json"]),
        "windows_command": "python -I -B run_gpu.py --out EMPTY_RECORDER_PAYLOAD",
        "expected_render_frames": sum(len(x["source_order"]) + len(x["target_order"]) for x in sequences),
    })


def inputs_archive(project, case, out):
    """Snapshot only source photographs, source cameras and published map weights."""
    records = {}
    for c, names in (("M2", ("s00", "s03")), ("M4", ("s00", "s01", "s02", "s03"))):
        for f in ("scene.npz", "reconstruction.json", "source_cameras.json", *("sources/" + s + ".png" for s in names)):
            records["bell/" + c + "/" + f] = case / c / f
    for f in ("LICENSE", "NOTICE", "licenses/CC-BY-4.0.txt", "src/evidence_lens_bundle/assets/profile.json"):
        records["resources/" + f] = project / f
    with ZipFile(out, "x", compression=ZIP_DEFLATED) as z:
        for n, path in sorted(records.items()):
            z.write(path, n)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--bell", type=Path)
    p.add_argument("--archive-inputs", type=Path)
    p.add_argument("--inputs-zip", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.archive_inputs:
        inputs_archive(args.project, args.bell, args.archive_inputs)
    elif args.inputs_zip:
        with tempfile.TemporaryDirectory() as tmp, ZipFile(args.inputs_zip) as z:
            for info in z.infolist():
                parts = Path(info.filename).parts
                require(not Path(info.filename).is_absolute() and ".." not in parts and parts[0] in ("bell", "resources"),
                        "unsafe original material archive")
                z.extract(info, tmp)
            args.bell, args.resources = Path(tmp) / "bell", Path(tmp) / "resources"
            make(args)
    else:
        args.resources = args.project
        make(args)
