"""Read only the preregistered ETH3D PINHOLE calibration headers, then open an allowlisted photo role."""

import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile

import numpy as np


def roster(archive, registration, scene, out):
    (row,) = [r for r in registration["external"]["roster"] if r["id"] == scene]
    names = subprocess.check_output(["bsdtar", "-tf", str(archive)], text=True).splitlines()
    if any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts or "\\" in n for n in names):
        raise ValueError("unsafe archive member")
    members = {n for n in names if not n.endswith("/")}
    if len(members) != len([n for n in names if not n.endswith("/")]):
        raise ValueError("duplicate archive member")
    files = {}
    for basename in ("cameras.txt", "images.txt"):
        matches = [n for n in members if PurePosixPath(n).name == basename]
        if len(matches) != 1:
            raise ValueError(f"expected one official {basename}: {matches}")
        files[basename] = matches[0]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(tmp), *files.values()], check=True)
        intr = {}
        for text in (tmp / files["cameras.txt"]).read_text().splitlines():
            if not text.strip() or text.startswith("#"):
                continue
            cid, kind, width, height, fx, fy, cx, cy = text.split()
            if kind != "PINHOLE" or cid in intr:
                raise ValueError("ETH3D candidate is not the official undistorted pinhole release")
            intr[cid] = {
                "width": int(width),
                "height": int(height),
                "distortion": "rectified",
                "K": [[float(fx), 0, float(cx) - 0.5], [0, float(fy), float(cy) - 0.5], [0, 0, 1]],
                "official_corner_principal_point": [float(cx), float(cy)],
            }
        poses = []
        with (tmp / files["images.txt"]).open("rb") as stream:
            for line in stream:
                if line.startswith(b"#") or not line.strip():
                    continue
                image_id, qw, qx, qy, qz, tx, ty, tz, cid, name = line.decode().strip().split(maxsplit=9)
                skipped = (
                    stream.readline()
                )  # Discard the alternating observation record without parsing any feature or track.
                if skipped == b"":
                    raise ValueError("missing separator/observation record")
                q = np.array([qw, qx, qy, qz], np.float64)
                if not np.isclose(np.linalg.norm(q), 1, atol=1e-5):
                    raise ValueError("wrong camera quaternion")
                w, x, y, z = q / np.linalg.norm(q)
                rot = np.array(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    ]
                )
                c2w = np.eye(4)
                c2w[:3, :3] = rot.T
                c2w[:3, 3] = -rot.T @ np.array([tx, ty, tz], np.float64)
                jpg = [n for n in members if n.endswith("/" + name) or n == name]
                if len(jpg) != 1 or not name.lower().endswith((".jpg", ".jpeg")):
                    raise ValueError("ambiguous official original for image pose: " + name)
                poses.append(
                    (
                        name,
                        {
                            **intr[cid],
                            "official_image_id": image_id,
                            "official_camera_id": cid,
                            "original_name": name,
                            "member": jpg[0],
                            "c2w": c2w.tolist(),
                        },
                    )
                )
    if len(poses) != row["official_total"] or len({n for n, _ in poses}) != len(poses):
        raise ValueError("official camera/photo count differs from registered scene")
    positions = {((2 * j + 1) * len(poses)) // 20: j for j in range(10)}
    views = {}
    source_count = 0
    for idx, (_, entry) in enumerate(sorted(poses, key=lambda x: x[0].encode())):
        if idx in positions:
            name = f"h{positions[idx]:02d}"
        else:
            name = f"r{source_count:03d}"
            source_count += 1
        views[name] = entry
    if source_count != row["R"]:
        raise ValueError("different frozen R/H sizes")
    out.write_text(
        json.dumps(
            {
                "schema": "el-many-eth-role-roster-1",
                "scene_id": scene,
                "split": row["split"],
                "views": views,
                "R": sorted(k for k in views if k.startswith("r")),
                "H": sorted(k for k in views if k.startswith("h")),
                "unit": "scene_units",
                "camera_header_members": files,
                "archive_metadata_members": len(members),
                "photo_pixel_decodes": 0,
                "H_photo_members_opened": 0,
                "points3D_members_opened": 0,
                "keypoint_or_track_rows_parsed": 0,
                "discarded_observation_rows": len(poses),
                "official_format": "https://www.eth3d.net/documentation",
                "license": registration["external"]["license"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def extract_role(archive, roster, ids, role, destination):
    expected = set(roster[role])
    if not ids or set(ids) != expected or set(ids) & set(roster["H" if role == "R" else "R"]):
        raise ValueError("photo role extraction differs from frozen manifest")
    members = [roster["views"][i]["member"] for i in ids]
    subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(destination), *members], check=True)
    real = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
    if {PurePosixPath(n).as_posix() for n in members} != real or any(
        p.is_symlink() for p in destination.rglob("*")
    ):
        raise ValueError("unexpected actual image extraction set")
    return {i: destination / roster["views"][i]["member"] for i in ids}
