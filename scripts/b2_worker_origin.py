"""Expose only the two original source views for a development selector and register the optimizer rule."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def load(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def check(value, message):
    if not value:
        raise ValueError(message)


def selector(cfg, inputs, out):
    import numpy as np
    import torch

    roles = {
        "checkpoint": "checkpoint.pt",
        "recipe": "recipe.json",
        "cameras": "source_cameras.json",
        "s00": "sources/s00.png",
        "s03": "sources/s03.png",
        "r00": "source-renders/s00.npz",
        "r03": "source-renders/s03.npz",
    }
    check(inputs.keys() == roles.keys() and cfg["object_id"] in ("set1", "set6"), "Unapproved selector files")
    for role, p in inputs.items():
        check(digest(p) == cfg["sources"][role]["sha256"], "Changed source role " + role)
    cams = load(inputs["cameras"])
    checkpoint = torch.load(inputs["checkpoint"], map_location="cpu", weights_only=True)
    recipe = load(inputs["recipe"])
    check(
        set(cams["views"]) == set(checkpoint["source_calibration"]) == {"s00", "s03"}
        and checkpoint["identity"]["condition"] == "M2"
        and checkpoint["identity"]["object_id"] == cfg["object_id"]
        and checkpoint["producer_run_id"] == cfg["sources"]["checkpoint"]["run_id"]
        and checkpoint["step"] == 5000
        and checkpoint["recipe"] == recipe
        and recipe["seed"] == 43171
        and checkpoint["identity"]["source_camera_sha256"] == digest(inputs["cameras"])
        and checkpoint["identity"]["source_rgb_sha256"] == {k: digest(inputs[k]) for k in ("s00", "s03")},
        "Checkpoint is not the selected two-source M2 origin",
    )
    for name in ("r00", "r03"):
        with np.load(inputs[name], allow_pickle=False) as frame:
            check(
                set(frame.files) == {"rgb", "z", "alpha"}
                and frame["rgb"].shape == (256, 256, 3)
                and frame["alpha"].shape == frame["z"].shape == (256, 256),
                "Malformed original self render",
            )
    base = out / "selector"
    docs = {}
    for role, name in roles.items():
        dst = base / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inputs[role], dst)
        docs[role] = {
            "path": name,
            "bytes": dst.stat().st_size,
            "sha256": digest(dst),
            "source": cfg["sources"][role],
        }
    (out / "selector-input.json").write_bytes(
        encoded(
            {
                "schema": "el-b2-selector-input-1",
                "object_id": cfg["object_id"],
                "run_id": os.environ["EL_RUN_ID"],
                "condition": "M2",
                "source_view_ids": {"s00": 24, "s03": 22},
                "checkpoint_sha256": digest(inputs["checkpoint"]),
                "files": docs,
                "candidate_RGB_opened": 0,
                "evaluation_poses_in_selector": 0,
                "evaluation_photographs_opened": 0,
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("selector",))
    for key in ("config", "out"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--input", action="append", nargs=2, required=True)
    a = p.parse_args()
    selector(load(a.config), {k: Path(v) for k, v in a.input}, a.out)
