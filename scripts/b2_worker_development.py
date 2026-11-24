"""Materialize the registered four sources for one exposed object under its own access authority."""

import argparse
import os
from pathlib import Path
import shutil
import struct

from t0_worker_pack import check, digest, encoded, load, spec

SLOTS = ("s00", "s01", "s02", "s03")
LOWER = ("s00", "s03")
ALLOWED = (7, 10, 13, 17, 20, 23, 31, 40)
TARGETS = tuple(f"v{i:02}" for i in range(1, 7))


def checked_member(net, archive, row):
    from t0_dtu_archive import ExactArchive

    reader = ExactArchive(net, archive["url"], archive)
    head = reader.exact(row["offset"], 30)
    check(head[:4] == b"PK\x03\x04", "Wrong named candidate ZIP header")
    flags = struct.unpack_from("<H", head, 6)[0]
    return reader.member({**row, "flags": flags})


def selected_camera(matrix, crop, metric):
    import numpy as np
    from evidence_lens.dtu_camera import decompose

    k, pose = decompose(np.loadtxt(matrix), metric["metres_per_native_unit"], 0)
    k = np.asarray(crop["affine"]) @ k
    check(abs(k[0, 1]) <= 0.003 and abs(k[1, 0]) <= 1e-9, "Unsupported skew in selected native calibration")
    k[0, 1] = k[1, 0] = 0
    return {
        "width": 256,
        "height": 256,
        "K": k.tolist(),
        "c2w": (np.asarray(metric["dtu_metres_to_canonical_metres"]) @ pose).tolist(),
        "clip_m": [0.05, 30.0],
    }


def supply(cfg, roles, out):
    from evidence_lens.dtu_camera import convert_rgb
    from dtu_sample import Network

    study = load(roles["study"])
    obj = cfg["object_id"]
    mode = cfg["layout"]
    auth = load(roles["authority"])
    check(
        obj in ("set1", "set6")
        and study["base_recipe_sha256"] == auth["base_recipe_sha256"]
        and mode in ("M4-fixed", "M4-selected"),
        "Out of development scope",
    )
    net = None
    selection = None
    if mode == "M4-selected":
        decision = load(roles["decision"])
        selection = cfg["sources"]["decision"]
        sample = load(roles["start"])
        ids = decision["selected_official_ids"]
        check(
            decision["schema"] == "el-b2-nbv-selection-1"
            and decision["object_id"] == sample["object_id"] == obj
            and decision["strategy_id"] == cfg["strategy_id"]
            and decision["strategy_id"] in study["development_selection"]["strategies"]
            and decision["pool"] == list(ALLOWED)
            and type(ids) == list
            and len(ids) == 2
            and all(type(i) is int for i in ids)
            and len(set(ids)) == 2
            and set(ids) <= set(ALLOWED)
            and decision["m2_checkpoint_sha256"] == sample["checkpoint_sha256"]
            and digest(roles["decision"]) == selection["sha256"],
            "The two private candidates must be independently sealed together before access",
        )
        numbers = dict(zip(SLOTS, [24, *sorted(ids), 22], strict=True))
    else:
        numbers = study["calibration"]["fixed_M4"]
    old = load(roles["prior"])
    sc = load(roles["cameras"])
    rc = load(roles["rendercameras"])
    check(
        old["object_id"] == obj
        and old["condition"] == ("M2" if mode == "M4-selected" else "M4")
        and set(sc["views"]) == set(LOWER if mode == "M4-selected" else SLOTS)
        and set(rc["views"]) == set(sc["views"]) | set(TARGETS)
        and all(sc["views"][s] == rc["views"][s] for s in sc["views"]),
        "Historic origin differs",
    )
    records = {}
    (out / "sources").mkdir()
    for sid in LOWER if mode == "M4-selected" else SLOTS:
        oldimage = old["source_images"][sid]
        check(
            oldimage["official_view_id"] == numbers[sid] and digest(roles[sid]) == oldimage["rgb"]["sha256"],
            "Wrong historic observation",
        )
        f = out / "sources" / (sid + ".png")
        shutil.copyfile(roles[sid], f)
        records[sid] = {
            **oldimage,
            "acquisition": "same frozen source",
            "rgb": {"path": "sources/" + sid + ".png", **spec(f)},
        }
    if mode == "M4-selected":
        directory = {x["name"]: x for x in load(roles["directory"])}
        archive = load(roles["archive"])
        crop = load(roles["crop"])
        metric = load(roles["metric"])
        check(digest(roles["crop"]) == study["calibration"]["crop_sha256"], "Changed registered crop")
        net = Network(32 * 1024**2, out / "http.jsonl")
        for sid in ("s01", "s02"):
            n = numbers[sid]
            name = f"SampleSet/MVS Data/Rectified/scan{int(obj[3:])}/rect_{n:03d}_3_r5000.png"
            row = directory[name]
            check(row["name"] == name, "Wrong candidate path")
            img = checked_member(net, archive, row)
            raw = out / "original" / (sid + ".png")
            raw.parent.mkdir(exist_ok=True)
            raw.write_bytes(img)
            f = out / "sources" / (sid + ".png")
            f.write_bytes(convert_rgb(img, crop))
            sc["views"][sid] = rc["views"][sid] = selected_camera(roles["projection" + sid], crop, metric)
            records[sid] = {
                "official_view_id": n,
                "original_bytes": len(img),
                "original_crc32": row["crc32"],
                "original_sha256": digest(raw),
                "acquisition": "authorized named HTTP ranges",
                "rgb": {"path": "sources/" + sid + ".png", **spec(f)},
            }
    if mode == "M4-fixed":
        for key, name in [("cameras", "source_cameras.json"), ("rendercameras", "render_cameras.json")]:
            shutil.copyfile(roles[key], out / name)
    else:
        (out / "source_cameras.json").write_bytes(encoded(sc))
        (out / "render_cameras.json").write_bytes(encoded(rc))
    (out / "source-provenance.json").write_bytes(
        encoded(
            {
                "schema": "el-b2-development-source-1",
                "run_id": os.environ["EL_RUN_ID"],
                "object_id": obj,
                "condition": "M4",
                "layout": mode,
                "sources": records,
                "source_camera_sha256": digest(out / "source_cameras.json"),
                "render_camera_sha256": digest(out / "render_cameras.json"),
                "official_by_slot": numbers,
                "selection": selection,
                "S0": {sid: numbers[sid] for sid in LOWER},
                "reference_photo_reads": 0,
                "other_branch_candidate_RGB_reads": 0,
                "candidate_http_response_bytes": net.total if net else 0,
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--input", action="append", nargs=2, required=True)
    a = p.parse_args()
    supply(load(a.config), {k: Path(f) for k, f in a.input}, a.out)
