"""Load and validate the frozen 24-family experiment without Blender or a model."""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


CATALOG = "protocol/families-v1.json"
DESIGNS = "configs/formal-scenes-v1.json"
CAMERAS = "configs/formal-cameras-v1.json"
BASE = "configs/reference.json"
STUDY = "protocol/study-v1.json"
LOCK = "configs/formal-freeze-v1.json"
TARGETS = [f"v{i:02}" for i in range(1, 7)]
REPRESENTATIVES = ["scene_001_tabletop", "scene_002_book_nook", "scene_003_pulley_frame",
                   "scene_006_fruit_crate", "scene_019_corner_booth"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value_hash(data):
    value = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def geometry_fingerprint(objects):
    # Disregard naming, ID, ordering and pigment when checking for duplicate layouts.
    parts = [{k: v for k, v in o.items() if k not in {"material", "name", "id"}} for o in objects]
    return value_hash(sorted(parts, key=value_hash))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(project):
    lock = read(project/LOCK)
    for relative, expected in lock["frozen_files"].items():
        require(digest(project/relative) == expected, f"Frozen input has changed: {relative}")
    require(lock["catalog_path"] == CATALOG and lock["catalog_sha256"] == digest(project/CATALOG), "Catalog checksum")
    catalog, designs, base, study, cameras = [read(project/p) for p in (CATALOG, DESIGNS, BASE, STUDY, CAMERAS)]
    rows = catalog["families"]
    require(catalog["schema"] == "el-formal-families-1" and catalog["dataset_id"] == "evidence-lens-families-v1", "Catalog identity")
    require([r["ordinal"] for r in rows] == list(range(1, 25)), "Family ID order")
    require(Counter(r["split"] for r in rows) == {"train": 16, "development": 4, "test": 4}, "Split counts")
    require(Counter(r["cv_fold"] for r in rows if r["split"] == "train") == {i: 4 for i in range(4)}, "Training fold sizes")
    require(len({r["scene_family_id"] for r in rows}) == len({r["geometry_fingerprint"] for r in rows}) == 24,
            "Cross-family ID or geometry overlap")
    require(catalog["representative_families"] == study["storage"]["representative_families"] == REPRESENTATIVES,
            "Representative group conflicts with preregistration")
    require(digest(project/STUDY) == catalog["preregistration_sha256"], "Scientific preregistration checksum")
    require(catalog["condition_matrix"] == {"B0": 24, "R1": 4, "Q1": 4} and
            Counter(c for row in rows for c in row["trials"]) == catalog["condition_matrix"], "Condition matrix")
    require(set(designs["families"]) == {r["scene_family_id"] for r in rows[1:]}, "Geometry inventory mismatch")
    require(len({t for r in rows for t in r["trials"].values()}) == 32, "Duplicate scientific trial ID")
    require(cameras["schema"] == "el-camera-1" and set(cameras["views"]) == {"source", *TARGETS}, "Camera inventory")
    for row in rows:
        i, f, split = row["ordinal"], row["scene_family_id"], row["split"]
        require(re.fullmatch(f"scene_{i:03d}_[a-z0-9_]+", f) is not None and row["scene_id"] == f+"/v001"
                and row["scene_version"] == "v001" and row["example_view"] == "v01", f"{f}: invalid identity")
        require(row["cv_fold"] == ((i-1)//4 if i<=16 else None) and split == ("train" if i<=16 else "development" if i<=20 else "test"),
                f"{f}: split/fold deviates from frozen unstratified blocks")
        require(set(row["trials"]) == ({"B0","R1","Q1"} if split=="test" else {"B0"})
                and row["trial_id"] == row["trials"]["B0"], f"{f}: invalid trial plan")
        if i == 1:
            require(f == "scene_001_tabletop" and row["trial_id"] == "tabletop-v001-b0" and
                    row["reuse"]["selection_sha256"] == study["known_at_registration"]["engineering_selection_sha256"],
                    "Engineering lineage")
            objects = base["reference"]["objects"]
        else:
            require(row["trials"] == {c: f+"-"+c for c in row["trials"]} and
                    row["camera_sha256"] == digest(project/CAMERAS), f"{f}: camera/trial identity")
            recipe = designs["families"][f]
            require(value_hash(recipe) == row["recipe_sha256"], f"{f}: recipe checksum")
            objects = designs["shared_world"]["objects"] + recipe["objects"]
        require(len({o["name"] for o in objects}) == len({o["id"] for o in objects}) == len(objects),
                f"{f}: duplicate mesh identity")
        require(all(o["material"] in base["reference"]["materials"] and o["id"] > 0 for o in objects), f"{f}: material or ID")
        require(geometry_fingerprint(objects) == row["geometry_fingerprint"], f"{f}: geometry checksum")
    return catalog, designs, base, lock


def resolved(row, catalog, designs, base, lock):
    require(row["ordinal"] != 1, "Engineering reference is reused verbatim")
    family = row["scene_family_id"]
    spec = copy.deepcopy(base["reference"])
    world = designs["shared_world"]
    spec["objects"] = copy.deepcopy(world["objects"] + designs["families"][family]["objects"])
    spec["camera"] = copy.deepcopy(world["camera"])
    spec["area_light"] = copy.deepcopy(world["area_light"])
    spec["scene_name"] = family+"_v001"
    spec["validation"].pop("occluder_id")
    spec["validation"].pop("revealed_object_id")
    spec["validation"]["occlusion_profile"] = "opaque_mesh_pairs"
    spec["resize_rounding"] = "nearest_even"
    return {"scene_family_id": family, "scene_id": row["scene_id"], "scene_version": row["scene_version"],
            "dataset_id": catalog["dataset_id"], "dataset_hash": lock["catalog_sha256"],
            "camera_set_id": row["camera_set_id"], "camera_set_hash": row["camera_sha256"],
            "split": row["split"], "condition_id": "B0", "experiment_id": catalog["experiment_id"], "trial_id": row["trial_id"],
            "recording": {"expected_output_bytes": 170_000_000},
            "formal": {"schema": "el-formal-reference-1", "recipe_sha256": row["recipe_sha256"],
                       "geometry_fingerprint": row["geometry_fingerprint"], "cv_fold": row["cv_fold"],
                       "example_view": row["example_view"], "preregistration_sha256": catalog["preregistration_sha256"]},
            "reference": spec}
