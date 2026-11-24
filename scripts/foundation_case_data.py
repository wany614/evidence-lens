"""Publish the registered two authored exhibit cases directly from completed A and B recorder artifacts."""

import argparse
import json
import os
from pathlib import Path
import shutil

from dual_records import selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from foundation_case_bell import bell_outputs
from foundation_case_gallery import gallery_outputs
from foundation_case_io import ALLOWED_CASES, ATTRIBUTION, PHOTO_ZIP, Native, Published, decode_png, digest

LABEL = "c-foundation-case-data"
PARENTS = {
    "registration": "c-foundation-registration", "A": "a-baseline", "B": "b-research-release",
    "bell_bundles": "a-bell-bundle-publication", "bell_sources": "a-bell-source-observations",
    "bell_scores": "a-bell-scores-sealed", "bell_opening": "a-bell-reference-opening",
    "bell_truth": "a-bell-references", "bell_evaluation": "a-bell-landscape-comparison-packaged",
    "gallery_source": "b-source-b05-perforated-gallery", "gallery_bank": "b-bank-b05-perforated-gallery-r128",
    "gallery_scores": "b-matrix-b05-perforated-gallery-r128", "gallery_freeze": "b-model-freeze",
    "gallery_opening": "b-confirmation-h-opening", "gallery_truth": "b-truth-b05-perforated-gallery",
    "gallery_evaluation": "b-eval-b05-perforated-gallery-r128", "gallery_arithmetic": "b-independent-arithmetic-check",
}


def register(store, retry=None):
    pins = {key: selected(store, name) for key, name in PARENTS.items()}
    return stage(store, LABEL, "archive_verify", pins,
                 {"cases": list(ALLOWED_CASES), "bell_methods": ["MS-L1", "MS-ft", "MS-frozen", "B-L1-transfer"],
                  "bell_grid": [448, 252], "gallery_view": "h00", "coverage_k": [4, 8, 16, 32, "all"],
                  "public_photography": "only both registered authored scenes under CC BY 4.0",
                  "new_model_fits": 0, "new_renders": 0},
                 ["{python}", "-B", "{source}/scripts/foundation_case_data.py", "worker", "--store", str(store),
                  "--config", "{config}", "--out", "{out}"], "payload/case-index.json", 155_000_000, retry=retry)


def local_refs(item):
    if isinstance(item, list):
        for value in item:
            yield from local_refs(value)
    if isinstance(item, dict):
        if isinstance(item.get("path"), str) and item["path"].endswith(".png"):
            yield item
        for value in item.values():
            yield from local_refs(value)


def verify_local(directory, index, inventory):
    files = inventory["files"]
    items = {r["path"]: r for r in files}
    require(len(items) == len(files) and {q["path"] for q in local_refs(index)} == set(items),
            "case references and positive local PNG inventory disagree")
    actual = {str(f.relative_to(directory)) for case in ALLOWED_CASES for f in (directory / case).rglob("*") if f.is_file()}
    require(actual == set(items), "unlisted file or missing published authored file")
    for rec in local_refs(index):
        require(items[rec["path"]] == rec, "case PNG does not match the positive inventory")
    for name, item in items.items():
        Published.allowed(name)
        path = directory / name
        require(not path.is_symlink() and path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "changed " + name)
        rgb = decode_png(path.read_bytes())
        require(list(rgb.shape[1::-1]) == item["size"] and digest(rgb.tobytes()) == item["pixel_sha256"],
                "decoded RGB differs " + name)
    require([case["id"] for case in index["cases"]] == list(ALLOWED_CASES)
            and [len(case["views"]) for case in index["cases"]] == [6, 1]
            and [len(case["sources"]) for case in index["cases"]] == [4, 128]
            and [c["k"] for c in index["cases"][1]["views"][0]["conditions"]] == [4, 8, 16, 32, 128],
            "incomplete registered exhibit")


def worker(store, config_path, out):
    pins = load(config_path)["sources"]
    native = {k: Native(store, pin) for k, pin in pins.items()}
    registered = native["registration"].metadata
    require(registered["schema"] == "el-humanities-foundation-registration-1" and registered["parents"]["A"] == pins["A"]
            and registered["parents"]["B"] == pins["B"] and native["B"].metadata["status"] == "completed_negative",
            "the research foundations must be sealed before extracting cases")
    pub = Published(out)
    try:
        cases = [bell_outputs(native, pub), gallery_outputs(native, pub)]
        archive_index, inside = pub.finish()
    finally:
        pub.zip.close()
    archive_path = out / PHOTO_ZIP
    archive = dict(file=PHOTO_ZIP, sha256=sha(archive_path), bytes=archive_path.stat().st_size,
                   url="https://github.com/wany614/evidence-lens/releases/download/humanities-foundation-v1/" + PHOTO_ZIP,
                   index_member="contents.json", index_sha256=inside["sha256"], member_count=len(archive_index["files"]),
                   contents="Source and independent reference PNGs, native prediction and rank arrays, and original independent bell FP32 measurements. Each role is recorded for each member.")
    save(out / "source-photos.json", dict(**archive_index, archive=archive))
    inventory = dict(schema="el-foundation-offline-png-inventory-1", rights=ATTRIBUTION,
                     files=sorted(pub.local.values(), key=lambda row: row["path"]))
    save(out / "case-files.json", inventory)
    index = dict(schema="el-foundation-case-index-1", version="1.0.0", run_id=os.environ["EL_RUN_ID"],
                 registration=pins["registration"], A_BASELINE_ID=registered["A_BASELINE_ID"],
                 B_research_release=pins["B"], B_status=registered["B_status"], rights=ATTRIBUTION,
                 photograph_archive=archive, source_photo_archive=archive, cases=cases,
                 offline_inventory_sha256=sha(out / "case-files.json"),
                 full_resolution_inventory_sha256=sha(out / "source-photos.json"),
                 display="Analysis, prediction and error maps use their native grid. Source thumbnails are for viewing; arrays and full-size sources have independent digests.",
                 array_digests="Hashes labelled typed include canonical dtype and shape bytes before contiguous C-order data. target_rgb_sha256 hashes only the contiguous native FP32 RGB values. target_prediction_sha256 is the complete original RGB/alpha/Z NPZ file.",
                 new_training_runs=0, new_renders=0)
    verify_local(out, index, inventory)
    save(out / "case-index.json", index)


def copy_offline(store, dest):
    pin = selected(store, LABEL)
    native = Native(store, pin)
    index = native.metadata
    inventory = load(native.path("case-files.json", index["offline_inventory_sha256"]))
    dest.mkdir(parents=True, exist_ok=True)
    for item in inventory["files"] + [dict(path="case-index.json", sha256=pin["sha256"])]:
        p = item["path"]
        if p.endswith(".png"):
            Published.allowed(p)
        src = native.path(p, item)
        target = dest / p
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(sha(target) == item["sha256"], "existing checkout diverges from sealed member " + p)
        else:
            shutil.copyfile(src, target)
    verify_local(dest, index, inventory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["record", "worker", "copy-offline"])
    for name in ("store", "out", "config"):
        parser.add_argument("--" + name, type=Path, required=name == "store")
    parser.add_argument("--retry")
    a = parser.parse_args()
    if a.mode == "record":
        print(json.dumps(register(a.store.resolve(), a.retry), indent=2))
    elif a.mode == "worker":
        worker(a.store, a.config, a.out)
    else:
        copy_offline(a.store, a.out)
