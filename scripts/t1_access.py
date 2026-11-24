"""Broker exact named DTU members for T1. The selector never runs in this process."""

import argparse
import hashlib
import io
import os
import time
import zlib
from pathlib import Path

from dtu_sample import Network
from PIL import Image
from t0_dtu_archive import ExactArchive

from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens._verify import require
from evidence_lens.dtu_camera import convert_rgb

OBJECTS = ["set118", "set76", "set45", "set103", "set61", "set32"]
STRATEGIES = ["uniform", "risk", "random-137101", "random-137102", "random-137103"]
CONDITIONS = ["M2-fixed", "M4-fixed", *["M4-" + s for s in STRATEGIES]]
POOL = [7, 10, 13, 17, 20, 23, 31, 40]
FIXED = [24, 23, 17, 22]
S0 = [24, 22]
H = [16, 18, 25, 37, 35, 36]
TARGETS = [f"v{i:02}" for i in range(1, 7)]
SLOTS = {"M2": ["s00", "s03"], "M4": ["s00", "s01", "s02", "s03"]}
FREEZE_SHA = "b59dddc26d22cb28908ba5b4d27d7795871998e878408243a178ad1c7a2bf946"


def verified_inputs(cfg, inputs):
    require(set(inputs) == set(cfg["sources"]), "Access roles differ from the recorded request")
    for name, path in inputs.items():
        require(sha(path) == cfg["sources"][name]["sha256"], "Changed input " + name)


def registered(start, freeze, freeze_pin):
    require(
        freeze_pin["sha256"] == FREEZE_SHA
        and freeze["T1_objects"] == OBJECTS
        and start["T1_objects"] == OBJECTS
        and start["freeze_sha256"] == FREEZE_SHA
        and start["schema"] == "el-general-t1-entry-1"
        and start["freeze"] == freeze_pin
        and start["source_authorization"] == "S0_only"
        and start["S0"] == S0
        and start["target_reference_permission"] == "closed"
        and start["selection_before_candidate_access"] == 5
        and start["selected_descriptive_risk_scores"] == 0,
        "A current scoped G0 receipt and the unchanged batch-two freeze are required",
    )


def decisions(doc, inputs, obj):
    require(
        doc["schema"] == "el-b2-joint-nbv-choices-1"
        and doc["object_id"] == obj
        and doc["pool"] == POOL
        and doc["legal_photo_ids_before_selection"] == S0
        and set(doc["results"]) == set(STRATEGIES) == set(doc["selections"]),
        "Incomplete five-way decision",
    )
    for name in STRATEGIES:
        f = inputs["choice-" + name]
        row = load(f)
        pair = row["selected_official_ids"]
        require(
            sha(f) == doc["selections"][name]
            and row["strategy_id"] == name
            and row["run_id"] == doc["run_id"]
            and row["object_id"] == obj
            and row["m2_checkpoint_sha256"] == doc["m2_checkpoint_sha256"]
            and pair == doc["results"][name]
            and len(pair) == len(set(pair)) == 2
            and pair == sorted(pair)
            and set(pair) <= set(POOL),
            "Unsealed or inconsistent camera pair",
        )


def photo(raw, row, cam, name, out, crop, reference, source):
    require(len(raw) == row["bytes"] and f"{zlib.crc32(raw):08x}" == row["crc32"], "Wrong archive member")
    png = convert_rgb(raw, crop)
    relative = ("targets/" if reference else "sources/") + name + ".png"
    original = "original/" + name + ".png"
    for path, data in ((original, raw), (relative, png)):
        f = out / path
        f.parent.mkdir(exist_ok=True)
        with f.open("xb") as stream:
            stream.write(data)
    with Image.open(io.BytesIO(png)) as im:
        require(im.mode == "RGB" and im.size == (256, 256), "Unexpected display crop")
        pixel_hash = hashlib.sha256(im.tobytes()).hexdigest()
    result = {
        "official_view_id": cam,
        "archive_member": row["name"],
        "original_path": original,
        "original_crc32": row["crc32"],
        "original_bytes": len(raw),
        "original_sha256": sha(out / original),
        "rgb": {"path": relative, "bytes": len(png), "sha256": sha(out / relative)},
        "cropped_display_rgb8_pixel_sha256": pixel_hash,
        "acquisition": source,
    }
    if reference:
        result.update(
            valid_mask={"kind": "all_true", "width": 256, "height": 256},
            object_mask=None,
            object_mask_reason="No independently verified 2D object mask.",
        )
    return result


def access(cfg, inputs, out):
    verified_inputs(cfg, inputs)
    g0, freeze, cohort = (load(inputs[k]) for k in ("entry", "freeze", "cohort"))
    registered(g0, freeze, cfg["sources"]["freeze"])
    obj, name = cfg["object_id"], cfg["logical_condition"]
    require(
        obj in OBJECTS
        and cohort["splits"]["T1"] == OBJECTS
        and cohort["lighting_id"] == 3
        and freeze["cohort"] == cfg["sources"]["cohort"],
        "Unregistered physical object or light",
    )
    (subject,) = [r for r in cohort["objects"] if r["object_id"] == obj and r["role"] == "T1"]
    crop = load(inputs["crop"])
    require(sha(inputs["crop"]) == cohort["camera"]["crop_sha256"], "Changed crop")
    reference = name == "references"
    if reference:
        op = load(inputs["opening"])
        require(
            op["schema"] == "el-t1-joint-opening-1"
            and op["object_ids"] == OBJECTS
            and op["target_ids"] == TARGETS
            and op["predictions_and_scores_sealed"] is True
            and op["condition_count"] == 42
            and op["fixed_score_count"] == 288
            and op["entry"] == cfg["sources"]["entry"],
            "Wrong joint opening",
        )
        ends = load(inputs["terminals"])
        require(
            ends["schema"] == "el-t1-reconstruction-terminals-1"
            and ends["entry"] == cfg["sources"]["entry"]
            and len(ends["cells"]) == 42
            and ends["sealed_strategy_terminals"] == 30
            and [(c["object_id"], c["condition"]) for c in ends["cells"]]
            == [(o, c) for o in OBJECTS for c in CONDITIONS],
            "Wrong terminal manifest",
        )
        slots, cameras = TARGETS, H
    else:
        require(name in CONDITIONS, "Unregistered trajectory")
        cond = name[:2]
        slots = SLOTS[cond]
        cameras = S0
        if cond == "M4":
            choices = load(inputs["choices"])
            decisions(choices, inputs, obj)
            cameras = FIXED if name == "M4-fixed" else [24, *choices["results"][name[3:]], 22]
        source, render = (load(inputs[k]) for k in ("sourceRig", "renderRig"))
        fixed = cohort["camera"]["conditions"][cond]
        require(
            sha(inputs["sourceRig"]) == fixed["source_cameras"]["sha256"]
            and sha(inputs["renderRig"]) == fixed["render_cameras"]["sha256"]
            and set(source["views"]) == set(slots)
            and set(render["views"]) == set(slots + TARGETS),
            "Wrong frozen rig",
        )
        allowed = load(inputs["candidateRig"])
        require(
            allowed["candidate_ids"] == POOL
            and allowed["endpoint_ids"] == S0
            and set(allowed["cameras"]) == {str(n) for n in S0 + POOL},
            "Wrong candidate rig",
        )
        for slot, cam in zip(slots, cameras, strict=True):
            v = allowed["cameras"][str(cam)]
            if name.endswith("-fixed"):
                require(source["views"][slot] == render["views"][slot] == v, "Fixed camera changed")
            source["views"][slot] = render["views"][slot] = v
        for typ, document in (("source", source), ("render", render)):
            original = inputs[typ + "Rig"]
            # Keep the exact original file bytes for the two fixed layouts.
            (out / (typ + "_cameras.json")).write_bytes(
                original.read_bytes() if name.endswith("-fixed") else encoded(document)
            )
    net = Network(cohort["budget"]["max_response_bytes_per_data_cell"], out / "http.jsonl")
    remote = None
    times = time.monotonic()
    rows = {}
    try:
        for slot, cam in zip(slots, cameras, strict=True):
            m = subject["members"][str(cam)]
            require(
                m["name"] == f"Rectified/scan{subject['scan_id']}/rect_{cam:03}_3_r5000.png",
                "Wrong named range",
            )
            cache = "cached-" + slot
            if cache in inputs:
                raw = inputs[cache].read_bytes()
                acquisition = {"kind": "authorized byte cache", "pin": cfg["sources"][cache]}
            else:
                if remote is None:
                    remote = ExactArchive(net, cohort["archive"]["url"], cohort["archive"])
                raw = remote.member(m)
                acquisition = {"kind": "exact named HTTP ranges"}
            rows[slot] = photo(raw, m, cam, slot, out, crop, reference, acquisition)
    finally:
        save(
            out / "access.json",
            {
                "run_id": os.environ["EL_RUN_ID"],
                "object_id": obj,
                "logical_condition": name,
                "authorized_order": cameras,
                "accessed_order": [r["official_view_id"] for r in rows.values()],
                "HTTP_response_bytes": net.total,
                "elapsed_seconds": time.monotonic() - times,
                "observed_reference_count": len(rows) if reference else 0,
            },
        )
    common = {
        "run_id": os.environ["EL_RUN_ID"],
        "object_id": obj,
        "physical_family_id": subject["physical_family_id"],
        "scan_id": subject["scan_id"],
        "lighting_id": 3,
        "cohort": cfg["sources"]["cohort"],
        "entry": cfg["sources"]["entry"],
        "camera_plan_sha256": cohort["camera"]["plan_sha256"],
        "camera_sha256": cohort["camera"]["camera_sha256"],
        "view_set_sha256": cohort["camera"]["view_set_sha256"],
        "crop_sha256": sha(inputs["crop"]),
        "source_view_ids": cohort["source_view_ids"],
        "target_view_ids": cohort["target_view_ids"],
    }
    if reference:
        save(
            out / "references.json",
            dict(
                **common,
                schema="el-t1-references-1",
                views=rows,
                target_ids=slots,
                opening=cfg["sources"]["opening"],
            ),
        )
    else:
        save(
            out / "source-provenance.json",
            dict(
                **common,
                schema="el-t1-source-1",
                condition=name[:2],
                logical_condition=name,
                sources=rows,
                source_ids=slots,
                official_by_slot=dict(zip(slots, cameras, strict=True)),
                reference_photo_reads=0,
                other_branch_candidate_observations=0,
                decisions=cfg["sources"].get("choices"),
                own_decision=cfg["sources"].get("choice-" + name[3:]),
                source_camera_sha256=sha(out / "source_cameras.json"),
                render_camera_sha256=sha(out / "render_cameras.json"),
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ("config", "out"):
        parser.add_argument("--" + n, type=Path, required=True)
    parser.add_argument("--input", action="append", nargs=2, required=True)
    a = parser.parse_args()
    access(load(a.config), {k: Path(v) for k, v in a.input}, a.out)
