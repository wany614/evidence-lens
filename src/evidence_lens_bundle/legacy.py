"""Adapt a validated immutable Stage C Case by copying only declared deployment roles."""

import os
import uuid
from pathlib import Path

from evidence_lens.multisource_io import Case, read_member
from evidence_lens.scoring_io import new_output

from .bundle import Bundle
from .common import (
    PROFILE,
    RELEASE_INDEX_SHA256,
    SEMANTICS,
    encoded,
    json_bytes,
    profile,
    read_file,
    require,
    row,
    sha,
    text,
    view_objects,
    write,
)


def adapt(case_dir, expected_sha256, output, *, input_domain, run_id=None, proof=None, proof_sha256=None):
    case_raw = read_file(case_dir, "case.json")
    require(sha(case_raw) == expected_sha256, "case identity mismatch")
    case_meta = json_bytes(case_raw)
    declared = text(case_meta.get("reconstruction_run_id"), "case.json reconstruction_run_id")
    prediction_spec = case_meta.get("prediction")
    require(isinstance(prediction_spec, dict), "case.json prediction must describe prediction.json")
    prediction = json_bytes(read_member(case_dir, prediction_spec, "prediction.json"))
    reported = text(prediction.get("reconstruction_run_id"), "prediction.json reconstruction_run_id")
    require(declared == reported, "case.json and prediction.json reconstruction_run_id differ")
    view_objects(prediction, "prediction")
    old = Case(case_dir, expected_sha256)
    p = old.pred
    for key, name in (("rgb_encoding", "rgb"), ("depth_semantics", "z"), ("alpha_semantics", "alpha")):
        require(p.get(key) == SEMANTICS[name], "verified explicit legacy channel semantics required: " + key)
    require(
        (proof is None) == (proof_sha256 is None), "supply both --proof and its independent --proof-sha256"
    )
    geometry = None
    if proof is None:
        require(
            p.get("run_id") and p.get("reconstruction_sha256"),
            "legacy exporter omits producer identities; supply a verified --proof and --proof-sha256",
        )
        generator = {"run_id": p["reconstruction_run_id"], "receipt_sha256": p["reconstruction_sha256"]}
        renderer = {"run_id": p["run_id"], "receipt_sha256": old.meta["prediction"]["sha256"]}
    else:
        from .proof import identities

        generator, renderer, geometry = identities(old, proof, proof_sha256)

    source_order = list(old.generator)
    target_order = list(p["target_ids"])
    cfg = {
        "source_order": source_order,
        "target_order": target_order,
        "primary_source_id": "s00",
        "endpoint_source_ids": ["s00", "s03"],
        "original_case_sha256": old.sha256,
        "input_domain": input_domain,
    }
    if proof_sha256:
        cfg["legacy_proof_sha256"] = proof_sha256
    with new_output(output) as dest:

        def copy(spec, relative):
            data = read_member(old.root, spec, spec["path"])
            path = dest / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return row(relative, data)

        if geometry is None:
            camera = copy(old.meta["render_cameras"], "cameras.json")
        else:
            spec = old.meta["render_cameras"]
            calibration = json_bytes(read_member(old.root, spec, spec["path"]))
            for view in calibration["views"].values():
                require(
                    view.get("projection", "pinhole") == "pinhole"
                    and view.get("distortion", "none") == "none",
                    "camera fields contradict authenticated rectification",
                )
                view.update(projection="pinhole", distortion="none")
            raw = encoded(calibration)
            (dest / "cameras.json").write_bytes(raw)
            camera = row("cameras.json", raw)
        photos = {
            s: copy(old.meta["source_images"][s], f"sources/{i:03d}.png") for i, s in enumerate(source_order)
        }
        frames = {}
        for i, v in enumerate(source_order + target_order):
            frames[v] = (
                {"status": "render_failed", "reason": old.failed[v]}
                if v in old.failed
                else {"status": "ready", "file": copy(old.meta["renders"][v], f"renders/{i:03d}.npz")}
            )
        producer = {
            "schema": "el-render-evidence-1",
            "generator": generator,
            "renderer": renderer,
            "checkpoint_sha256": p["checkpoint_sha256"],
            "source_rgb_sha256": p["source_rgb_sha256"],
            "source_order": source_order,
            "target_order": target_order,
            "camera_sha256": camera["sha256"],
            "semantics": SEMANTICS,
            "depth_verified": p["depth_verified"],
            "depth_probe_sha256": p.get("depth_probe_sha256"),
            "frames": frames,
        }
        raw = encoded(producer)
        (dest / "producer.json").write_bytes(raw)
        declared = "Prepared legacy grid retained; preceding operation is not specified in the Case."
        b = {
            "schema": "el-evidence-bundle-1",
            "creation": {
                "run_id": run_id or os.getenv("EL_RUN_ID") or str(uuid.uuid4()),
                "implementation": "evidence-lens adapt-legacy v1",
                "code_sha256": sha(Path(__file__).read_bytes()),
                "config_sha256": sha(encoded(cfg)),
                "original_case_sha256": old.sha256,
            },
            "input_domain": input_domain,
            "condition": old.condition,
            "source_order": source_order,
            "target_order": target_order,
            "primary_source_id": "s00",
            "endpoint_source_ids": ["s00", "s03"],
            "camera": camera,
            "geometry": geometry
            or {
                "prepared_camera_and_z_unit": "m",
                "meters_per_original_unit": 1,
                "metric_evidence": "Original Case camera document declares metres and renderer declares expected metric camera Z.",
                "crop": declared,
                "resize": declared,
                "undistortion": "Prepared camera declares pinhole with zero distortion.",
                "scoring_resampling": "none",
            },
            "sources": photos,
            "producer": row("producer.json", raw),
            "risk_profile": {
                "id": PROFILE,
                "sha256": sha(encoded(profile())),
                "release_index_sha256": RELEASE_INDEX_SHA256,
            },
        }
        if proof_sha256:
            b["creation"]["legacy_proof_sha256"] = proof_sha256
        write(dest / "bundle.json", b)
        result = Bundle(dest).validate()
    return result
