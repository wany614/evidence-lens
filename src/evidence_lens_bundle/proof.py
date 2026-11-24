"""Offline, fixed-role provenance for incomplete legacy DTU and recovery deployment Cases."""

from __future__ import annotations

import re

from .common import digest, exact_keys, json_bytes, member, read_file, require, sha

RID = re.compile(r"[0-9]{8}T[0-9]{12}Z-[a-f0-9]{16}")
DOCS = {
    "deployment": "payload/case-provenance.json",
    "normalized": "payload/outputs.json",
    "generator": "payload/reconstruction.json",
    "renderer": "payload/outputs.json",
    "original": "payload/case-provenance.json",
    "camera_plan": "payload/camera-plan.json",
    "crop": "payload/crop.json",
    "metric": "payload/metric-frame.json",
    "depth_probe": "payload/ed-probe.json",
}
RIG_ROLES = {"camera_plan", "crop", "metric"}


def verified_docs(directory, expected):
    raw = read_file(directory, "proof.json")
    require(sha(raw) == digest(expected, "independent proof index"), "legacy proof SHA-256 differs")
    meta = json_bytes(raw)
    exact_keys(meta, ("schema", "kind", "original_case_sha256", "members", "seals"), "legacy proof")
    require(
        meta["schema"] == "el-legacy-deploy-proof-1" and meta["kind"] in ("dtu", "recovery"),
        "unsupported legacy proof schema",
    )
    names = set(DOCS) if meta["kind"] == "recovery" else RIG_ROLES | {"deployment"}
    exact_keys(meta["members"], names, "evidence roles")
    docs = {}
    pins = {}
    sealed = {}
    for role, item in meta["members"].items():
        exact_keys(item, ("origin", "file"), "evidence member")
        origin = item["origin"]
        exact_keys(origin, ("run_id", "manifest_sha256", "member", "sha256"), "sealed origin")
        rid = origin["run_id"]
        require(
            isinstance(rid, str) and RID.fullmatch(rid) and origin["member"] == DOCS[role],
            "unexpected evidence origin: " + role,
        )
        if rid not in sealed:
            seal = meta["seals"][rid]
            exact_keys(seal, ("manifest", "run"), "seal")
            manifest = member(directory, seal["manifest"], expected="seals/" + rid + "/manifest.json")
            body = json_bytes(manifest)
            require(
                sha(manifest) == origin["manifest_sha256"] and body["run_id"] == rid,
                "evidence seal does not match",
            )
            entries = {r["path"]: r for r in body["files"]}
            require(len(entries) == len(body["files"]), "duplicate original manifest path")
            rb = member(directory, seal["run"], expected="seals/" + rid + "/run.json")
            execution = json_bytes(rb)
            require(
                sha(rb) == entries["run.json"]["sha256"]
                and len(rb) == entries["run.json"]["bytes"]
                and execution["run_id"] == rid
                and execution["execution_status"] == "succeeded",
                "original evidence must have a successful sealed run",
            )
            sealed[rid] = (sha(manifest), entries)
        seal_sha, entries = sealed[rid]
        data = member(directory, item["file"], expected="documents/" + role + ".json")
        original = entries[DOCS[role]]
        require(
            origin["manifest_sha256"] == seal_sha
            and original["sha256"] == origin["sha256"] == sha(data)
            and original["bytes"] == len(data),
            "document differs from original seal: " + role,
        )
        pins[role] = origin
        docs[role] = json_bytes(data)
    require(set(meta["seals"]) == set(sealed), "undeclared or missing evidence seal")
    return meta, docs, pins, sealed


def run_pin(pin):
    return {k: pin[k] for k in ("run_id", "manifest_sha256")}


def geometry(case, physical, docs, pins):
    plan, crop, metric = (docs[k] for k in ("camera_plan", "crop", "metric"))
    rig = physical["rig_registration"]
    require(
        all(run_pin(pins[k]) == rig for k in RIG_ROLES)
        and plan["schema"] == "el-dtu-camera-plan-1"
        and pins["camera_plan"]["sha256"] == physical["camera_plan_sha256"]
        and plan["crop_sha256"] == physical["crop_sha256"] == pins["crop"]["sha256"]
        and plan["metric_frame_sha256"] == pins["metric"]["sha256"]
        and plan["converter"] == crop["id"] == "dtu-crop256-v1"
        and crop["schema"] == "el-dtu-crop-1"
        and metric["schema"] == "el-dtu-metric-frame-1"
        and crop["output"] == [256, 256]
        and "Rectified" in plan["evidence"]["rectification"],
        "unsupported or unrelated rectified camera proof",
    )
    for k in ("source_cameras", "render_cameras"):
        require(
            plan["condition_cameras"][case.condition][k]["sha256"] == case.meta[k]["sha256"],
            "proof camera binding differs: " + k,
        )
    require(
        physical["condition"] == case.condition
        and physical["source_view_ids"] == plan["source_view_ids"]
        and physical["target_view_ids"] == plan["target_view_ids"]
        and set(physical["source_images"]) == set(case.generator)
        and all(physical["source_images"][s]["rgb"] == case.meta["source_images"][s] for s in case.generator)
        and [physical["source_view_ids"][k] for k in ("s00", "s03")] == metric["endpoint_ids"],
        "camera or original source identities differ",
    )
    info = {
        "prepared_camera_and_z_unit": "m",
        "meters_per_original_unit": metric["metres_per_native_unit"] * metric["additional_scale"],
        "metric_evidence": pins["metric"]["sha256"],
        "crop": f"({crop['left']}, {crop['top']}), side {crop['side']}; verified {pins['crop']['sha256']}",
        "resize": crop["resampling"],
        "undistortion": plan["evidence"]["rectification"],
        "scoring_resampling": "none",
    }
    return info


def recovery(case, docs, pins, seals):
    p = case.pred
    a, d, g, r = (docs[k] for k in ("deployment", "normalized", "generator", "renderer"))
    require(
        a["schema"] == "el-recovery-deployment-case-1"
        and d["schema"] == r["schema"] == "el-recovery-outputs-1"
        and g["schema"] == "el-recovery-reconstruction-1"
        and d["status"] == g["status"] == r["status"] == "completed",
        "incomplete recovery producer evidence",
    )
    require(
        a["originals"]["outputs"] == pins["normalized"]
        and a["original_provenance"] == d["original_case_provenance"] == pins["original"]
        and a["originals"]["depth_probe"] == d["depth_probe"] == pins["depth_probe"]
        and a["originals"]["original_case"] == d["original_case"]
        and a["originals"]["original_case"]["sha256"] == docs["original"]["case_sha256"]
        and docs["original"]["schema"] == "el-dtu-case-provenance-1",
        "recovery evidence references a different original bundle or probe",
    )
    require(
        d["original_generation"] == run_pin(pins["generator"])
        and d["original_render"] == run_pin(pins["renderer"])
        and d["run_id"] == pins["normalized"]["run_id"]
        and g["run_id"]
        == p["reconstruction_run_id"]
        == a["parameter_source"]
        == d["reconstruction_run_id"]
        == r["reconstruction_run_id"]
        == pins["generator"]["run_id"],
        "wrong recovery generation or render seal",
    )
    require(
        a["object_id"]
        == d["object_id"]
        == g["identity"]["object_id"]
        == r["object_id"]
        == docs["original"]["object_id"]
        and a["condition"] == d["condition"] == g["identity"]["condition"] == r["condition"] == case.condition
        and a["recipe_sha256"] == d["recipe_sha256"] == g["identity"]["recipe_sha256"] == r["recipe_sha256"],
        "wrong recovery object, condition or recipe",
    )
    require(
        p.get("wrapper_schema") == "el-recovery-render-adapter-1"
        and d["depth_verified"] is p["depth_verified"] is True
        and docs["depth_probe"]["passed"] is True
        and p["depth_probe_sha256"] == pins["depth_probe"]["sha256"]
        and all(d[n] == p[n] for n in ("rgb_encoding", "depth_semantics", "alpha_semantics"))
        and r["rgb_encoding"] == "display_srgb_float32"
        and r["depth_semantics"] == "opacity-normalized expected camera Z in metres",
        "recovery raw or normalized channel/probe proof differs",
    )
    require(
        all(
            x["generator_source_ids"] == list(case.generator)
            and x["source_camera_sha256"] == case.meta["source_cameras"]["sha256"]
            and x["source_rgb_sha256"] == p["source_rgb_sha256"]
            for x in (d, g["identity"], r)
        )
        and d["render_camera_sha256"] == r["render_camera_sha256"] == case.meta["render_cameras"]["sha256"]
        and a["source_images"] == p["source_rgb_sha256"],
        "recovery source pixels or cameras differ",
    )
    step = a["step"]
    checkpt = d["checkpoints"][str(step)]
    rawr = r["checkpoints"][str(step)]
    rawg = g["checkpoints"][str(step)]
    cp = checkpt["checkpoint"]
    require(
        a["checkpoint_sha256"]
        == p["checkpoint_sha256"]
        == checkpt["checkpoint_sha256"]
        == rawr["checkpoint_sha256"]
        == rawg["checkpoint_sha256"]
        == cp["sha256"]
        and step == checkpt["step"] == rawr["step"] == rawg["step"]
        and run_pin(cp) == run_pin(pins["generator"])
        and cp["member"] == "payload/" + rawg["path"]
        and seals[cp["run_id"]][1][cp["member"]]["sha256"] == cp["sha256"],
        "wrong recovery checkpoint or step",
    )
    require(
        set(a["render_sha256"]) == set(checkpt["views"]) == set(rawr["views"]) == set(case.meta["renders"]),
        "incomplete source or target render set",
    )
    for v, f in case.meta["renders"].items():
        normalized, raw = checkpt["views"][v], rawr["views"][v]
        require(
            run_pin(normalized) == run_pin(pins["renderer"])
            and normalized["member"] == "payload/" + raw["path"]
            and f["sha256"] == a["render_sha256"][v] == normalized["sha256"] == raw["sha256"]
            and f["bytes"] == normalized["bytes"] == raw["bytes"],
            "wrong selected recovery frame: " + v,
        )
        source = seals[pins["renderer"]["run_id"]][1][normalized["member"]]
        require(
            source["sha256"] == f["sha256"] and source["bytes"] == f["bytes"],
            "original renderer frame mismatch: " + v,
        )
    return (
        {"run_id": g["run_id"], "receipt_sha256": pins["generator"]["sha256"]},
        {"run_id": pins["renderer"]["run_id"], "receipt_sha256": pins["renderer"]["sha256"]},
    )


def identities(case, directory, expected):
    meta, docs, pins, seals = verified_docs(directory, expected)
    a = docs["deployment"]
    require(
        meta["original_case_sha256"] == a["case_sha256"] == case.sha256
        and a["run_id"] == pins["deployment"]["run_id"]
        and seals[a["run_id"]][1]["payload/case/case.json"]["sha256"] == case.sha256,
        "legacy proof identifies a different Case",
    )
    if meta["kind"] == "recovery":
        gen, ren = recovery(case, docs, pins, seals)
        physical = docs["original"]
    else:
        physical = a
        require(
            a["schema"] == "el-dtu-case-provenance-1"
            and a["generation"]["run_id"] == case.pred.get("reconstruction_run_id")
            and a["render"]["run_id"] == case.pred.get("run_id"),
            "wrong DTU generator or renderer",
        )
        gen = {
            "run_id": case.pred["reconstruction_run_id"],
            "receipt_sha256": case.pred["reconstruction_sha256"],
        }
        ren = {"run_id": case.pred["run_id"], "receipt_sha256": case.meta["prediction"]["sha256"]}
    return gen, ren, geometry(case, physical, docs, pins)
