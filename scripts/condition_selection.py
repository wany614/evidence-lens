"""Validate a held-out stress chain and freeze its first eligible scientific attempts."""

import hashlib
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from evidence_lens._bundle import encoded, load, sha, utc
from evidence_lens._index import writer_gate
from evidence_lens._verify import require, timestamp
from evidence_lens.recorder import worker_started
from evidence_lens.scoring_io import TARGETS, VIEWS, PredictionBundle, ReadAudit, load_cameras, load_source_png
from condition_jobs import QUALITY
from formal_lineage import run_pin
from inspect_sharp_ply import VERTICES, read_ply
from make_handoff import publish
from select_trial import transfer_evidence
from stress_condition import COLOUR_FIELDS, envelope


def path(ctx, rid, name):
    return ctx["store"] / "runs" / rid / "payload" / name


def digest(ctx, rid, name):
    row = ctx["verified"].get(rid)["artifacts"].get("payload/" + name)
    require(row and row["bytes"] > 0 and row["status"] == "valid", f"unavailable condition asset {name}")
    return row["sha256"]


def bound(ctx, child, owner, names):
    rows = ctx["verified"].get(child)["inputs"]["inputs"]
    received = {f["sha256"] for i in rows if i["parent_run_id"] == owner for f in i["files"]}
    require({digest(ctx, owner, n) for n in names} <= received, f"condition run {child} did not bind exact parent inputs from {owner}")


def accepted_local(ctx, rid, kind, role, parents, dispatch):
    checked = ctx["verified"].get(rid, kind)
    meta = checked["run"]
    cfg = load(ctx["store"] / "runs" / rid / "config.resolved.yaml")
    require(meta["condition_id"] == ctx["condition"] and meta["trial_id"] == ctx["quality_trial"] and meta["experiment_id"] == QUALITY
            and set(meta["parent_run_ids"]) == set(parents) and all(meta[k] == ctx["family"][k] for k in ("scene_id","scene_family_id","dataset_id","dataset_hash","split"))
            and cfg["condition_stage_role"] == role and cfg["formal_handoff_sha256"] == dispatch["handoff_sha256"] and
            cfg["method_freeze_sha256"] == ctx["release"]["method_freeze_sha256"] and cfg["b0_selection_sha256"] == ctx["b0_selection_sha256"],
            "condition Mac stage has a different role, family, release or exact parent graph")
    gate_inputs = [obj for entry in checked["inputs"]["inputs"] if entry["role"] == "method-freeze" for obj in entry["files"]]
    require(len(gate_inputs) == 1 and gate_inputs[0]["sha256"] == ctx["release"]["method_freeze_sha256"],
            "condition stage did not freeze the exact method release file")
    return checked


def validate_windows(ctx, ids, dispatch):
    job, condition = dispatch["job"], ctx["condition"]
    main, render, basegen = ids["producer"], ids["render"], ctx["b0_ids"]["generation"]
    source = ids["preparation"] if condition == "R1" else ids["reference"]
    gen = ctx["verified"].get(main, "generation" if condition == "R1" else "derivation")
    rendered = ctx["verified"].get(render, "render")
    for checked in (gen, rendered):
        rec = checked["run"]
        cfg = load(ctx["store"] / "runs" / rec["run_id"] / "config.resolved.yaml")
        require(rec["experiment_id"] == QUALITY and rec["trial_id"] == ctx["quality_trial"] and rec["condition_id"] == condition and
                all(rec[k] == job["identity"][k] for k in ("scene_id","scene_family_id","scene_hash","dataset_id","dataset_hash","split","camera_set_hash")) and
                cfg["timing_repeat"] is None and cfg["formal_handoff_sha256"] == dispatch["handoff_sha256"] and cfg["worker_source_sha256"] == job["worker_source_sha256"],
                "Windows stage is a different condition dispatch or timing repeat")
    require(gen["run"]["parent_run_ids"] == [source if condition == "R1" else basegen] and
            set(rendered["run"]["parent_run_ids"]) == {source, main}, "condition generator/derivation/render parent graph differs")
    if condition == "R1":
        bound(ctx, main, source, ("input.png", "cameras.json"))
    else:
        bound(ctx, main, basegen, ("scene.ply", "generation.json"))
    bound(ctx, render, source, ("input.png", "cameras.json"))
    bound(ctx, render, main, ("scene.ply", "generation.json"))
    g = load(path(ctx, main, "generation.json"))
    require(g["producer_run_id"] == main and g["input_sha256"] == digest(ctx, source, "input.png") and
            g["camera_sha256"] == digest(ctx, source, "cameras.json") and g["ply_sha256"] == digest(ctx, main, "scene.ply") and
            type(g["gaussian_count"]) is int and g["gaussian_count"] > 0, "condition PLY/input hashes differ")
    if condition == "R1":
        require(g["schema"] == "el-generation-1" and g["source_size_wh"] == [256, 256] and g["network_input_wh"] == [1536, 1536] and
                g["checkpoint_sha256"] == ctx["baseline"]["evidence"]["checkpoint_sha256"] and
                g["checkpoint_sha256"] in {obj["sha256"] for obj in gen["shared"]}, "R1 prediction is not from the pinned generator")
    else:
        require(g["schema"] == "el-gaussian-derivation-1" and g["condition_id"] == "Q1" and g["parent_generation_run_id"] == basegen and
                g["parent_ply_sha256"] == digest(ctx, basegen, "scene.ply") and g["forward_calls_in_this_stage"] == 0 and
                tuple(g["colour_fields"]) == COLOUR_FIELDS, "Q1 has invalid geometry provenance or performed a network call")
    pred = load(path(ctx, render, "prediction.json"))
    probe = load(path(ctx, render, "depth-probe.json"))
    require(pred["producer_run_id"] == render and pred["generation_run_id"] == main and pred["ply_sha256"] == g["ply_sha256"] and
            pred["camera_sha256"] == g["camera_sha256"] and pred["input_sha256"] == g["input_sha256"] and not pred.get("failed_views") and
            set(pred["views"]) == set(VIEWS) and all(probe.get(k) is True for k in ("off_axis_plane_passed","alpha_normalization_passed","translation_passed","linear_blend_passed"))
            and type(probe.get("empty_pixel_alpha")) in (int, float) and probe["empty_pixel_alpha"] == 0 and pred["depth_validation"] == probe and pred["depth_validation_receipt_sha256"] == digest(ctx, render, "depth-probe.json"),
            "condition render or measured depth, translation, colour or empty calibration failed")
    for v in VIEWS:
        require(pred["views"][v]["sha256"] == digest(ctx, render, v + ".npz"), "condition view asset hash differs")
        digest(ctx, render, v + ".png")
    receipts = transfer_evidence(ctx["store"], {main:gen, render:rendered})
    require(receipts.get(main) and receipts.get(render), "missing Mac transfer receipt for a condition worker stage")
    return receipts


def q1_fields(ctx, ids):
    src, dst = path(ctx, ids["generation"], "scene.ply"), path(ctx, ids["producer"], "scene.ply")
    before, _ = read_ply(src)
    after, _ = read_ply(dst)
    require(len(before) == len(after) and envelope(src, before) == envelope(dst, after), "Q1 PLY vertex count/header/metadata changed")
    receipt = load(path(ctx, ids["producer"], "generation.json"))
    require(tuple(receipt["header_and_metadata_sha256"]) == envelope(dst, after), "Q1 PLY envelope hash differs")
    for name in VERTICES:
        a, b = np.ascontiguousarray(before[name]), np.ascontiguousarray(after[name])
        if name in COLOUR_FIELDS:
            # Python's struct binary16 implementation is independent of the
            # producer's NumPy cast. Compare every binary32 bit after roundtrip.
            for offset in range(0, len(a), 16384):
                chunk = a[offset:offset + 16384]
                width = len(chunk)
                half = struct.pack(f"<{width}e", *chunk.tolist())
                expected = struct.pack(f"<{width}f", *struct.unpack(f"<{width}e", half))
                require(expected == b[offset:offset + width].tobytes(), f"Q1 {name} differs from independent binary16 rounding")
        else:
            require(a.tobytes() == b.tobytes(), f"protected Q1 field {name} changed")
        require(receipt["schema_fields"][name]["input_sha256"] == hashlib.sha256(a.tobytes()).hexdigest() and
                receipt["schema_fields"][name]["output_sha256"] == hashlib.sha256(b.tobytes()).hexdigest(), f"Q1 {name} failed independent field check")
    return {"checked_fields":list(VERTICES), "rounded_fields":list(COLOUR_FIELDS), "vertices":len(before), "header_and_tail_match":True,
            "struct_binary16_values_checked":int(len(before) * len(COLOUR_FIELDS))}


def r1_focal_scaling(ctx, ids):
    baseline_id, reduced_id = ctx["b0_ids"]["generation"], ids["producer"]
    baseline = load(path(ctx, baseline_id, "generation.json"))
    reduced = load(path(ctx, reduced_id, "generation.json"))
    baseline_cam = load(path(ctx, ids["reference"], "cameras.json"))["views"]["source"]
    reduced_cam = load(path(ctx, ids["preparation"], "cameras.json"))["views"]["source"]
    values = []
    for gen, camera in ((baseline, baseline_cam), (reduced, reduced_cam)):
        focal = gen["focal_length_px"]
        size = gen["source_size_wh"]
        network = gen["network_input_wh"]
        require(np.isfinite(focal) and focal > 0 and size == [camera["width"], camera["height"]] and network == [1536,1536] and
                np.allclose([camera["K"][0][0], camera["K"][1][1]], focal, atol=1e-8, rtol=0), "generator focal telemetry differs from frozen camera")
        values.append({"external_focal_px":focal, "source_width":size[0], "focal_over_width":focal / size[0],
                       "effective_internal_focal_px":focal * network[0] / size[0]})
    b0, r1 = values
    require(b0["source_width"] == 512 and r1["source_width"] == 256 and np.isclose(b0["external_focal_px"], 2*r1["external_focal_px"], atol=1e-8, rtol=0) and
            np.isclose(b0["effective_internal_focal_px"],r1["effective_internal_focal_px"], atol=1e-8, rtol=0), "R1 internal-resize focal scale differs from B0")
    return {"B0":b0, "R1":r1, "network_width":1536}


def products(ctx, ids, owner):
    render, score, evaluation, check, ref = (ids[k] for k in ("render","score","evaluation","recompute","reference"))
    audit = ReadAudit()
    cameras = load_cameras(path(ctx, owner, "cameras.json"), audit)
    rgb = load_source_png(path(ctx, owner, "input.png"), cameras["source"], audit)
    predictions = PredictionBundle(path(ctx, render, "."), audit)
    s, e = load(path(ctx, score, "score.json")), load(path(ctx, evaluation, "evaluation.json"))
    require(predictions.depth_verified and not predictions.failed and predictions.depth_receipt == digest(ctx, render, "depth-probe.json") and
            s["schema"] == "el-score-1" and s["run_id"] == score and e["schema"] == "el-evaluation-1" and e["run_id"] == evaluation and
            s["producer_run_id"] == e["producer_run_id"] == render and e["scoring_run_id"] == score and
            s["depth_verified"] is True and e["depth_verified"] is True and e["depth_validation_receipt_sha256"] == s["depth_validation_receipt_sha256"] == predictions.depth_receipt and
            e["condition_id"] == ctx["condition"] and e["trial_id"] == ctx["quality_trial"] and e["scene_family_id"] == ctx["family"]["scene_family_id"] and
            set(s["views"]) == set(e["views"]) == set(TARGETS) and e["completed_target_views"] == e["planned_target_views"] == len(TARGETS), "invalid condition score/evaluation")
    bound(ctx, score, owner, ("input.png","cameras.json"))
    bound(ctx, score, render, ("prediction.json", *(v + ".npz" for v in VIEWS)))
    bound(ctx, evaluation, owner, ("input.png","cameras.json","reference.json"))
    bound(ctx, evaluation, ref, ("validation/results.json", *(f"reference/{v}.npz" for v in TARGETS)))
    bound(ctx, evaluation, score, ("score.json","feature_schema.json", *(v + ".npz" for v in TARGETS)))
    bound(ctx, evaluation, render, ("prediction.json", *(v + ".npz" for v in VIEWS)))
    for role, key in (("input_png","input.png"), ("cameras","cameras.json")):
        require(s["inputs"][role]["sha256"] == e["inputs"][role]["sha256"] == digest(ctx, owner, key), "score/evaluation used a different condition source")
    require(e["inputs"]["reference_metadata"]["sha256"] == digest(ctx, owner, "reference.json"), "private reference metadata differs")
    counts, source = {}, {}
    for v in VIEWS:
        data = predictions.read(v, cameras[v])
        require(data is not None, f"missing condition prediction {v}")
        with Image.open(path(ctx, render, v + ".png")) as im:
            require(im.format == "PNG" and im.mode == "RGB" and im.size == cameras[v].shape[::-1] and
                    np.array_equal(np.asarray(im), np.rint(255 * np.nan_to_num(data["rgb"])).astype("uint8")), "PNG differs from FP32 prediction")
        if v == "source":
            err = np.abs(data["rgb"] - rgb).mean(axis=-1)
            good = np.isfinite(err)
            source = {"valid_pixels":int(good.sum()),"total_pixels":int(good.size),"mae":float(err[good].mean(dtype=np.float64)) if good.any() else None}
            for item, keys in ((predictions.meta["source_rgb_check"], ("finite_pixels", "pixels")), (s["source_comparison"], ("valid_pixels", "total_pixels"))):
                require(item[keys[0]] == source["valid_pixels"] and item[keys[1]] == source["total_pixels"] and
                        ((source["mae"] is None and item["mae"] is None) or (source["mae"] is not None and item["mae"] is not None and
                         np.isclose(item["mae"], source["mae"], rtol=1e-6, atol=1e-8))), "condition source self-check does not reproduce")
            continue
        se, er = s["views"][v], e["views"][v]
        require(se["status"] == "scored" and er["render_status"] == "completed" and se["score_sha256"] == digest(ctx, score, v + ".npz") and
                er["label_sha256"] == digest(ctx, evaluation, f"labels/{v}.npz") and e["inputs"]["reference/" + v]["sha256"] == digest(ctx, ref, f"reference/{v}.npz"),
                "condition target scores/labels/reference hashes differ")
        with np.load(path(ctx, evaluation, f"labels/{v}.npz"), allow_pickle=False) as labels:
            counts[v] = {}
            for k in ("reference_valid","rgb_evaluable","depth_evaluable"):
                mask = labels[k]
                require(mask.shape == cameras[v].shape and mask.dtype == np.bool_ and type(er[k]) is int and int(mask.sum()) == er[k], "condition mask count differs")
                counts[v][k] = er[k]
            require(not np.any(labels["rgb_evaluable"] & ~labels["reference_valid"]), "evaluable pixels exceed the valid reference")
    board = load(path(ctx, evaluation, "board_v01.json"))
    require(board["run_id"] == evaluation and board["camera_id"] == "v01" and board["board_sha256"] == digest(ctx, evaluation, "board_v01.png"), "missing fixed v01 condition board")
    with Image.open(path(ctx, evaluation, "board_v01.png")) as image:
        image.verify()
    for producer, names in ((owner,("cameras.json",)),(ref,("reference/v01.npz",)),(render,("prediction.json","source.npz","v01.npz")),
                            (score,("v01.npz",)),(evaluation,("metrics.csv","labels/v01.npz"))):
        bound(ctx, check, producer, names)
    proof = load(path(ctx, check, "independent-view-check.json"))
    ratios = proof["ratios_numerator_denominator"]
    v01 = counts["v01"]
    require(proof["passed"] is True and proof["view"] == "v01" and proof["pixel_labels_checked"] == int(np.prod(cameras["v01"].shape)) and
            proof["scalar_l0_witnesses"] == min(17, cameras["v01"].height)*min(17, cameras["v01"].width) and
            proof["signed_camera_z_pixels"] == v01["depth_evaluable"] and ratios["output/prediction_valid_fraction"] == [v01["rgb_evaluable"], v01["reference_valid"]] and
            ratios["output/depth_error_fraction"][1] == v01["depth_evaluable"] and
            {"output/invalid_output_fraction","output/rgb_error_fraction_gt_0.05","L0/high_risk_recall","L0/warning_precision","L0/low_risk_error_rate","L0/overall_low_risk_coverage","L0/unknown_fraction"} <= ratios.keys(),
            "independent condition v01 proof is incomplete")
    return {"target_denominators":counts,"source_self_check":source,"camera_set_sha256":digest(ctx,owner,"cameras.json"),"input_sha256":digest(ctx,owner,"input.png"),
            "depth_probe_sha256":predictions.depth_receipt,"board_sha256":board["board_sha256"],"independent_view_check_sha256":digest(ctx, check, "independent-view-check.json"),
            "ply_sha256":digest(ctx,ids["producer"],"scene.ply"),"views":list(VIEWS)}


def first_chain(ctx, ids, inventory):
    cond = ctx["condition"]
    roles = ["preparation", *(["producer"] if cond == "R1" else []), "render", "score", "evaluation", "recompute"]
    ref, rb, bgen = (ctx["b0_ids"][k] for k in ("reference","readback","generation"))
    disposition = []
    def visit(index, lineage):
        if index == len(roles):
            return lineage
        role = roles[index]
        prep = lineage.get("preparation")
        prod = lineage.get("producer", prep)
        src = prep if cond == "R1" else ref
        base = {ref, rb, prep, lineage.get("render")}
        parents = {"preparation":({ref,rb} if cond == "R1" else {bgen}), "producer":{prep}, "render":{src,prod},
                   "score":base, "evaluation":base | {lineage.get("score")},
                   "recompute":base | {lineage.get("score"),lineage.get("evaluation")}}[role]
        kind = {"preparation":"derivation","producer":"generation","render":"render","score":"model_score","evaluation":"evaluation","recompute":"archive_verify"}[role]
        matches = sorted((node for node in inventory if node["run_kind"] == kind and set(node["parent_run_ids"]) == parents),
                         key=lambda r:(timestamp(r["started_at_utc"]), timestamp(r["ended_at_utc"]), r["run_id"]))
        successes = [m for m in matches if m["execution_status"] == "succeeded"]
        if not successes:
            terminal = [m["run_id"] for m in matches if worker_started(ctx["store"] / "runs" / m["run_id"],m)]
            require(terminal, f"earlier condition lineage is unresolved at {role}: {lineage}")
            disposition.append({"lineage":lineage,"terminal_role":role,"failed_runs":terminal})
        for m in successes:
            candidate = visit(index+1, dict(lineage, **{role:m["run_id"]}))
            if candidate is not None:
                return candidate
        return None
    chosen = visit(0,{})
    require(chosen is not None and all(chosen[k] == ids[k] for k in roles), "explicit condition chain differs from first eligible scientific chain")
    return disposition


def select(ctx, ids, owner, dispatch):
    store, trial = ctx["store"], ctx["quality_trial"]
    dest = store / "trials" / trial / "trial_selection.json"
    with writer_gate(store):
        old = load(dest) if dest.exists() else None
        if old:
            require(old["schema"] == "el-condition-selection-1" and {k:v["run_id"] for k,v in old["selected"].items()} == ids,
                    "immutable condition selection conflicts")
            rids = [r["run_id"] for r in old["attempts"]]
        else:
            rids = []
            for path_ in sorted((store / "runs").glob("*/run.json")):
                r = load(path_)
                if r["trial_id"] == trial and r["experiment_id"] == QUALITY:
                    rids.append(r["run_id"])
        attempts = [ctx["verified"].get(rid)["run"] for rid in rids]
        terminal = first_chain(ctx, ids, attempts)
        evidence = products(ctx, ids, owner)
        if ctx["condition"] == "Q1":
            evidence["q1_field_check"] = q1_fields(ctx, ids)
        else:
            evidence["r1_focal_scaling"] = r1_focal_scaling(ctx, ids)
        def pin_asset(rid, name):
            return dict(run_pin(ctx["verified"].get(rid)), path="payload/" + name, sha256=digest(ctx,rid,name))
        receipts = transfer_evidence(store, {ids[k]:ctx["verified"].get(ids[k]) for k in ("producer","render")})
        selected = {role:dict(run_pin(ctx["verified"].get(rid)),run_kind=ctx["verified"].get(rid)["run"]["run_kind"]) for role,rid in ids.items()}
        inventory = [{**{k:r.get(k) for k in ("run_id","run_kind","parent_run_ids","retry_of_run_id","execution_status","started_at_utc","ended_at_utc","failure_reason")},
                      "manifest_sha256":ctx["verified"].get(r["run_id"])["manifest_sha256"]} for r in attempts]
        out = {"schema":"el-condition-selection-1", "selection_status":"selected", "rule_version":"el-condition-first-scientific-1",
               "frozen_at_utc":old["frozen_at_utc"] if old else utc(), "selection_reason":"First completed scientific chain in recorded start, completion and run-ID order. All six targets, measured calibration and independent v01 recomputation verified.",
               "trial_id":trial, "experiment_id":QUALITY, "condition_id":ctx["condition"], **{k:ctx["family"][k] for k in ("scene_id","scene_family_id","dataset_id","dataset_hash","split")},
               "selected":selected,"attempts":inventory,"terminal_dispositions":terminal,"method_freeze_sha256":ctx["release"]["method_freeze_sha256"],
               "b0_selection_sha256":ctx["b0_selection_sha256"],"handoff_sha256":dispatch["handoff_sha256"],"evidence":evidence,
               "sources":{k:pin_asset(owner,n) for k,n in (("input_png","input.png"),("cameras","cameras.json"),("reference_metadata","reference.json"))}}
        if old:
            require(all(r in receipts[rid] for rid, rows in old["transfer_receipts"].items() for r in rows), "preserved condition transfer receipt differs")
            out["transfer_receipts"] = old["transfer_receipts"]
        else:
            out["transfer_receipts"] = receipts
        publish(dest, encoded(out))
    return {"trial_id":trial,"selection":str(dest),"sha256":sha(dest),"status":"identical" if old else "selected"}
