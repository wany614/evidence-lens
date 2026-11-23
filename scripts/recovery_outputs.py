"""Verify the actual Windows recovery optimizer, checkpoints and renderer, and project their original output index."""

from collections import Counter
import hashlib
import json
import math
import os
import zipfile

import numpy as np
from PIL import Image

from evidence_lens._bundle import encoded, load, local_path, sha
from evidence_lens._snapshot import source_name
from evidence_lens._verify import require
from evidence_lens.stage3_gaussians import PARAMS, digest_values
from recovery_inputs import INIT, REG, SETS, STUDY, TARGETS, located, member, validate_regression_freeze
from recovery_receive import authenticate_native, check_grid, contained_asset

WINDOWS_PARAM = dict(zip(PARAMS, ("means", "scales", "quats", "opacities", "colors"), strict=True))


def pinned_member(store, p):
    path = member(store, p, p["member"])
    require(
        sha(path) == p["sha256"] and ("bytes" not in p or path.stat().st_size == p["bytes"]),
        "Changed pinned " + p["member"],
    )
    return path


def member_pin(pin, row, name):
    return {
        **{k: pin[k] for k in ("run_id", "manifest_sha256")},
        "member": name,
        "sha256": row["sha256"],
        "bytes": row["bytes"],
    }


def verify_code(pack, root, identity):
    with zipfile.ZipFile(pack) as z:
        body = json.loads(z.read("SOURCE-MANIFEST.json"))
        listing = body["files"]
        expected_tree = hashlib.sha256(
            json.dumps(listing, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        require(body["tree_sha256"] == expected_tree, "Different supplied Windows source listing")
        want = {n: meta for n, meta in listing.items() if source_name(n)}
        actual = load(root / "code_snapshot.json")
        rows = {r["path"]: {k: r[k] for k in ("bytes", "sha256")} for r in actual["files"]}
        require(
            rows == want
            and actual["snapshot_sha256"] == identity["source_sha256"]
            and identity["gsplat"] == "1.5.3",
            "Pinned Windows source and its numeric worker identity differ",
        )
        for name, desc in want.items():
            data = z.read(name)
            require(
                len(data) == desc["bytes"] and hashlib.sha256(data).hexdigest() == desc["sha256"],
                "Supplied worker member differs",
            )
    return expected_tree


def result_phase(step, recipe):
    if step == 0:
        return "initial"
    t = recipe["topology"]
    return "warmup" if step < t["start"] else "refinement" if step < t["stop"] else "fixed_topology"


def verified_semantics(store, req, worker, tree):
    """Bind the admitted renderer to its actual two-layer CUDA readback and Mac verification."""
    resource = load(pinned_member(store, req["resource_probe"]))
    probe_pin = req["depth_probe"]
    measured = load(pinned_member(store, probe_pin))
    admission = load(member(store, req["recipe"], "payload/admission.json"))
    optimizer = load(member(store, resource["raw_runs"]["continuous"], "payload/reconstruction.json"))
    require(
        set(probe_pin) == {"run_id", "manifest_sha256", "member", "sha256"}
        and probe_pin["member"] == "payload/ed-probe.json"
        and resource["raw_runs"]["projection"] == {k: probe_pin[k] for k in ("run_id", "manifest_sha256")}
        and resource["schema"] == "el-recovery-resource-probe-1"
        and admission["inputs"]["resource_probe"] == req["resource_probe"]
        and admission["inputs"]["worker_source"] == req["worker"]
        and resource["worker_zip_sha256"] == req["worker"]["sha256"]
        and resource["worker_tree_sha256"] == tree
        and resource["actual_worker_snapshots"]["projection"] == worker["source_sha256"]
        and optimizer["identity"]["worker"] == worker
        and resource["passed"] is True
        and measured["passed"] is True
        and measured == resource["geometry"]["actual"]
        and measured["resources"]["gsplat"] == worker["gsplat"] == "1.5.3",
        "The admitted worker has no matching verified source, colour, opacity and expected camera-Z probe",
    )
    return {
        "depth_verified": True,
        "depth_probe": probe_pin,
        "depth_semantics": "expected_camera_z_m",
        "rgb_encoding": "display_srgb_0_1",
        "alpha_semantics": "accumulated_opacity",
    }


def calibration_readback(source, rendering, ids):
    """Hash the calibrated physical quantities independently of either camera JSON's text formatting."""

    def physical(camera):
        record = {key: np.asarray(camera[key], dtype=np.float64).tolist() for key in ("K", "c2w", "clip_m")}
        record.update(width=camera["width"], height=camera["height"])
        return hashlib.sha256(encoded(record)).hexdigest()

    sc, rc = source["views"], rendering["views"]
    require(set(sc) == set(ids) and set(rc) == set(ids + TARGETS), "Wrong camera subset")
    table = {s: physical(sc[s]) for s in ids}
    require(table == {s: physical(rc[s]) for s in ids}, "Source camera changed in render calibration")
    endpoints = {s: np.asarray(sc[s]["c2w"], dtype=np.float64).tolist() for s in ("s00", "s03")}
    return table, hashlib.sha256(encoded(endpoints)).hexdigest()


def checkpoint_audit(path, *, genid, identity, original, recipe, step, n, counts, calibration=None):
    """Read only tensors and standard types with Torch's restricted CPU-only loader."""
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=True)
    ids = SETS[identity["condition"]]
    require(
        ck["schema"] == "el-recovery-checkpoint-1"
        and ck["identity"] == identity
        and ck["producer_run_id"] == genid
        and ck["step"] == step
        and ck["horizon"] == 5000
        and ck["phase"] == result_phase(step, recipe)
        and ck["recipe"] == recipe
        and ck["source_order"] == list(ids)
        and ck["next_source_index"] == step % len(ids)
        and ck["counts"] == counts
        and ck["actual_method"] == original["actual_method"]
        and math.isclose(ck["scene_scale_m"], original["scene_scale_m"], rel_tol=1e-6, abs_tol=1e-8),
        "Different optimizer state identity, geometry, sampler or phase",
    )
    if calibration is not None:
        physical, frame_hash = calibration
        require(
            ck["source_calibration"] == physical and ck["endpoint_frame_sha256"] == frame_hash,
            "Different source calibration in optimizer checkpoint",
        )
    require(
        set(ck["rng"]) == {"torch_cpu", "torch_cuda0", "python", "numpy"}
        and set(ck["params"]) == set(WINDOWS_PARAM.values()) == set(ck["adam"])
        and isinstance(ck["strategy"], dict),
        "Incomplete full state for continuation",
    )
    nums = {}
    for field, tail in zip(PARAMS, ((3,), (3,), (4,), (), (3,)), strict=True):
        key = WINDOWS_PARAM[field]
        t = ck["params"][key]
        require(
            t.dtype == torch.float32
            and t.shape == (n, *tail)
            and t.device.type == "cpu"
            and bool(torch.isfinite(t).all()),
            "Wrong FP32 Gaussian parameter precision/shape",
        )
        nums[field] = t.detach().numpy()
        opt = ck["adam"][key]
        require(
            len(opt["param_groups"]) == 1 and len(opt["param_groups"][0]["params"]) == 1,
            "Unbound Adam parameter group",
        )
        st = opt["state"]
        if step == 0:
            require(not st, "Adam moments unexpectedly present before step one")
        else:
            require(
                len(st) == 1 and opt["param_groups"][0]["params"][0] in st, "Different Adam state parameter"
            )
            state = next(iter(st.values()))
            require(
                set(state) == {"step", "exp_avg", "exp_avg_sq"}
                and state["step"].item() == step
                and all(
                    state[m].shape == t.shape
                    and state[m].dtype == t.dtype
                    and bool(torch.isfinite(state[m]).all())
                    for m in ("exp_avg", "exp_avg_sq")
                ),
                "Unusable Adam moments",
            )
    digest = digest_values(nums)
    if step == 0:
        require(
            n == original["N"] and digest == original["parameter_sha256"],
            "Initial parameters differ from sealed Mac geometry",
        )
    return digest


def trace_summary(path, reconstruction, ids, recipe):
    totals = Counter()
    counts = {s: 0 for s in ids}
    nonzero = Counter()
    last = None
    first = reconstruction["first_completed_step"]
    end = reconstruction["last_completed_step"]
    require(reconstruction["trace_sha256"] == sha(path), "Optimizer trace differs")
    with path.open() as log:
        for step, line in enumerate(log, first + 1):
            row = json.loads(line)
            src = ids[(step - 1) % len(ids)]
            gr = row["gradient_l2"]
            counts = {s: (step + len(ids) - 1 - i) // len(ids) for i, s in enumerate(ids)}
            require(
                row["step"] == step
                and row["source"] == src
                and row["pixels"] == 65536
                and row["source_counts"] == counts
                and row["phase"] == result_phase(step, recipe)
                and 1 <= row["n_before"] <= recipe["topology"]["cap"]
                and 1 <= row["n_after"] <= recipe["topology"]["cap"]
                and (last is None or row["n_before"] == last)
                and set(gr) == set(WINDOWS_PARAM.values())
                and all(math.isfinite(x) and x >= 0 for x in gr.values()),
                "Source sampler, recorded gradient or Gaussian population differs at update " + str(step),
            )
            totals[src] += 1
            nonzero[src] += int(any(v > 0 for v in gr.values()))
            last = row["n_after"]
    require(
        sum(totals.values()) == end - first == reconstruction["updates_in_process"]
        and counts == reconstruction["sampling_counts"],
        "Missing or mismatched optimizer updates",
    )
    return {
        "source_updates_in_this_process": dict(totals),
        "updates_with_nonzero_gradient": dict(nonzero),
        "actual_source_pixels": sum(totals.values()) * 65536,
        "final_gaussians": last,
        "trace_sha256": sha(path),
    }


def actual_inputs(root):
    return {
        r["role"]: {k: r["files"][0][k] for k in ("sha256", "bytes")}
        for r in load(root / "inputs.json")["inputs"]
        if len(r["files"]) == 1
    }


def png_readback(path, native):
    with np.load(native, allow_pickle=False) as z:
        rgb = z["rgb"]
        expected = np.floor(rgb * 255 + 0.5).astype(np.uint8)
        alpha = z["alpha"]
        depth = z["z"]
        foreground = alpha > 0
        invalid_foreground = int(np.count_nonzero(foreground & ~(np.isfinite(depth) & (depth > 0))))
        invalid_background = int(np.count_nonzero(~foreground & ~np.isnan(depth)))
        require(
            np.isfinite(alpha).all()
            and np.all((alpha >= 0) & (alpha <= 1))
            and invalid_foreground == invalid_background == 0,
            f"{native.name} has {invalid_foreground}/{int(foreground.sum())} positive-alpha pixels with invalid "
            f"expected camera-Z and {invalid_background}/{int((~foreground).sum())} zero-alpha pixels with non-NaN depth",
        )
    with Image.open(path) as img:
        require(
            img.format == "PNG"
            and img.mode == "RGB"
            and img.size == (256, 256)
            and np.array_equal(np.asarray(img), expected),
            "Issued display PNG differs from its source float RGB",
        )


def original_case(store, req, source_handoff):
    spec = req["original_case"]
    obj = req["object_id"]
    cond = req["condition"]
    require(
        set(spec) == {"run_id", "manifest_sha256", "member", "sha256"},
        "Original deployment CASE needs its exact four-part identity",
    )
    body = load(pinned_member(store, spec))
    ids = SETS[cond]
    require(
        body["schema"] == "el-stage3-deploy-case-1"
        and body["condition"] == cond
        and body["generator_source_ids"] == list(ids)
        and body["target_ids"] == list(TARGETS),
        "Different original deployment CASE",
    )
    parent = spec["member"].rsplit("/", 1)[0]
    roles = [
        (body["source_cameras"], source_handoff["source_camera_sha256"]),
        (body["render_cameras"], source_handoff["render_camera_sha256"]),
    ]
    roles.extend((body["source_images"][s], source_handoff["source_rgb_sha256"][s]) for s in ids)
    for evidence, target in roles:
        relative = local_path(evidence["path"])
        data = member(store, spec, parent + "/" + relative)
        require(
            evidence["sha256"] == sha(data) == target and data.stat().st_size == evidence["bytes"],
            "Source pixels/calibration differ from historical deployment",
        )
    if obj == "scene_057_offset_bell_cage":
        require(
            spec
            == {
                "run_id": "20261123T193505072501Z-6046afc1fce4480b",
                "manifest_sha256": "0b5ce75ed8670762b035dcf7b9b5820ec47a20a86ac92db08246fead9e8af33a",
                "member": "payload/cases/M4/case.json",
                "sha256": "8ab6d51ec5c348fc61121478631a7ead7a9299659337cc8a8a5811922ec1d598",
            },
            "Wrong released authored development control",
        )
    else:
        q = req["original_provenance"]
        provenance = load(pinned_member(store, q))
        require(
            q["run_id"] == spec["run_id"]
            and q["manifest_sha256"] == spec["manifest_sha256"]
            and q["member"] == "payload/case-provenance.json"
            and provenance["run_id"] == spec["run_id"]
            and provenance["object_id"] == obj
            and provenance["condition"] == cond
            and provenance["case_sha256"] == spec["sha256"]
            and provenance["optimization_steps"] == 256,
            "Original photometric CASE not in the same object, condition and D1 main arm",
        )
    return spec


def normalize(store, req):
    obj, cond = req["object_id"], req["condition"]
    require(
        cond in SETS
        and (obj in ("set1", "set6") or (obj == "scene_057_offset_bell_cage" and cond == "M4")),
        "Unknown registered recovery cell",
    )
    stop = 5000
    if obj == "set6":
        validate_regression_freeze(store, req.get("freeze"))
        freeze = load(pinned_member(store, req["freeze"]))
        stop = freeze["stop_step"]
        require(freeze["recipe_sha256"] == req["recipe"]["sha256"], "Different frozen regression recipe")
    cell = store / "recovery/inputs" / INIT / obj / cond / "selection.json"
    require(sha(cell) == req["input_selection_sha256"], "Wrong original source selection")
    sel = load(cell)
    init = load(member(store, sel["init_pin"], "payload/init.json"))
    hand = load(member(store, sel["handoff_pin"], "payload/handoff.json"))
    original_case(store, req, hand)
    protocol = load(pinned_member(store, req["registration"]))
    recipe = load(pinned_member(store, req["recipe"]))
    require(
        {k: req["registration"][k] for k in REG} == REG
        and protocol["protocol_sha256"] == STUDY
        and req["registration"]["member"] == "payload/registration.json"
        and protocol["registered_files"]["protocol/recovery-initialization-v1.json"] == INIT
        and init["object_id"] == hand["object_id"] == obj
        and init["condition"] == hand["condition"] == cond
        and init["init_npz_sha256"] == sel["init_npz_sha256"],
        "Wrong source registration",
    )
    ids = SETS[cond]
    gp = req["generation"]
    rp = req["render"]
    rootg, rootr = located(store, gp), located(store, rp)
    science_g = load(member(store, gp, "config.resolved.yaml"))
    science_r = load(member(store, rp, "config.resolved.yaml"))
    g = load(member(store, gp, "payload/reconstruction.json"))
    r = load(member(store, rp, "payload/outputs.json"))
    issued = load(member(store, gp, "payload/issued.json"))
    ident = g["identity"]
    require(
        science_g["science"] == science_r["science"] == ident == issued["identity"]
        and all(r[k] == v for k, v in ident.items())
        and ident["object_id"] == obj
        and ident["condition"] == cond
        and ident["source_camera_sha256"] == hand["source_camera_sha256"]
        and ident["recipe_sha256"] == req["recipe"]["sha256"]
        and ident["generator_source_ids"] == list(ids)
        and ident["source_rgb_sha256"] == hand["source_rgb_sha256"]
        and ident["init_sha256"] == sel["init_npz_sha256"]
        and ident["init_json_sha256"] == sel["init_meta_sha256"]
        and r["render_camera_sha256"] == hand["render_camera_sha256"],
        "Optimizer and renderer used another cell, recipe, source input, geometry or camera",
    )
    require(
        r["schema"] == "el-recovery-outputs-1"
        and g["schema"] == "el-recovery-reconstruction-1"
        and g["run_id"] == issued["run_id"] == r["reconstruction_run_id"] == gp["run_id"]
        and science_r["generation"] == gp
        and gp in load(rootr / "inputs.json")["parents"]
        and g["checkpoints"] == issued["checkpoints"]
        and g["job_stop_step"] == science_g["planned_stop_step"] == stop
        and g["recipe_stop_step"] == g["schedule_horizon"] == 5000
        and r["depth_semantics"] == "opacity-normalized expected camera Z in metres"
        and r["rgb_encoding"] == "display_srgb_float32",
        "Wrong actual checkpoint producer, schedule, depth or colour semantics",
    )
    job = req["job"]
    jobdata = load(pinned_member(store, job))
    require(
        sha(pinned_member(store, job)) == science_g["cell_sha256"] == science_r["cell_sha256"]
        and jobdata["object_id"] == obj
        and jobdata["condition"] == cond
        and jobdata["files"] == hand["files"]
        and jobdata["recipe_sha256"] == req["recipe"]["sha256"],
        "Windows stage not bound to registered cell ticket",
    )
    pack = pinned_member(store, req["worker"])
    tree = verify_code(pack, rootg, ident["worker"])
    verify_code(pack, rootr, ident["worker"])
    semantics = verified_semantics(store, req, ident["worker"], tree)
    expect_g = {
        "recipe": ident["recipe_sha256"],
        "init": ident["init_sha256"],
        "initmeta": ident["init_json_sha256"],
        "cameras": ident["source_camera_sha256"],
        **ident["source_rgb_sha256"],
    }
    if science_g.get("resume_origin"):
        expect_g["resume"] = science_g["resume_origin"]["checkpoint_sha256"]
    require(
        science_g["role_hashes"] == expect_g == {k: v["sha256"] for k, v in actual_inputs(rootg).items()}
        and science_r["role_hashes"] == {k: v["sha256"] for k, v in actual_inputs(rootr).items()}
        and science_r["role_hashes"]
        == {
            "cameras": hand["render_camera_sha256"],
            **{"c" + s: t["checkpoint_sha256"] for s, t in r["checkpoints"].items()},
        },
        "Recorded worker inputs include different pixels, cameras or parameters",
    )
    calibration = calibration_readback(
        load(member(store, sel["handoff_pin"], "payload/handoff/source_cameras.json")),
        load(member(store, sel["handoff_pin"], "payload/handoff/render_cameras.json")),
        ids,
    )
    require(
        g["endpoint_frame_sha256"] == r["endpoint_frame_sha256"] == calibration[1]
        and math.isclose(r["scene_scale_m"], init["scene_scale_m"], rel_tol=1e-6, abs_tol=1e-8),
        "Returned camera frame differs from source observations",
    )
    traces = trace_summary(member(store, gp, "payload/steps.jsonl"), g, ids, recipe)
    normalized = {
        **r,
        **semantics,
        "run_id": os.environ["EL_RUN_ID"],
        "protocol_sha256": STUDY,
        "schedule_steps": g["schedule_horizon"],
        "completed_steps": g["last_completed_step"],
        "checkpoints": {},
        "original_generation": gp,
        "original_render": rp,
        "input_handoff": sel["handoff_pin"],
        "original_case": req["original_case"],
        "original_case_provenance": req.get("original_provenance"),
        "job": job,
        "recipe": req["recipe"],
    }
    check_grid({**normalized, "checkpoints": r["checkpoints"]}, cond, stop, full_curve=obj != "set6")
    rootfiles = {m["path"]: m for m in load(rootr / "manifest.json")["files"]}
    diagnostics = {}
    for k, point in r["checkpoints"].items():
        step = int(k)
        origin = science_r["artifact_origins"]["c" + k]
        producer = {n: origin[n] for n in ("run_id", "manifest_sha256")}
        require(
            producer in load(rootr / "inputs.json")["parents"]
            and origin["execution_status"] == "succeeded"
            and producer["run_id"] == point["producer_run_id"]
            and origin["member"] == f"payload/checkpoints/step-{step:04d}.pt",
            "Checkpoint render does not have the indicated scientific parent",
        )
        proof = load(member(store, producer, "payload/issued.json"))
        source = proof["checkpoints"][k]
        params = member(store, producer, origin["member"])
        require(
            sha(params) == source["checkpoint_sha256"] == point["checkpoint_sha256"]
            and params.stat().st_size == source["bytes"]
            and proof["identity"] == ident
            and point["gaussians"] == source["gaussians"]
            and step == source["step"]
            and point["phase"] == source["phase"] == result_phase(step, recipe),
            "Different checkpoint or phase",
        )
        cnt = {s: (step + len(ids) - 1 - i) // len(ids) for i, s in enumerate(ids)}
        parameter_sha = checkpoint_audit(
            params,
            genid=producer["run_id"],
            identity=ident,
            original=init,
            recipe=recipe,
            step=step,
            n=point["gaussians"],
            counts=cnt,
            calibration=calibration,
        )
        views = {}
        counts = {}
        for view, frame in point["views"].items():
            raw = contained_asset(rootr, rootr / "payload", frame, rootfiles)
            img = contained_asset(
                rootr,
                rootr / "payload",
                {"path": frame["png_path"], "sha256": frame["png_sha256"], "bytes": frame["png_bytes"]},
                rootfiles,
            )
            require(
                frame["path"] == f"step-{step:04d}/{view}.npz"
                and frame["png_path"] == f"step-{step:04d}/{view}.png",
                "Wrong returned checkpoint view",
            )
            counts[view] = authenticate_native(raw)
            png_readback(img, raw)
            views[view] = {
                **member_pin(rp, frame, "payload/" + frame["path"]),
                "display_png": member_pin(
                    rp,
                    {"sha256": frame["png_sha256"], "bytes": frame["png_bytes"]},
                    "payload/" + frame["png_path"],
                ),
            }
        normalized["checkpoints"][k] = {
            **point,
            "views": views,
            "parameter_sha256": parameter_sha,
            "checkpoint": member_pin(
                producer,
                {"sha256": point["checkpoint_sha256"], "bytes": params.stat().st_size},
                origin["member"],
            ),
        }
        diagnostics[k] = {
            "cpu_restricted_checkpoint_readback": {
                "five_finite_fp32_tensors": True,
                "five_adam_groups_and_moments_checked": True,
                "sampler_phase_counts_rng_and_geometry_checked": True,
            },
            "views": counts,
            "gaussians": point["gaussians"],
            "parameter_sha256": parameter_sha,
        }
    proof = {
        "schema": "el-recovery-mac-return-audit-1",
        "run_id": os.environ["EL_RUN_ID"],
        "passed": True,
        "object_id": obj,
        "condition": cond,
        "input_selection_sha256": req["input_selection_sha256"],
        "windows_worker_tree_sha256": tree,
        "depth_probe": semantics["depth_probe"],
        "resource_probe": req["resource_probe"],
        "registration": req["registration"],
        "recipe": req["recipe"],
        "original_generation": gp,
        "original_render": rp,
        "actual_initial_parameter_sha256": init["parameter_sha256"],
        "endpoint_frame_sha256": calibration[1],
        "source_calibration": calibration[0],
        "source_fit": traces,
        "checkpoints": diagnostics,
    }
    return normalized, proof
