"""Freeze a truth-isolated held-out condition or timing handoff from its actual selected B0 chain."""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from evidence_lens._bundle import RUN_ID, encoded, load, local_path, sha
from evidence_lens._index import writer_gate
from evidence_lens._verify import require
from formal_lineage import VIEWS, VerifiedRuns, family_record, heldout_method, run_pin
from make_handoff import ROOT, publish, sha_bytes, zip_bytes
from select_trial import chain_ids, scientific_gates, transfer_evidence

QUALITY = "p6-heldout-conditions"
TIMING = "p6-timing-repeats"
EXTRA_WORKER = ("scripts/condition_stage.py", "scripts/stress_condition.py", "scripts/inspect_sharp_ply.py")


def context(store, catalog, freeze_file, method_file, family_id, condition):
    family = family_record(catalog, freeze_file, family_id)
    require(family["split"] == "test" and condition in {"B0", "R1", "Q1"}, "P6 conditions apply only to held-out families")
    data = load(catalog)
    row = next(r for r in data["families"] if r["scene_family_id"] == family_id)
    verified = VerifiedRuns(store)
    release = heldout_method(verified, family, method_file, catalog.parent / "study-v1.json", freeze_file)
    b0_file = store / "trials" / row["trial_id"] / "trial_selection.json"
    b0 = load(b0_file)
    require(b0.get("schema") == "el-p2-trial-selection-1" and b0["selection_status"] == "selected" and
            b0["condition_id"] == "B0" and all(b0[k] == family[k] for k in ("trial_id", "scene_id", "scene_family_id")),
            "P6 requires this formal family's individually selected B0 chain")
    ids = {key: value["run_id"] for key, value in b0["selected"].items()}
    records = {rid: verified.get(rid) for rid in ids.values()}
    require(chain_ids(records, ids["recompute"]) == ids and
            all(records[rid]["manifest_sha256"] == b0["selected"][role]["manifest_sha256"] for role, rid in ids.items()),
            "selected baseline chain manifest or parent graph differs")
    for role, rid in ids.items():
        meta = records[rid]["run"]
        require(all(meta[k] == family[k] for k in ("dataset_id", "dataset_hash", "scene_family_id", "scene_id", "trial_id", "split")),
                "baseline is outside the frozen held-out family")
    evidence = scientific_gates(store, records, ids, transfer_evidence(store, records), b0["experiment_id"])
    require(all(b0["evidence"][k] == v for k, v in evidence.items() if k != "transfer_receipts") and
            load(store / "runs" / ids["evaluation"] / "config.resolved.yaml").get("method_freeze_sha256") == release["method_freeze_sha256"],
            "selected baseline used different scientific evidence or held-out release")
    return {"store":store, "verified":verified, "catalog":catalog, "method_file":method_file, "release":release,
            "family":family, "condition":condition, "quality_trial":row["trials"][condition], "baseline":b0,
            "b0_ids":ids, "b0_selection_sha256":sha(b0_file), "first_test":min(r["scene_family_id"] for r in data["families"] if r["split"] == "test")}


def validate_r1(ctx, rid):
    v, family = ctx["verified"], ctx["family"]
    result = v.get(rid, "derivation")
    run = result["run"]
    ref, rb = (ctx["b0_ids"][role] for role in ("reference", "readback"))
    require(run["experiment_id"] == ctx.get("quality_experiment", QUALITY) and run["trial_id"] == ctx["quality_trial"] and run["condition_id"] == "R1"
            and all(run[k] == family[k] for k in ("dataset_id", "dataset_hash", "scene_id", "scene_family_id", "split")) and
            set(run["parent_run_ids"]) == {ref, rb}, "R1 source preparation belongs to a different family or condition")
    root = ctx["store"] / "runs"
    payload, original = root / rid / "payload", root / ref / "payload"
    cfg = load(root / rid / "config.resolved.yaml")
    require(cfg["method_freeze_sha256"] == ctx["release"]["method_freeze_sha256"] and cfg["b0_selection_sha256"] == ctx["b0_selection_sha256"],
            "R1 derivative did not use the selected B0 and method")
    if "opening_evidence" in ctx:
        roles = {i["role"]:[f["sha256"] for f in i["files"]] for i in result["inputs"]["inputs"]}
        require(roles.get("stage2-opening") == [ctx["opening_evidence"]["opening_sha256"]]
                and roles.get("method-freeze") == [ctx["release"]["method_freeze_sha256"]],
                "Stage-two R1 preparation did not bind the exact opening and method")
    raw_ref, meta = load(original / "reference.json"), load(payload / "source_derivation.json")
    require(meta["schema"] == "el-r1-source-1" and meta["producer_run_id"] == rid and meta["condition_id"] == "R1" and
            meta["parent_input_sha256"] == raw_ref["input_sha256"] and meta["parent_camera_sha256"] == raw_ref["camera_sha256"] and
            meta["input_sha256"] == sha(payload / "input.png") and meta["camera_sha256"] == sha(payload / "cameras.json") and
            meta["parent_reference_sha256"] == sha(original / "reference.json"), "R1 derivation source hashes differ")
    # Integer quotient/remainder is independent of the producer's float rint.
    with Image.open(original / "input.png") as before, Image.open(payload / "input.png") as after:
        require(before.size == (512, 512) and after.size == (256, 256) and before.mode == after.mode == "RGB", "R1 source dimensions differ")
        raw = np.asarray(before, dtype=np.uint16)
        sums = raw[0::2,0::2] + raw[0::2,1::2] + raw[1::2,0::2] + raw[1::2,1::2]
        quotient, remainder = np.divmod(sums, 4)
        rounded = quotient + ((remainder > 2) | ((remainder == 2) & ((quotient & 1) == 1)))
        require(np.array_equal(rounded, np.asarray(after)), "R1 independent integer pixel reduction differs")
    before_cam, after_cam = load(original / "cameras.json"), load(payload / "cameras.json")
    first, second = before_cam["views"]["source"], after_cam["views"]["source"]
    edge, centre = (np.array(first[k], dtype=np.float64) for k in ("K_edge", "K"))
    edge[:2] *= .5
    centre[:2,:2] *= .5
    centre[:2,2] = (centre[:2,2] + .5) * .5 - .5
    exempt = {"width", "height", "K_edge", "K"}
    require((first["width"], first["height"], second["width"], second["height"]) == (512,512,256,256) and
            {k:v for k,v in first.items() if k not in exempt} == {k:v for k,v in second.items() if k not in exempt} and
            np.allclose(second["K_edge"], edge, atol=1e-10, rtol=0) and np.allclose(second["K"], centre, atol=1e-10, rtol=0) and
            set(before_cam["views"]) == set(after_cam["views"]) == set(VIEWS) and
            all(before_cam["views"][v] == after_cam["views"][v] for v in VIEWS if v != "source") and
            {k:v for k,v in before_cam.items() if k != "views"} == {k:v for k,v in after_cam.items() if k != "views"},
            "R1 half-size intrinsics, source pose or unchanged target camera geometry differs")
    expected = dict(raw_ref, input_sha256=meta["input_sha256"], camera_sha256=meta["camera_sha256"], source_condition="R1",
                    source_derivation_sha256=sha(payload / "source_derivation.json"), parent_reference_sha256=meta["parent_reference_sha256"])
    private_ref = load(payload / "reference.json")
    scope = private_ref.pop("source_ground_truth_scope", None)
    require(isinstance(scope, str) and bool(scope) and private_ref == expected, "R1 private reference does not preserve the B0 targets and validation")
    return result


def worker_source(ctx):
    directory = ctx["store"] / "formal" / ctx["family"]["dataset_hash"]
    base = directory / "worker-source.zip"
    require(sha(base) == load(directory / "worker-source.json")["sha256"], "frozen B0 worker source changed")
    destination = directory / "conditions" / "worker-source.zip"
    receipt = destination.with_suffix(".json")
    with writer_gate(ctx["store"]):
        if receipt.is_file():
            raw = destination.read_bytes()
            rec = load(receipt)
            require(sha_bytes(raw) == rec["sha256"] and rec["b0_worker_sha256"] == sha(base), "frozen condition worker differs")
        else:
            with zipfile.ZipFile(base) as z:
                members = {local_path(n): z.read(n) for n in z.namelist()}
            for n in EXTRA_WORKER:
                members[n] = (ROOT / n).read_bytes()
            raw = zip_bytes(members)
            publish(destination, raw)
            publish(receipt, encoded({"schema":"el-condition-worker-1", "b0_worker_sha256":sha(base), "sha256":sha_bytes(raw)}))
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        return {"code/" + local_path(n): z.read(n) for n in z.namelist()}, sha_bytes(raw)


def make_job(ctx, preparation, repeat, out):
    family, cond, ids = ctx["family"], ctx["condition"], ctx["b0_ids"]
    require(cond in {"R1", "Q1"} or repeat is not None, "formal B0 uses its original worker; only B0 timing uses this controller")
    selection = None
    if repeat is not None:
        require(type(repeat) is int and 1 <= repeat <= 3 and family["scene_family_id"] == ctx["first_test"],
                "timing requires a registered repeat on the lexicographically first test family")
        warm_path = ctx["store"] / "trials" / ctx["quality_trial"] / "trial_selection.json"
        warm = load(warm_path)
        require(warm["selection_status"] == "selected" and warm["condition_id"] == cond and warm["scene_family_id"] == family["scene_family_id"],
                "timing repeats require the same condition's selected warm-up")
        if cond != "B0":
            require(warm.get("schema") == "el-condition-selection-1" and warm["method_freeze_sha256"] == ctx["release"]["method_freeze_sha256"] and
                    warm["b0_selection_sha256"] == ctx["b0_selection_sha256"], "condition warm-up belongs to another release or baseline")
        if cond == "R1":
            require(preparation == warm["selected"]["preparation"]["run_id"], "R1 timing must reuse the selected warm-up source preparation")
        selection = {"trial_id":ctx["quality_trial"], "selection_sha256":sha(warm_path), "render":warm["selected"]["render"]["run_id"]}
    require((cond == "R1") == bool(preparation), "only R1 requires its explicit source preparation run")
    source = validate_r1(ctx, preparation) if preparation else ctx["verified"].get(ids["reference"], "scene_render")
    code, worker_sha = ctx["worker_source"](ctx) if "worker_source" in ctx else worker_source(ctx)
    source_path = ctx["store"] / "runs" / source["run"]["run_id"] / "payload"
    trial = ctx["quality_trial"] if repeat is None else f'{ctx["quality_trial"]}-timing-{repeat:02}'
    ref = ctx["verified"].get(ids["reference"])
    gen = ctx["verified"].get(ids["generation"], "generation")
    identity = {k: family[k] for k in ("dataset_id", "dataset_hash", "split", "scene_family_id", "scene_id", "scene_version")}
    identity.update(trial_id=trial, condition_id=cond, scene_hash=ctx["baseline"]["evidence"]["scene_sha256"],
                    camera_set_hash=source["artifacts"]["payload/cameras.json"]["sha256"])
    job = {"schema":"el-condition-handoff-1", "identity":identity, "experiment_id":TIMING if repeat else ctx.get("quality_experiment", QUALITY), "trial_id":trial,
           "condition":cond, "reference":run_pin(ref), "readback":run_pin(ctx["verified"].get(ids["readback"])), "source":run_pin(source),
           "baseline_generation":run_pin(gen), "baseline_ply_sha256":gen["artifacts"]["payload/scene.ply"]["sha256"],
           "b0_selection_sha256":ctx["b0_selection_sha256"], "method_freeze_sha256":ctx["release"]["method_freeze_sha256"],
           "worker_source_sha256":worker_sha, "timing_repeat":repeat, "warmup":selection}
    if "opening_evidence" in ctx:
        require(repeat is None and cond in {"R1","Q1"}, "Stage-two jobs must be quality R1/Q1")
        job.update(schema="el-stage2-condition-handoff-1", opening=ctx["opening_evidence"], baseline_trial_id=ctx["family"]["trial_id"])
    members = {**code, **{n:(source_path / n).read_bytes() for n in ("input.png", "cameras.json")}}
    job["files"] = {n:sha_bytes(data) for n, data in members.items()}
    members["handoff.json"] = encoded(job)
    digest = sha_bytes(members["handoff.json"])
    raw = zip_bytes(members)
    with writer_gate(ctx["store"]):
        dest = ctx["store"] / "formal" / family["dataset_hash"] / "conditions" / "dispatch" / (trial + ".json")
        publish(dest, encoded({"handoff_sha256":digest, "container_sha256":sha_bytes(raw), "job":job}))
        publish(out, raw)
    return {"trial_id":trial, "name":out.name, "sha256":sha_bytes(raw), "bytes":len(raw), "handoff_sha256":digest,
            "primary":"derive" if cond == "Q1" else "generate", "timing_repeat":repeat}


def common(parser):
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--family", required=True)
    parser.add_argument("--condition", required=True, choices=("B0", "R1", "Q1"))
    parser.add_argument("--families", type=Path, default=ROOT / "protocol/families-v1.json")
    parser.add_argument("--freeze", type=Path, default=ROOT / "configs/formal-freeze-v1.json")
    parser.add_argument("--method-freeze", type=Path)
    parser.add_argument("--preparation-run", help="R1 Mac half-resolution source derivation")


def cli_context(args):
    store = args.store.resolve()
    return context(store, args.families.resolve(), args.freeze.resolve(),
                   args.method_freeze or store / "protocol/method-freeze-v1.json", args.family, args.condition)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    common(p)
    p.add_argument("--timing-repeat", type=int, choices=(1, 2, 3))
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(make_job(cli_context(a), a.preparation_run, a.timing_repeat, a.out)))
