"""Record transfer-verified timing repeats separately from the quality selections."""

from evidence_lens._bundle import encoded, load, sha
from evidence_lens._index import writer_gate
from evidence_lens._verify import require
from condition_jobs import TIMING, validate_r1
from formal_lineage import run_pin
from make_handoff import publish, sha_bytes
from select_trial import transfer_evidence
from condition_selection import bound, digest


def explicit_timing(ctx, n, preparation, producer, render):
    family, cond, store, verified = ctx["family"], ctx["condition"], ctx["store"], ctx["verified"]
    require(type(n) is int and n in (1, 2, 3) and family["scene_family_id"] == ctx["first_test"], "unregistered timing family or repeat")
    trial = f'{ctx["quality_trial"]}-timing-{n:02}'
    directory = store / "formal" / family["dataset_hash"] / "conditions"
    binding = load(directory / "dispatch" / (trial + ".json"))
    job = binding["job"]
    warm_file = store / "trials" / ctx["quality_trial"] / "trial_selection.json"
    warm = load(warm_file)
    require(sha_bytes(encoded(job)) == binding["handoff_sha256"] and job["experiment_id"] == TIMING and job["timing_repeat"] == n and
            job["condition"] == cond and job["warmup"]["selection_sha256"] == sha(warm_file) and
            warm["selection_status"] == "selected" and warm["condition_id"] == cond and
            job["b0_selection_sha256"] == ctx["b0_selection_sha256"] and job["method_freeze_sha256"] == ctx["release"]["method_freeze_sha256"],
            "timing dispatch does not identify this condition's selected warm-up or method")
    source_id = ctx["b0_ids"]["reference"]
    if cond == "R1":
        require(preparation == warm["selected"]["preparation"]["run_id"], "R1 timing used a different source")
        validate_r1(ctx, preparation)
        source_id = preparation
    else:
        require(not preparation, "only R1 timing uses a separate source preparation")
    require(run_pin(verified.get(source_id)) == job["source"], "timing source manifest differs")
    primary_kind = "derivation" if cond == "Q1" else "generation"
    produced = verified.get(producer, primary_kind)
    rendered = verified.get(render, "render")
    baseline_gen = ctx["b0_ids"]["generation"]
    require(produced["run"]["parent_run_ids"] == [baseline_gen if cond == "Q1" else source_id] and
            set(rendered["run"]["parent_run_ids"]) == {source_id, producer}, "timed stage parents differ")
    warm_primary = warm["selected"]["producer" if cond != "B0" else "generation"]["run_id"]
    warm_render = warm["selected"]["render"]["run_id"]
    for current, original in ((produced,warm_primary),(rendered,warm_render)):
        run = current["run"]
        cfg = load(store / "runs" / run["run_id"] / "config.resolved.yaml")
        previous = load(store / "runs" / original / "config.resolved.yaml")
        require(run["experiment_id"] == TIMING and run["trial_id"] == trial and run["condition_id"] == cond and
                cfg["timing_repeat"] == n and cfg["formal_handoff_sha256"] == binding["handoff_sha256"] and cfg["warmup"] == job["warmup"] and
                all(cfg[k] == previous[k] for k in ("renderer","generator","seed")), "timing parameters differ from the quality warm-up")
    bound(ctx, render, source_id, ("input.png","cameras.json"))
    bound(ctx, render, producer, ("scene.ply","generation.json"))
    source_hash = digest(ctx,source_id,"input.png")
    cameras_hash = digest(ctx,source_id,"cameras.json")
    gm = load(store / "runs" / producer / "payload/generation.json")
    pm = load(store / "runs" / render / "payload/prediction.json")
    require(gm["producer_run_id"] == producer and pm["producer_run_id"] == render and pm["generation_run_id"] == producer and
            gm["ply_sha256"] == pm["ply_sha256"] == digest(ctx,producer,"scene.ply") and
            gm["input_sha256"] == pm["input_sha256"] == source_hash and gm["camera_sha256"] == pm["camera_sha256"] == cameras_hash,
            "timed products differ from the registered input or Gaussian producer")
    if cond == "Q1":
        bound(ctx,producer,baseline_gen,("scene.ply","generation.json"))
        require(gm["forward_calls_in_this_stage"] == 0 and gm["parent_generation_run_id"] == baseline_gen, "invalid timed Q1 derivation")
    else:
        bound(ctx,producer,source_id,("input.png","cameras.json"))
        require(gm["checkpoint_sha256"] == ctx["baseline"]["evidence"]["checkpoint_sha256"] and
                gm["checkpoint_sha256"] in {obj["sha256"] for obj in produced["shared"]}, "timing used a different checkpoint")
    transfers = transfer_evidence(store,{producer:produced, render:rendered})
    require(transfers.get(producer) and transfers.get(render), "missing timing transfer receipt")
    attempts = []
    for p in sorted((store / "runs").glob("*/run.json")):
        run = load(p)
        if run["trial_id"] == trial and run["experiment_id"] == TIMING:
            rec = verified.get(run["run_id"])
            require(run["run_kind"] in {primary_kind,"render"}, "timing trials must contain only the two timed stages")
            if run["execution_status"] == "succeeded":
                require(run["run_id"] in (producer,render), "ambiguous successful timing stages")
            attempts.append(dict(run_pin(rec), kind=run["run_kind"], status=run["execution_status"], retry_of=run["retry_of_run_id"]))
    result = {"schema":"el-timing-repeat-1", "experiment_id":TIMING,"trial_id":trial,"timing_repeat":n,"condition_id":cond,
              "scene_family_id":family["scene_family_id"],"catalog_sha256":family["dataset_hash"],"method_freeze_sha256":job["method_freeze_sha256"],
              "included_in_quality_or_training":False,"warmup":job["warmup"],"handoff_sha256":binding["handoff_sha256"],
              "producer":run_pin(produced),"render":run_pin(rendered),"producer_kind":primary_kind,"attempts":attempts,
              "source_sha256":source_hash,"camera_sha256":cameras_hash,"producer_elapsed_seconds":produced["run"]["elapsed_seconds_monotonic"],
              "render_elapsed_seconds":rendered["run"]["elapsed_seconds_monotonic"],"render_phases":pm["phases"],
              "producer_phases":gm.get("phases",{}),"producer_seconds":{k:gm[k] for k in ("preprocess_seconds","forward_seconds","unproject_seconds","derivation_seconds") if k in gm}}
    dest = directory / "timing" / (trial + ".json")
    with writer_gate(store):
        if dest.is_file():
            earlier = load(dest)["transfer_receipts"]
            require(all(entry in transfers[rid] for rid, rows in earlier.items() for entry in rows), "timing transfer receipt changed")
            result["transfer_receipts"] = earlier
        else:
            result["transfer_receipts"] = transfers
        publish(dest, encoded(result))
    return {"trial_id":trial,"timing_receipt":str(dest),"sha256":sha(dest)}
