"""Admit the six frozen confirmation fits and close the independently measured upstream resource ledger."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path

from dual_records import bind, resolve, selected, stage
from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens._verify import require
from make_handoff import publish
from many_upstream_control import prep, publish_packet
from many_upstream_diagnostics import COHORT, FIRST_QUEUE, PILOT


DEVELOPMENT_TRAINING = (PILOT, *FIRST_QUEUE)
REMAINING = tuple(key for key, _, _ in COHORT if key not in DEVELOPMENT_TRAINING)


def inputs_for_case(store, tag, files, native_metadata):
    directory = store / "work/controller"
    if (directory / ("b-bank-" + tag) / "selected.json").is_file():
        bank = selected(store, "b-bank-" + tag)
        proof = bind(store, bank["run_id"], "payload/upstream-evidence.json")
        recorded = load(resolve((store, store / "history"), proof))
        files["bank-" + tag] = bank
        files["proof-" + tag] = proof
        files["training-" + tag] = recorded["native_registry"]["training"]
        # These are sealed recorder metadata, not parent payload artifacts. The worker opens
        # these four-field pins against the same original manifest as the native training receipt.
        native_metadata[tag] = {name: bind(store, recorded["native"]["run_id"], member)
                                for name, member in (("run", "run.json"), ("telemetry", "telemetry.csv"))}
    else:
        files["failure-" + tag] = selected(store, "b-terminal-failure-" + tag)
    files["sfm-" + tag] = selected(store, "b-sfm-" + tag)


def register(store, mode):
    confirmation = mode == "confirmation-queue"
    development_resources = mode == "development-resources"
    files = {"study": selected(store, "b-study-registration"),
             "recipe": selected(store, "b-upstream-recipe"),
             "budget": selected(store, "b-pilot-budget"),
             "development": selected(store, "b-development-diagnostics"),
             "development_queue": selected(store, "b-training-development-fit-queue")}
    if not development_resources:
        files["freeze"] = selected(store, "b-model-freeze")
        files["development_resources"] = selected(store, "b-upstream-training-development-resources")
    if mode == "closeout":
        files["confirmation_queue"] = selected(store, "b-confirmation-fit-queue")
        files["external_import"] = selected(store, "b-external-import-acceptance")
    cases = (*DEVELOPMENT_TRAINING, *REMAINING) if mode == "closeout" else DEVELOPMENT_TRAINING
    native_metadata = {}
    for tag in cases:
        inputs_for_case(store, tag, files, native_metadata)
    confirm_tags = () if development_resources else REMAINING
    for tag in confirm_tags:
        files["ticket-" + tag] = selected(store, "b-fit-ticket-" + tag)
    cmd = ["{python}", "-B", "{source}/scripts/many_upstream_ledger.py", "worker", "--config", "{config}", "--out", "{out}", "--store", str(store)]
    for name in files:
        cmd += ["--input", name, "{input:" + name + "}"]
    name, output = {"confirmation-queue": ("b-confirmation-fit-queue","payload/queue.json"),
                    "development-resources": ("b-upstream-training-development-resources","payload/resources.json"),
                    "closeout": ("b-upstream-closeout","payload/upstream.json")}[mode]
    return stage(store, name, "archive_verify" if confirmation else "report", files,
                 {"mode": mode, "observed_tags": cases, "confirmation_tags": confirm_tags, "native_metadata":native_metadata},
                 cmd, output, 170000)


def prepare_confirmations(store, sfm_python):
    frozen = selected(store, "b-model-freeze")
    receipt = load(resolve((store, store / "history"), frozen))
    require(receipt["schema"] == "el-many-confirmation-model-freeze-1"
            and receipt["all_confirmation_references_closed"], "common confirmation freeze missing")

    def prepare(tag):
        family, n = tag.rsplit("-r", 1)
        pin = prep(store, "b-source-" + family, int(n), sfm_python, "b-model-freeze")
        print(json.dumps({"prepared": tag, "pin": pin}), flush=True)

    with ThreadPoolExecutor(max_workers=2) as workers:
        list(workers.map(prepare, REMAINING))


def device_telemetry(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        samples = list(csv.DictReader(stream))
    device = [d for sample in samples for d in json.loads(sample.get("gpu_devices") or "[]")]
    used = [float(d["memory.used"]) for d in device if d.get("memory.used") is not None]
    utilization = [float(d["utilization.gpu"]) for d in device if d.get("utilization.gpu") is not None]
    process = [int(float(sample["gpu_process_memory_bytes"])) for sample in samples if sample.get("gpu_process_memory_bytes")]
    return {"recorder_samples":len(samples), "device_samples":len(device), "observed_peak_whole_device_mib":max(used,default=None),
            "observed_min_whole_device_mib":min(used,default=None), "observed_peak_device_utilization_percent":max(utilization,default=None),
            "observed_peak_driver_process_bytes":max(process,default=None),
            "scope":"nvidia-smi device-wide samples include desktop/background use; missing per-process driver counters remain null"}


def case_summary(tag, sizes, cfg, files, store):
    original = load(files["sfm-" + tag])
    require(original["target_photos_opened"] == original["official_points_or_tracks_read"] == 0,
            "source-only triangulation provenance differs")
    base = {"R": sizes[0], "target_poses": sizes[1], "scene_id": original["scene_id"],
            "sfm": cfg["sources"]["sfm-" + tag], "source_sfm_seconds": original["source_cpu_seconds"],
            "source_sfm_wall_seconds": original["wall_seconds"], "initial_gaussians": original["initial_points"]}
    if "failure-" + tag in files:
        d = load(files["failure-" + tag])
        require(d["schema"] == "el-many-terminal-failure-1" and d["family"] == base["scene_id"]
                and d["R"] == sizes[0] and len(d["target_ids"]) == sizes[1], "incorrect failed terminal case")
        return dict(base, status=d["status"], failure=cfg["sources"]["failure-" + tag], native_record=d["failed_native_run"], reason=d["reason"],
                    charged_gpu_seconds=d["elapsed_seconds"], gpu_accounting="complete unsuccessful native recorder wall time")

    bank, proof, fit = (load(files[key + "-" + tag]) for key in ("bank", "proof", "training"))
    metadata=cfg["native_metadata"][tag]
    require(all(p["run_id"]==proof["native"]["run_id"] and p["manifest_sha256"]==proof["native"]["manifest_sha256"]
                for p in metadata.values()),"resource metadata and scientific checkpoint have different native manifests")
    roots=store,store/"history"
    info=load(resolve(roots,metadata["run"]))
    runtime = proof["runtime"]
    require(bank["reconstruction_ids"] == bank["accessible_ids"] == fit["R"] == original["source_ids"]
            and len(bank["accessible_ids"]) == sizes[0] and len(bank["target_ids"]) == sizes[1]
            and bank["scene_id"] == base["scene_id"] and bank["checkpoint_sha256"] == fit["native_checkpoint"]["sha256"]
            and proof["bank_sha256"] == sha(files["bank-" + tag])
            and proof["native"]["run_id"] == info["run_id"] == fit["run_id"]
            and info["execution_status"] == "succeeded" and info["trial_id"] == "b-fit-" + tag,
            "native upstream/fit/source identity mismatch")
    visits = fit["source_visits"]
    require(set(visits) == set(fit["R"]) and all(isinstance(v, int) and v > 0 for v in visits.values())
            and sum(visits.values()) == fit["training_steps"] == 6000
            and fit["target_photo_opens"] == fit["internal_holdout"] == fit["lpips_evaluations"] == 0,
            "incomplete or unregistered native source exposure")
    source = proof["source_self"]
    parity = proof["native_frame_parity"]
    errors = {chan: max(view[chan]["max_abs"] for view in parity.values()) for chan in ("rgb", "alpha", "z")}
    require(len(parity) == sizes[0] + sizes[1], "incomplete source/target native parity")
    return dict(base, status="succeeded", bank=cfg["sources"]["bank-" + tag], proof=cfg["sources"]["proof-" + tag],
                native_record=metadata["run"], native_telemetry=metadata["telemetry"], checkpoint_sha256=bank["checkpoint_sha256"],
                final_gaussians=fit["gaussians"], training_updates=fit["training_steps"], min_actual_visits=min(visits.values()),
                max_actual_visits=max(visits.values()), source_order_sha256=fit["ordered_actual_source_step_sha256"],
                official_commit=fit["official_commit"], native_checkpoint_step=fit["native_checkpoint_step_zero_based"],
                units=bank["frame_receipt"]["unit"], fit_seconds=runtime["fit_seconds"],
                charged_gpu_seconds=runtime["job_wall_seconds"], gpu_accounting="native worker wall including fit and rendering",
                native_recorder_wall_seconds=info["elapsed_seconds_monotonic"], native_process_cpu_seconds=runtime["process_cpu_seconds"],
                native_peak_torch_allocated_bytes=runtime["peak_torch_allocated_bytes"],
                native_peak_torch_reserved_bytes=runtime["peak_torch_reserved_bytes"],
                native_sampled_peak_process_tree_rss_bytes=info["sampled_peak_tree_rss_bytes"],
                telemetry=device_telemetry(resolve(roots,metadata["telemetry"])),
                source_self_mae=proof["mean_source_self_mae"], source_mean_control_mae=proof["mean_source_colour_baseline_mae"],
                opacity_percentiles=fit["opacity_percentiles"], world_scale_percentiles=fit["scale_percentiles"],
                gaussians_above_0_005_opacity=fit["alpha_above_0_005"],
                mean_projected_gaussians=sum(row["projected_gaussians"] for row in source) / len(source),
                source_alpha_gt_0_5_pixels=sum(row["alpha_gt_0_5"] for row in source),
                source_finite_expected_z_pixels=sum(row["finite_z"] for row in source),
                source_pixels=sizes[0] * 378 * 252, native_render_max_absolute_difference=errors,
                auxiliary_validation_photos_read=0, internal_holdout=0)


def execute(cfg, paths, out, store):
    study, recipe, pilot = (load(paths[k]) for k in ("study", "recipe", "budget"))
    limits = study["study"]["budget"]
    require(recipe["baseline_id"] == study["baseline"] and pilot["projected_15_fits_budget"],
            "A baseline or prospective pilot budget differs")
    size = {key: (n, nh) for key, n, nh in COHORT}
    registered = cfg["observed_tags"]
    rows = {tag: case_summary(tag, size[tag], cfg, paths, store) for tag in registered}
    spent = sum(case["charged_gpu_seconds"] for case in rows.values())
    require(all(case["charged_gpu_seconds"] > 0 for case in rows.values()), "missing native elapsed time")
    over_time = [tag for tag, case in rows.items() if case["charged_gpu_seconds"] > limits["max_worker_seconds_per_fit"]]
    compliant = not over_time and spent <= limits["max_gpu_seconds"] and len(rows) <= limits["max_fits_including_failures"]
    rids = [case["native_record"]["run_id"] for case in rows.values()]
    require(len(set(rids)) == len(rids), "an upstream run counted twice")
    if cfg["mode"] == "development-resources":
        require(tuple(registered)==DEVELOPMENT_TRAINING,"unregistered development/training cohort")
        return save(out/"resources.json",{"schema":"el-many-development-training-resources-1","run_id":os.environ["EL_RUN_ID"],
                    "recipe":cfg["sources"]["recipe"],"development":cfg["sources"]["development"],"pilot_budget":cfg["sources"]["budget"],
                    "cases":rows,"issued":len(rows),"succeeded":sum(r["status"]=="succeeded" for r in rows.values()),
                    "charged_native_gpu_seconds":spent,"remaining_gpu_seconds":limits["max_gpu_seconds"]-spent,
                    "source_sfm_cpu_seconds":sum(r["source_sfm_seconds"] for r in rows.values()),
                    "fit_time_and_count_within_declared_limits":compliant,"cases_over_native_wall_cap":over_time})
    frozen = load(paths["freeze"])
    dev_resources = load(paths["development_resources"])
    require(frozen["schema"] == "el-many-confirmation-model-freeze-1" and frozen["all_confirmation_references_closed"]
            and frozen["upstream_recipe"] == cfg["sources"]["recipe"]
            and frozen["parents"]["development_diagnostics"] == cfg["sources"]["development"]
            and frozen["parents"]["pilot_budget"] == cfg["sources"]["budget"]
            and all(dev_resources["cases"][key]==rows[key] for key in DEVELOPMENT_TRAINING),
            "required upstream recipe, development resource census or common freeze differs")
    tickets = []
    for tag in cfg["confirmation_tags"]:
        t = load(paths["ticket-" + tag])
        require(t["permit_one_fit"] and t["trial"] == "b-fit-" + tag and t["recipe"] == cfg["sources"]["recipe"]
                and t["model_freeze"] == cfg["sources"]["freeze"]
                and len(t["R"]) == size[tag][0] and len(t["H_poses_only"]) == size[tag][1], "confirmation ticket differs")
        tickets.append({"tag": tag, "pin": cfg["sources"]["ticket-" + tag], "trial": t["trial"], "packet": t["packet"]})
    if cfg["mode"] == "confirmation-queue":
        worst_case = spent + len(tickets) * limits["max_worker_seconds_per_fit"]
        require(compliant and len(rows) == 9 and len(tickets) == 6 and len(rows) + len(tickets) <= limits["max_fits_including_failures"]
                and worst_case <= limits["max_gpu_seconds"], "insufficient reserved time for the six mandatory fits")
        return save(out / "queue.json", {"schema": "el-many-confirmation-fit-queue-1", "run_id": os.environ["EL_RUN_ID"],
                    "freeze": cfg["sources"]["freeze"], "budget": cfg["sources"]["budget"],
                    "measured_development":cfg["sources"]["development_resources"], "fits": tickets,
                    "spent_native_gpu_seconds": spent, "max_total_if_all_six_use_full_hour": worst_case,
                    "available_seconds": limits["max_gpu_seconds"]-spent, "fit_count_before_this_queue": len(rows),
                    "target_reference_access": "closed", "instructions": "Run each ticket once and return its own immutable native record."})
    ok = [r for r in rows.values() if r["status"] == "succeeded"]
    artifact_bytes = {}
    for tag, case in rows.items():
        if case["status"] != "succeeded":
            continue
        bank = load(paths["bank-" + tag])
        proof = load(paths["proof-" + tag])
        fit = load(paths["training-" + tag])
        source = bank["sources"].values()
        artifact_bytes[tag] = {
            "native_transfer_zip": proof["native"]["bytes"], "native_checkpoint": fit["native_checkpoint"]["bytes"],
            "scoring_bank_source_photographs": sum(v["photo"]["bytes"] for v in source),
            "scoring_bank_source_self_renders": sum(v["render"]["bytes"] for v in source),
            "scoring_bank_target_predictions": sum(v["render"]["bytes"] for v in bank["targets"].values())}
    save(out / "upstream.json", {"schema": "el-many-upstream-resource-closeout-1", "run_id": os.environ["EL_RUN_ID"],
         "baseline_id": study["baseline"], "recipe": cfg["sources"]["recipe"], "development": cfg["sources"]["development"],
         "common_model_freeze": cfg["sources"]["freeze"], "external_adapter": cfg["sources"]["external_import"],
         "measured_development_resources":cfg["sources"]["development_resources"],
         "train_development_queue": cfg["sources"]["development_queue"], "confirmation_queue": cfg["sources"]["confirmation_queue"],
         "issued_fit_count": len(rows), "succeeded": len(ok), "failed": len(rows) - len(ok), "policy": limits,
         "training_updates_definition":"Each successful fit executes 6000 single-source loss/backward and in-memory optimizer iterations. The official checkpoint is serialized before the 6000th optimizer application, and therefore contains the first 5999 updates.",
         "device_memory_scope":"On Windows the nvidia-smi samples are whole-device measurements; WDDM may withhold process-level counters. The maximum PyTorch allocated/reserved counts in this ledger are framework-local; a missing WDDM process counter stays null.",
         "fit_time_and_count_within_declared_limits":compliant, "cases_over_native_wall_cap":over_time,
         "charged_gpu_seconds": spent, "remaining_gpu_seconds": limits["max_gpu_seconds"] - spent,
         "successful_fit_seconds": sum(r["fit_seconds"] for r in ok),
         "successful_native_process_cpu_seconds": sum(r["native_process_cpu_seconds"] for r in ok),
         "source_sfm_cpu_seconds": sum(r["source_sfm_seconds"] for r in rows.values()),
         "total_source_renders": sum(r["R"] for r in ok), "total_target_predictions": sum(r["target_poses"] for r in ok),
         "max_cuda_reserved_bytes": max((r["native_peak_torch_reserved_bytes"] for r in ok), default=None),
         "max_device_observed_mib": max((r["telemetry"]["observed_peak_whole_device_mib"] for r in ok
                                         if r["telemetry"]["observed_peak_whole_device_mib"] is not None), default=None),
         "artifact_bytes_by_case":artifact_bytes,
         "artifact_byte_scope":"Logical transferred ZIP, checkpoint and scoring-bank component sizes. The same content occurs in multiple representations; these columns are not an estimate of physical disk use.",
         "independent_confirmation_photos_accessed_by_upstream": 0, "cases": rows})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("development-resources", "confirmation-queue", "closeout", "worker"))
    for name in ("store", "exchange", "config", "out", "sfm-python"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--input", nargs=2, action="append", default=[])
    args = p.parse_args()
    if args.action == "worker":
        execute(load(args.config), {k: Path(v) for k, v in args.input}, args.out, args.store.resolve())
    else:
        store = args.store.resolve()
        if args.sfm_python:
            require(args.action == "confirmation-queue", "source preparation is available only for the registered confirmation queue")
            prepare_confirmations(store, args.sfm_python)
        pin = register(store, args.action)
        if args.action == "confirmation-queue" and args.exchange:
            for tag in REMAINING:
                publish_packet(store, selected(store, "b-fit-ticket-" + tag), args.exchange)
            receipt = resolve((store, store / "history"), pin)
            publish(args.exchange / "confirmation-queue.json", receipt.read_bytes())
            publish(args.exchange / "confirmation-queue-pin.json", encoded(pin))
        print(json.dumps(pin))
