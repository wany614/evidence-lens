"""Authenticate the actual Stage B method, its bounded candidate selection and development CPU costs."""

import json
import math
import numpy as np
from pathlib import Path

from ._bundle import HASH, load, local_path, sha
from .formal_inputs import PinnedRun, compact_pin, require
from . import stage2_parity

METHODS = ("constant", "1-alpha", "L0", "L1-full", "L1-geometry", "L2-frozen", "L2-vision-ft")
NEURAL = METHODS[-2:]
SEEDS = (1729, 2718, 3141)
TARGETS = tuple(f"v{i:02d}" for i in range(1, 7))
RATES = {"L2-frozen": {(0.0, 1e-4), (0.0, 3e-4)}, "L2-vision-ft": {(1e-5, 3e-4), (3e-5, 3e-4)}}
WORKER_SOURCE_SHA = "ff0230c2edbd66e1e90c9fe9545cfb129b6379449e76ce0aa2890d8607020810"


def finite(value, low=None, high=None):
    return (
        type(value) in (float, int)
        and math.isfinite(value)
        and (low is None or value >= low)
        and (high is None or value <= high)
    )


def checked_file(run, name, wanted, digest_only=False):
    require(
        isinstance(wanted, str)
        and HASH.fullmatch(wanted)
        and name in run.files
        and run.files[name]["sha256"] == wanted,
        "original member digest mismatch " + name,
    )
    return None if digest_only else run.open(name)


def output(run, name, expected):
    return load(checked_file(run, "payload/" + name, expected))


def file_input(run, inputs, role, expected):
    matches = [r for r in inputs["inputs"] if r["role"] == role]
    require(
        len(matches) == 1
        and matches[0]["storage"] == "embedded"
        and matches[0]["kind"] == "file"
        and len(matches[0]["files"]) == 1,
        "missing single embedded " + role,
    )
    r = matches[0]
    item = r["files"][0]
    require(
        item["sha256"] == expected
        and r["path"] == item["path"]
        and r["path"].startswith("inputs/" + role + "/"),
        "different embedded " + role,
    )
    return load(checked_file(run, r["path"], expected))


class Parents:
    def __init__(self, roots, direct):
        self.roots = tuple(Path(p).resolve() for p in roots)
        self.direct = {p["run_id"]: p["manifest_sha256"] for p in direct}
        require(len(self.direct) == len(direct), "repeated original parent run")
        self.cache = {}

    def get(self, pin, *, direct=True):
        p = compact_pin(pin)
        rid, sha = p["run_id"], p["manifest_sha256"]
        require(not direct or self.direct.get(rid) == sha, "missing direct scientific parent " + rid)
        if rid not in self.cache:
            base = next((root for root in self.roots if (root / "runs" / rid).is_dir()), None)
            require(base is not None, "registered parent is unavailable " + rid)
            self.cache[rid] = PinnedRun(base, p)
        run = self.cache[rid]
        require(
            run.pin == p and run.run["execution_status"] == "succeeded",
            "original parent is unsuccessful or has changed",
        )
        return run


def check_model_members(run, spec, path, names):
    index_path = "payload/" + path + "/model.json"
    index = load(checked_file(run, index_path, spec["model_sha256"]))
    listed = {r["path"]: r for r in spec["files"]}
    expected = {index_path, *["payload/" + path + "/" + n for n in names]}
    require(len(listed) == len(spec["files"]) and set(listed) == expected, "wrong frozen model inventory")
    for n, r in listed.items():
        sha = spec["model_sha256"] if n == index_path else index["files"][n.split("/")[-1]]
        if isinstance(sha, dict):
            sha = sha["sha256"]
        require(
            sha == r["sha256"] and run.files[n]["bytes"] == r["bytes"], "misbound frozen model payload " + n
        )
        checked_file(run, n, sha)
    return index


def candidates(parents, method, roster, selection):
    receipts = method["training_receipts"]
    require(
        len(receipts) == 8
        and len({r["run_id"] for r in receipts}) == 8
        and len(roster["candidate_runs"]) == 8
        and {tuple(compact_pin(r).values()) for r in receipts}
        == {tuple(compact_pin(r).values()) for r in roster["candidate_runs"]},
        "incomplete eight-candidate roster or duplicate identity",
    )
    require(
        set(selection["chosen"]) == set(NEURAL)
        and len(selection["all_candidates"]) == 8
        and selection["schema"] == "el-vision-primary-selection-1"
        and selection["seed"] == SEEDS[0],
        "incomplete recorded candidate selection",
    )
    selected_rows = {r["run_id"]: r for r in selection["all_candidates"]}
    require(
        set(selected_rows) == {r["run_id"] for r in receipts}, "recorded primary candidate roster differs"
    )
    origin = (
        "training_dataset_sha256",
        "development_dataset_sha256",
        "preprocessing_sha256",
        "base_pin_sha256",
        "source_sha256",
    )
    groups = {m: [] for m in NEURAL}
    paired = {}
    for r in receipts:
        run = parents.get(r)
        rid = r["run_id"]
        m = r["method"]
        require(
            run.run["run_kind"] == "model_fit"
            and run.run["experiment_id"] == "stage2-vision-train"
            and m in NEURAL,
            "not an original formal candidate",
        )
        t = output(run, "training.json", r["training_json_sha256"])
        d = output(run, "development.json", r["development_json_sha256"])
        idx = output(run, "package/model.json", r["model_sha256"])
        require(
            t["schema"] == "el-vision-training-1"
            and t["phase"] == "train"
            and t["status"] == "completed"
            and t["run_id"] == idx["model_id"] == rid
            and idx["phase"] == "train"
            and t["branch"] == idx["branch"] == m
            and t["seed"] == idx["seed"] == r["seed"]
            and t["training_dataset_sha256"]
            == idx["training_dataset_sha256"]
            == method["train_dataset_sha256"]
            and t["development_dataset_sha256"]
            == idx["development_dataset_sha256"]
            == method["development_dataset_sha256"]
            and t["source_sha256"].get("worker-source.json")
            == method["formal_worker_index_sha256"]
            == WORKER_SOURCE_SHA
            and t["package"]["path"] == "package/model.json"
            and t["package"]["sha256"] == r["model_sha256"]
            and t["files"]["development.json"]["sha256"] == r["development_json_sha256"]
            and t["best_epoch"] == r["best_epoch"]
            and t["threshold"] == idx["threshold"] == d["threshold"]["tau"]
            and d["threshold"]["domain"] == "development-B0"
            and finite(t["best_development_family_equal_ap"], 0, 1)
            and t["best_development_family_equal_ap"] == d["family_equal_ap"]
            and type(t["best_epoch"]) is int
            and 1 <= t["best_epoch"] <= t["epochs"]
            and t["optimizer_steps"] > 0,
            "training, weights, original development or source identity differs",
        )
        require(
            all(selection[k] == t[k] for k in origin)
            and t["preprocessing_sha256"]
            == idx["preprocessing_sha256"]
            == idx["files"]["preprocessing.json"]["sha256"]
            and t["base_pin_sha256"] == idx["base_pin_sha256"] == idx["files"]["base.json"]["sha256"],
            "candidate preprocessing, backbone origin, or registered worker source differs",
        )
        setup = output(run, "setup.json", t["files"]["setup.json"]["sha256"])
        change = output(run, "parameter-changes.json", t["files"]["parameter-changes.json"]["sha256"])
        require(
            stage2_parity.trainable_parameters(m, change, setup) == r["trainable_parameters"],
            "method trainable tensor roster differs from its original files",
        )
        wh = r["weights_sha256"]
        require(
            idx["files"]["weights.safetensors"]["sha256"] == wh,
            "candidate model index and tensor digest differ",
        )
        # The selected packages below also verify every selected tensor file from its bytes.
        checked_file(run, "payload/package/weights.safetensors", wh, digest_only=True)
        rate = (r["backbone_lr"], r["head_lr"])
        require(
            rate in RATES[m] and rate == (t["backbone_lr"], t["head_lr"]) and r["seed"] in SEEDS,
            "unregistered learning rate or seed",
        )
        bind = selected_rows[rid]
        require(
            bind["branch"] == m
            and bind["manifest_sha256"] == r["manifest_sha256"]
            and bind["model_sha256"] == r["model_sha256"]
            and all(
                bind[k] == t[k]
                for k in (
                    "seed",
                    "head_lr",
                    "backbone_lr",
                    "best_epoch",
                    "best_development_family_equal_ap",
                    "threshold",
                )
            ),
            "selection row differs from its sealed training result",
        )
        key = (t["initial_head_sha256"], t["paired_design_sha256"])
        require(
            all(isinstance(v, str) and HASH.fullmatch(v) for v in key)
            and paired.setdefault(r["seed"], key) == key,
            "paired runs used different initial heads or designs",
        )
        curves = json.loads(
            checked_file(run, "payload/curves.json", t["files"]["curves.json"]["sha256"]).read_bytes()
        )
        require(
            [z["epoch"] for z in curves] == list(range(1, t["epochs"] + 1))
            and all(finite(z["development"]["family_equal_ap"], 0, 1) for z in curves),
            "incomplete original candidate epoch history",
        )
        best = max(curves, key=lambda z: z["development"]["family_equal_ap"])
        require(
            best["epoch"] == t["best_epoch"]
            and best["development"]["family_equal_ap"] == t["best_development_family_equal_ap"],
            "checkpoint differs from earliest maximum",
        )
        checked_file(run, "payload/initial-head.safetensors", key[0], digest_only=True)
        groups[m].append((r, t, rate))
    wins = {}
    for m, items in groups.items():
        a = [row for row in items if row[0]["seed"] == SEEDS[0]]
        require(
            len(items) == 4 and len(a) == 2 and {z[2] for z in a} == RATES[m],
            "both registered seed-A learning-rate candidates are required for " + m,
        )
        r, t, rate = min(a, key=lambda z: (-z[1]["best_development_family_equal_ap"], z[2]))
        repeats = [z for z in items if z[0]["seed"] != SEEDS[0]]
        require(
            len(repeats) == 2
            and {z[0]["seed"] for z in repeats} == set(SEEDS[1:])
            and all(z[2] == rate for z in repeats),
            "B and C must repeat the winning learning rate",
        )
        model = method["models"][m]
        chosen = selection["chosen"][m]
        require(
            compact_pin(model) == compact_pin(r) == compact_pin(chosen)
            and model["model_sha256"] == r["model_sha256"] == chosen["model_sha256"]
            and all(
                chosen[k] == t[k]
                for k in (
                    "seed",
                    "head_lr",
                    "backbone_lr",
                    "best_epoch",
                    "best_development_family_equal_ap",
                    "threshold",
                )
            )
            and method["thresholds"][m]["tau"] == t["threshold"],
            "primary model is not the selected seed-A package",
        )
        check_model_members(
            parents.get(r),
            model,
            "package",
            ("weights.safetensors", "architecture.json", "base.json", "preprocessing.json"),
        )
        run = parents.get(r)
        scores_hash = t["files"]["development-scores/scores.json"]["sha256"]
        scores = output(run, "development-scores/scores.json", scores_hash)
        require(
            scores["method"] == m
            and scores["schema"] == "el-vision-scores-1"
            and scores["model_sha256"] == r["model_sha256"]
            and scores["dataset_sha256"] == method["development_dataset_sha256"]
            and scores["threshold"] == t["threshold"]
            and scores["selected_epoch"] == t["best_epoch"]
            and str(scores["device"]).split(":")[0] == "cuda",
            "canonical development score producer differs",
        )
        wins[m] = (run, scores_hash, scores)
    return wins


def percentile(values, q):
    step = (len(values) - 1) * q
    lo = math.floor(step)
    a = sorted(values)
    return a[lo] + (a[math.ceil(step)] - a[lo]) * (step - lo)


def cost_parent_inputs(run):
    d = load(run.open("inputs.json"))
    owners = {r["run_id"]: r["manifest_sha256"] for r in d["parents"]}
    roles = {r["role"]: r for r in d["inputs"]}

    def role(k):
        r = roles[k]
        rid = r["parent_run_id"]
        require(
            rid in owners and r["kind"] == "file" and len(r["files"]) == 1, "unbound benchmark input " + k
        )
        h = r["files"][0]["sha256"]
        checked_file(run, r["files"][0]["path"], h)
        return {"run_id": rid, "manifest_sha256": owners[rid]}, h

    return role


def external_cost_fields(runtime, runtime_hash, process, process_hash, first_hash):
    """Project the original two clocks and RSS samplers after checking the external observer's two byte pins."""
    require(
        process["schema"] == "el-vision-external-process-cost-1"
        and process["model_id"] == runtime["model_id"]
        and process["model_sha256"] == runtime["model_sha256"]
        and process["runtime_report_sha256"] == runtime_hash
        and process["first_output_sha256"] == first_hash,
        "external process cost has a different model, native receipt or first output",
    )
    cold = process["launch_to_first_complete_npz_seconds"]
    inner = runtime["cold_process_to_first_output_seconds"]
    outer_rss = process["process_tree_peak_rss_bytes"]
    inner_rss = runtime["process_tree_peak_rss_bytes"]
    interval = process["sample_interval_seconds"]
    inside = runtime["rss_interval_seconds"]
    require(
        all(finite(x) and x > 0 for x in (cold, inner, interval, inside))
        and all(type(x) is int and x > 0 for x in (outer_rss, inner_rss)),
        "invalid external or native timing/RSS observation",
    )
    return {
        "process_cost_sha256": process_hash,
        "cold_seconds": cold,
        "rss_bytes": max(inner_rss, outer_rss),
        "external_cold_seconds": cold,
        "native_cold_seconds": inner,
        "external_process_tree_peak_rss_bytes": outer_rss,
        "native_process_tree_peak_rss_bytes": inner_rss,
        "external_sample_interval_seconds": interval,
        "native_rss_sample_interval_seconds": inside,
        "cold_boundary": "Parent process clock before launch to first observed complete cold/v01.npz.",
        "rss_method": "Maximum of the external process-tree and internal native process-tree sampled RSS peaks.",
    }


def original_grid(path, threshold, height=512, width=512):
    with np.load(path, allow_pickle=False) as z:
        require(
            set(z.files) == {"risk", "valid", "state"},
            "native output is missing numerical scores, support or states",
        )
        risk, valid, state = z["risk"], z["valid"], z["state"]
    require(
        risk.shape == valid.shape == state.shape == (height, width)
        and risk.dtype == np.float32
        and valid.dtype == np.bool_
        and state.dtype == np.uint8
        and not np.isinf(risk).any()
        and np.array_equal(np.isfinite(risk), valid)
        and np.all((risk[valid] >= 0) & (risk[valid] <= 1))
        and np.array_equal(
            state, np.where(valid, risk.astype(np.float64) >= threshold, 255).astype(np.uint8)
        ),
        "score payload does not contain native risk, support and threshold states",
    )
    return risk


def device_difference(a, b, tau):
    va, vb = np.isfinite(a), np.isfinite(b)
    common = va & vb
    x, y = a[common], b[common]
    dx = np.abs(x.astype(np.float64) - y.astype(np.float64))
    ulp = np.abs(x.view(np.uint32).astype(np.int64) - y.view(np.uint32).astype(np.int64))
    return {
        "total": int(a.size),
        "common_finite": int(common.sum()),
        "support_disagreement": int(np.count_nonzero(va != vb)),
        "unequal_finite": int(np.count_nonzero(ulp)),
        "max_ulps": int(np.max(ulp)) if x.size else None,
        "max_abs": float(np.max(dx)) if x.size else None,
        "sum_abs": float(np.sum(dx)),
        "threshold_crossings": int(
            np.count_nonzero((x.astype(np.float64) < tau) != (y.astype(np.float64) < tau))
        ),
    }


def costs(parents, method, roster, family, trial, development, selected):
    require(
        set(method["cpu_costs"]) == set(roster["cpu_costs"]) == set(NEURAL)
        and roster["cost_family"] == family,
        "two complete primary native Mac CPU cost records are required",
    )
    rows = {v["view"]: v for v in development["views"] if v["family"] == family and v["condition"] == "B0"}
    require(
        set(rows) == set(TARGETS) and len(set(v["selection_sha256"] for v in rows.values())) == 1,
        "incomplete registered benchmark development family",
    )
    require(
        all(row["trial_id"] == trial for row in rows.values())
        and local_path(trial) == trial
        and "/" not in trial,
        "wrong preregistered development trial",
    )
    selection_path = parents.roots[0] / "trials" / trial / "trial_selection.json"
    require(
        selection_path.is_file()
        and not selection_path.is_symlink()
        and sha(selection_path) == rows["v01"]["selection_sha256"],
        "benchmark selection differs from registered development dataset",
    )
    selection = load(selection_path)
    require(
        selection["selection_status"] == "selected"
        and selection["trial_id"] == trial
        and selection["scene_family_id"] == family
        and selection["condition_id"] == "B0",
        "wrong original selected development condition",
    )
    expected_ref, expected_render = tuple(
        compact_pin(selection["selected"][role]) for role in ("reference", "render")
    )
    reference_and_prediction = None
    for m in NEURAL:
        c = method["cpu_costs"][m]
        win = method["models"][m]
        run = parents.get(c["pin"])
        require(
            compact_pin(c["pin"]) == compact_pin(roster["cpu_costs"][m])
            and c["family"] == family
            and run.run["run_kind"] in ("model_score", "archive_verify")
            and c["model_sha256"] == win["model_sha256"],
            "native cost is not bound to the original primary model and development family",
        )
        d = output(run, "native-cost.json", c["native_cost_sha256"])
        require(
            d["schema"] == "el-vision-native-cost-1"
            and d["device"] == "cpu"
            and str(d["platform"]).startswith(("macOS", "Darwin"))
            and d["model_id"] == win["run_id"]
            and d["model_sha256"] == win["model_sha256"]
            and d["method"] == m
            and d["all_measured_views_512"] is True
            and d["all_measured_views_scored"] is True
            and d["warm_views"] == 18,
            "CPU cost is not the actual complete native matched measurement",
        )
        lookup = cost_parent_inputs(run)
        model_pin, model_hash = lookup("model")
        ref, source = lookup("input_image")
        camera_ref, cal = lookup("cameras")
        render, pred_sha = lookup("prediction")
        require(
            model_pin == compact_pin(win)
            and model_hash == win["model_sha256"]
            and ref == camera_ref == expected_ref
            and render == expected_render,
            "native cost recorder model/reference roles differ",
        )
        rr = parents.get(ref, direct=False)
        pr = parents.get(render, direct=False)
        require(
            rr.run.get("scene_family_id") == pr.run.get("scene_family_id") == family
            and rr.run.get("split") == pr.run.get("split") == "development"
            and pr.run.get("condition_id") == "B0",
            "benchmark inputs do not belong to the registered B0 development trial",
        )
        checked_file(rr, "payload/input.png", source)
        checked_file(rr, "payload/cameras.json", cal)
        pred = output(pr, "prediction.json", pred_sha)
        require(
            pred["input_sha256"] == source
            and pred["camera_sha256"] == cal
            and pred["producer_run_id"] == render["run_id"]
            and all(r["source_input_sha256"] == source for r in rows.values()),
            "native development source hash differs",
        )
        if reference_and_prediction is None:
            reference_and_prediction = (ref, render, source, cal, pred_sha)
        require(
            reference_and_prediction == (ref, render, source, cal, pred_sha),
            "neural CPU measurements use different development inputs",
        )
        observed = {(r["pass"], r["view"]): r for r in d["views"]}
        require(
            len(d["views"]) == len(observed) == 24
            and set(observed) == {(k, v) for k in range(4) for v in TARGETS}
            and set(c["canonical_win_to_mac"]) == set(TARGETS),
            "complete six cold and eighteen warm measurements are required",
        )
        tau = method["thresholds"][m]["tau"]
        selected_run, _, canonical = selected[m]
        producer = {(r["family"], r["condition"], r["view"]): r for r in canonical["views"]}
        expected = {(r["family"], r["condition"], r["view"]) for r in development["views"]}
        require(
            len(canonical["views"]) == len(producer) == 48 and set(producer) == expected,
            "incomplete native producer score frames",
        )
        support_cache = {}
        for (k, v), r in observed.items():
            require(
                type(r["pass"]) is int
                and r["height"] == r["width"] == 512
                and r["status"] == "scored"
                and r["supported_pixels"] > 0
                and finite(r["wall_through_npz_seconds"])
                and r["wall_through_npz_seconds"] > 0
                and r["producer_run_id"] == render["run_id"]
                and all(
                    r["inputs"][role]["sha256"] == h
                    for role, h in (
                        ("input_png", source),
                        ("cameras", cal),
                        ("prediction_metadata", pred_sha),
                        ("prediction/" + v, pred["views"][v]["sha256"]),
                    )
                ),
                "cost view is incomplete or came from a different native target",
            )
            path = checked_file(
                run,
                ("payload/cold/" if k == 0 else "payload/warm-" + str(k) + "/") + v + ".npz",
                r["output_sha256"],
            )
            h = r["output_sha256"]
            if k == 0 or h not in support_cache:
                risk = original_grid(path, tau)
                support_cache[h] = int(np.count_nonzero(np.isfinite(risk)))
                if k == 0:
                    record = producer[family, "B0", v]
                    entry = record["score"]
                    require(
                        record["rgb_sha256"] == rows[v]["rgb"]["sha256"]
                        and record["deploy_sha256"] == rows[v]["full"]["deploy"]["sha256"],
                        "wrong canonical development input or output",
                    )
                    if record["status"] == "unsupported":
                        require(
                            record["reason"] and entry is None,
                            "unsupported producer score lacks its original disposition",
                        )
                        upstream = np.full(risk.shape, np.nan, np.float32)
                    else:
                        require(
                            record["status"] == "scored" and entry["path"] == family + "/B0/" + v + ".npz",
                            "different canonical native score map",
                        )
                        canonical_path = checked_file(
                            selected_run, "payload/development-scores/" + entry["path"], entry["sha256"]
                        )
                        require(
                            canonical_path.stat().st_size == entry["bytes"],
                            "canonical grid byte count differs",
                        )
                        upstream = original_grid(canonical_path, tau)
                    require(
                        device_difference(upstream, risk, tau) == c["canonical_win_to_mac"][v],
                        "reported producer-to-Mac numerical diagnostics differ from native arrays",
                    )
            require(
                support_cache[h] == r["supported_pixels"], "Mac native support differs from recorded array"
            )
        warm = [r["wall_through_npz_seconds"] for (k, v), r in observed.items() if k > 0]
        require(
            all(
                math.isclose(d[f"hot_p{p}_seconds"], percentile(warm, p / 100), abs_tol=1e-12, rel_tol=1e-12)
                and d[f"hot_p{p}_seconds"] == c[f"hot_p{p}_seconds"]
                for p in (50, 95)
            )
            and d["cold_process_to_first_output_seconds"]
            == observed[0, "v01"]["fresh_process_to_first_output_seconds"],
            "native wall order statistics or cold measurement differ",
        )
        proc = output(run, "process-cost.json", c["process_cost_sha256"])
        measured = external_cost_fields(
            d, c["native_cost_sha256"], proc, c["process_cost_sha256"], observed[0, "v01"]["output_sha256"]
        )
        require(
            all(c.get(key) == value for key, value in measured.items()),
            "external or native cold/RSS projection differs from original recorder bytes",
        )


def verify_frozen_authority(roots, freeze, catalog_digest, expected_families):
    d = load(freeze.open("payload/method-freeze.json"))
    deps = load(freeze.open("inputs.json"))
    parents = Parents(roots, deps["parents"])
    require(
        freeze.run["run_kind"] == "report"
        and freeze.run["execution_status"] == "succeeded"
        and freeze.run["experiment_id"] == "stage2-method-freeze"
        and d["schema"] == "el-vision-method-freeze-1"
        and d["run_id"] == freeze.run["run_id"]
        and d["catalog_sha256"] == catalog_digest
        and tuple(d["methods"]) == METHODS
        and d["display_rule"] == "v01"
        and d["seeds"] == list(SEEDS)
        and d["primary_seed"] == SEEDS[0]
        and len(expected_families) == 8
        and sorted(d["test_family_ids"]) == sorted(expected_families)
        and set(d["models"]) == set(METHODS[3:])
        and set(d["thresholds"]) == set(METHODS),
        "incomplete registered method, thresholds, seeds or display",
    )
    roster = file_input(freeze, deps, "method_roster", d["input_roster_sha256"])
    catalog = file_input(freeze, deps, "catalog", catalog_digest)
    study = file_input(freeze, deps, "study", d["study_sha256"])
    require(
        roster["schema"] == "el-vision-postfit-roster-1"
        and roster["catalog_sha256"] == study["catalog_sha256"] == catalog_digest
        and roster["study_sha256"] == d["study_sha256"]
        and sorted(r["scene_family_id"] for r in catalog["families"] if r["split"] == "test")
        == sorted(expected_families),
        "frozen roster, study or population differs",
    )
    parity_sha = d["inference_parity_protocol_sha256"]
    parity_protocol = file_input(freeze, deps, "parity", parity_sha)
    require(
        roster["inference_parity_protocol_sha256"] == parity_sha,
        "different parity protocol in original roster",
    )
    stage2_parity.protocol(parity_protocol, parity_sha, d["study_sha256"], catalog_digest)
    devs = sorted(r["scene_family_id"] for r in catalog["families"] if r["split"] == "development")
    require(len(devs) == len(set(devs)) == 4, "expected four registered new development families")
    st = roster["stages"]
    th = output(parents.get(st["thresholds"]), "thresholds.json", d["thresholds_sha256"])
    ds = output(parents.get(st["dev_data"]), "dataset.json", d["development_dataset_sha256"])
    train = output(parents.get(st["train_data"]), "dataset.json", d["train_dataset_sha256"])
    original_sha = d["original_catalog_sha256"]
    original = file_input(freeze, deps, "original_catalog", original_sha)
    require(
        roster["original_catalog_sha256"] == original_sha
        and (
            catalog_digest != stage2_parity.CATALOG_SHA
            or original_sha == "fb7fabef06d006c992bc38cb03c283a41d90238e0b174664a1357f274bb36baa"
        )
        and stage2_parity.split_families(train, ds, catalog, original) == d["learning_split_audit"],
        "original training/development/test split audit differs",
    )

    require(
        ds["schema"] == "el-vision-dataset-1"
        and ds["split"] == "development"
        and len(ds["views"]) == 48
        and len({(r["family"], r["view"]) for r in ds["views"]}) == 48
        and len({r["family"] for r in ds["views"]}) == 8
        and all(r["condition"] == "B0" and r["height"] == r["width"] == 512 for r in ds["views"])
        and th["schema"] == "el-vision-thresholds-1"
        and th["primary_seed"] == SEEDS[0]
        and th["development_dataset_sha256"] == d["development_dataset_sha256"],
        "thresholds were not frozen on the eight-family native development B0",
    )
    for m, t in d["thresholds"].items():
        choice = th["choices"][m]
        require(
            t["domain"] == "development-B0"
            and finite(t["tau"], 0, 1.000001)
            and finite(t["coverage"], 0, 1)
            and t["tau"] == choice["tau"] == th["thresholds"][m]
            and t["coverage"] == choice["coverage"]
            and (m != "L0" or t["tau"] == 0.5),
            "frozen operating point differs for " + m,
        )
    sel = output(freeze, "selection.json", d["selection_sha256"])
    selected = candidates(parents, d, roster, sel)
    a, b = [d["models"][m] for m in METHODS[3:5]]
    require(
        compact_pin(a) == compact_pin(b) == compact_pin(st["lr_export"])
        and a["model_sha256"] == b["model_sha256"],
        "original logistic package identity differs",
    )
    for m in METHODS[3:5]:
        spec = d["models"][m]
        run = parents.get(spec)
        idx = check_model_members(run, spec, "model", ("feature_schema.json", m + ".json", "thresholds.json"))
        lr = output(run, "model/thresholds.json", idx["files"]["thresholds.json"])
        require(
            idx["status"] == "frozen"
            and idx["model_id"] == idx["threshold_run_id"] == spec["run_id"]
            and lr["threshold_run_id"] == spec["run_id"]
            and all(
                lr["thresholds"][n] == d["thresholds"][n]["tau"]
                for n in ("L1-full", "L1-geometry", "constant", "1-alpha")
            ),
            "frozen logistic operating points differ",
        )
    require(
        all(
            th["method_sources"][k]["model_sha256"] == d["models"][m]["model_sha256"]
            for k, m in (("original-five", "L1-full"), (NEURAL[0], NEURAL[0]), (NEURAL[1], NEURAL[1]))
        ),
        "development thresholds bind different score producers",
    )
    require(
        all(th["method_sources"][m]["sha256"] == selected[m][1] for m in NEURAL),
        "threshold freeze used different native development score bytes",
    )
    trial = next(r["trials"]["B0"] for r in catalog["families"] if r["scene_family_id"] == devs[0])
    costs(parents, d, roster, devs[0], trial, ds, selected)
    require(
        stage2_parity.parity(d["cpu_costs"], parity_protocol) == d["native_inference_parity"],
        "registered native device parity readback differs",
    )
    replay = output(freeze, "development-replay.json", d["development_replay_sha256"])
    require(
        stage2_parity.paired_domains(replay["paired_domain"]) == replay["matched_48_three_seed_domains"]
        and stage2_parity.independent_choice(replay["candidates"], sel) == replay["independent_winners"],
        "independent development winners or domain evidence differs",
    )
    fits = {r["run_id"]: r for r in d["training_receipts"]}
    require(
        set(replay["candidates"]) == set(fits)
        and all(
            (row["branch"], row["seed"], row["head_lr"], row["backbone_lr"])
            == (fits[rid]["method"], fits[rid]["seed"], fits[rid]["head_lr"], fits[rid]["backbone_lr"])
            for rid, row in replay["candidates"].items()
        ),
        "independent rank candidate roster differs from the sealed originals",
    )
    decision = d["deployment_rule"]
    pair = replay["paired"]
    v = [tuple(pair[str(s)][m]["value"] for m in ("L1-full", *NEURAL)) for s in SEEDS]
    good = all(all(finite(x, 0, 1) for x in row) for row in v)
    dif = [row[2] - row[1] for row in v] if good else []
    mean = float(np.mean(dif)) if dif else None
    sd = float(np.std(dif, ddof=1)) if dif else None
    cost = d["cpu_costs"][NEURAL[1]]
    checks = {
        "three_paired_seeds": good,
        "ft_exceeds_lr_every_seed": good and all(ft > lr for lr, _, ft in v),
        "paired_mean_exceeds_sample_sd": good and mean > sd,
        "hot_p95_at_most_5s": cost["hot_p95_seconds"] <= 5,
        "rss_at_most_2gib": cost["rss_bytes"] <= 2147483648,
    }
    require(
        decision["checks"] == checks
        and decision["paired_ft_minus_frozen"] == dif
        and decision["mean_difference"] == mean
        and decision["sample_sd_difference"] == sd
        and decision["primary_seed"] == 1729
        and decision["default"] == (NEURAL[1] if all(checks.values()) else "L1-full"),
        "frozen default differs from recorded paired AP or Mac cost",
    )
    return d
