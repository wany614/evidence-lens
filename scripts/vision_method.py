"""Independent development replay, CPU readback, and preregistered B method-freeze evidence."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evidence_lens.formal_inputs import compact_pin, require
from evidence_lens import stage2_parity
from evidence_lens.scoring_io import TARGETS, new_output, write_json
from vision_data import file_for, open_dataset, packed
from vision_evaluate import FIVE, TWO, score_sources, read_scores
from vision_sources import METHODS, find_run
from vision_replay import ap_readback, threshold_readback, verify_tensor_payloads


SEEDS = (1729, 2718, 3141)
FORMAL_WORKER_INDEX_SHA = "ff0230c2edbd66e1e90c9fe9545cfb129b6379449e76ce0aa2890d8607020810"


def payload(run, path):
    name = "payload/" + path
    p = run.open(name)
    return p, run.files[name]["sha256"]


def json_output(run, path):
    p, sha = payload(run, path)
    return json.loads(p.read_bytes()), sha


def candidate(roots, pin, training_sha, development_sha):
    run = find_run(roots, compact_pin(pin))
    rid = run.pin["run_id"]
    require(
        run.run["execution_status"] == "succeeded"
        and run.run["run_kind"] == "model_fit"
        and run.run["experiment_id"] == "stage2-vision-train",
        "an original formal candidate is required",
    )
    training, receipt_sha = json_output(run, "training.json")
    require(
        training["run_id"] == rid
        and training["training_dataset_sha256"] == training_sha
        and training["development_dataset_sha256"] == development_sha
        and training["source_sha256"].get("worker-source.json") == FORMAL_WORKER_INDEX_SHA
        and training["phase"] == "train"
        and training["status"] == "completed",
        "wrong candidate cohort or phase",
    )
    model, model_sha = json_output(run, "package/model.json")
    require(
        training["package"]["path"] == "package/model.json"
        and model_sha == training["package"]["sha256"]
        and model["model_id"] == rid
        and model["branch"] == training["branch"]
        and model["seed"] == training["seed"]
        and model["threshold"] == training["threshold"],
        "exported package differs from trained candidate",
    )
    for path in (
        "development.json",
        "curves.json",
        "gradient-witnesses.json",
        "parameter-changes.json",
        "base-verification.json",
        "setup.json",
    ):
        p, sha = payload(run, path)
        require(training["files"][path]["sha256"] == sha, "candidate receipt differs: " + path)
    dev, dev_sha = json_output(run, "development.json")
    require(
        dev["family_equal_ap"] == training["best_development_family_equal_ap"]
        and dev["threshold"]["tau"] == training["threshold"]
        and dev["threshold"]["domain"] == "development-B0",
        "candidate dev epoch and threshold disagree",
    )
    curves, _ = json_output(run, "curves.json")
    require(
        [a["epoch"] for a in curves] == list(range(1, training["epochs"] + 1)), "incomplete epoch history"
    )
    best = max(curves, key=lambda a: a["development"]["family_equal_ap"])
    require(
        best["epoch"] == training["best_epoch"]
        and best["development"]["family_equal_ap"] == dev["family_equal_ap"],
        "selected checkpoint is not the earliest maximum development AP",
    )
    change, _ = json_output(run, "parameter-changes.json")
    require(
        change["frozen_identical"] is True and change["frozen_before"] == change["frozen_after"],
        "frozen backbone tensors changed",
    )
    witnesses, _ = json_output(run, "gradient-witnesses.json")
    require(
        len(witnesses) == training["task_gradient_minibatches"] == 8,
        "missing training loss gradient witnesses",
    )
    grad_names = ["projection.weight", "head.2.weight"]
    if training["branch"] == "L2-vision-ft":
        grad_names += [
            "backbone.blocks.10.attn.qkv.weight",
            "backbone.blocks.11.attn.qkv.weight",
            "backbone.norm.weight",
        ]
    setup, _ = json_output(run, "setup.json")
    actual_trainable = stage2_parity.trainable_parameters(training["branch"], change, setup)
    flags = setup["precision"]["backend_readback"]
    require(
        setup["device"].startswith("cuda")
        and set(setup["gradient_witness_names"]) == set(grad_names)
        and all(set(w["gradients_of_unregularized_view_mean_bce"]) == set(grad_names) for w in witnesses)
        and flags["deterministic_algorithms"] is True
        and flags["CUBLAS_WORKSPACE_CONFIG"] in (":4096:8", ":16:8")
        and all(
            flags[k] is False
            for k in (
                "cuda_matmul_allow_tf32",
                "cudnn_allow_tf32",
                "cudnn_benchmark",
                "deterministic_warn_only",
            )
        ),
        "formal GPU precision or eight data-gradient witnesses differ from registered runtime",
    )
    from evidence_lens_vision.contract import json_bytes

    tensors = {**change["frozen_after"], **{k: r["after_sha256"] for k, r in change["trainable"].items()}}
    check = training["cpu_export"]
    require(
        check["all_tensor_sha256_equal"] is True
        and check["tensor_count"] == len(tensors) == 181
        and check["tensor_table_sha256"] == hashlib.sha256(json_bytes(tensors)).hexdigest()
        and check["support_equal"] is True,
        "CPU-exported learned and frozen tensor hashes differ",
    )
    for name in grad_names:
        norms = [w["gradients_of_unregularized_view_mean_bce"][name] for w in witnesses]
        require(
            all(np.isfinite(g["l2_norm"]) and np.isfinite(g["max_abs"]) for g in norms)
            and sum(g["l2_norm"] > 0 and g["nonzero_coordinates"] > 0 for g in norms) >= 3,
            "insufficient data-loss gradients before decay: " + name,
        )
    if training["branch"] == "L2-vision-ft":
        for prefix in ("backbone.blocks.10.", "backbone.blocks.11.", "backbone.norm."):
            require(
                any(n.startswith(prefix) and v["changed"] > 0 for n, v in change["trainable"].items()),
                "required pretrained parameter group did not change",
            )
    weights, weights_sha = payload(run, "package/weights.safetensors")
    require(
        weights_sha == model["files"]["weights.safetensors"]["sha256"],
        "selected tensors differ from recorded package",
    )
    require(
        verify_tensor_payloads(weights, tensors) == 181, "unexpected number of learned and frozen tensors"
    )
    scorep, score_sha = payload(run, "development-scores/scores.json")
    selection = {**training, "manifest_sha256": pin["manifest_sha256"], "model_sha256": model_sha}
    return {
        "run": run,
        "fit": selection,
        "dev": dev,
        "score_pin": (scorep, score_sha),
        "training_receipt_sha256": receipt_sha,
        "development_receipt_sha256": dev_sha,
        "weights_sha256": weights_sha,
        "weights_bytes": weights.stat().st_size,
        "trainable_parameters": actual_trainable,
    }


def model_file(run, relative):
    path = "payload/" + relative
    run.open(path)
    return {"path": path, "sha256": run.files[path]["sha256"], "bytes": run.files[path]["bytes"]}


def macro_ap(views):
    vals = {f: [r["ap"] for r in each.values() if r["ap"] is not None] for f, each in views.items()}
    families = {
        f: {"value": float(np.mean(a)) if a else None, "defined_views": len(a)} for f, a in vals.items()
    }
    defined = [r["value"] for r in families.values() if r["value"] is not None]
    return {
        "value": float(np.mean(defined)) if defined else None,
        "defined_families": len(defined),
        "families": families,
    }


def same_ap(a, b):
    return a is None and b is None or a is not None and b is not None and abs(a - b) <= 1e-12


def neural_grid(dsroot, row, source, threshold):
    root, views, _ = source
    rec = views[row["family"], row["condition"], row["view"]]
    shape = (row["height"], row["width"])
    if row["status"] == "render_failed":
        require(
            rec["status"] == "render_failed" and rec["reason"] and rec["score"] is None,
            "missing failure disposition",
        )
        return np.full(shape, np.nan, np.float32)
    require(
        rec["rgb_sha256"] == row["rgb"]["sha256"] and rec["deploy_sha256"] == row["full"]["deploy"]["sha256"],
        "different target or feature raster in candidate replay",
    )
    if rec["status"] == "unsupported":
        require(rec["reason"] and rec["score"] is None, "unexplained unsupported neural prediction")
        return np.full(shape, np.nan, np.float32)
    require(rec["status"] == "scored", "unexpected neural scoring disposition")
    a = packed(file_for(root, rec["score"]), ("risk", "valid", "state"))
    s = a["risk"]
    v = np.isfinite(s)
    require(
        s.dtype == np.float32
        and s.shape == shape
        and not np.isinf(s).any()
        and ((s[v] >= 0) & (s[v] <= 1)).all()
        and a["valid"].dtype == bool
        and np.array_equal(a["valid"], v)
        and a["state"].dtype == np.uint8
        and np.array_equal(a["state"], np.where(v, s.astype(np.float64) >= threshold, 255).astype(np.uint8)),
        "native neural score/valid/state disagrees with frozen package",
    )
    return s


def independent_development(dsroot, ds, ds_sha, base_pin, candidates, primary):
    base = score_sources(ds, ds_sha, [base_pin], ("original-five",))
    models = {c["fit"]["run_id"]: c for c in candidates}
    sources = {
        rid: score_sources(ds, ds_sha, [c["score_pin"]], (c["fit"]["branch"],))[c["fit"]["branch"]]
        for rid, c in models.items()
    }
    for rid, src in sources.items():
        require(
            src[2]["model_sha256"] == models[rid]["fit"]["model_sha256"],
            "wrong candidate package in score index",
        )
    chosen = {}
    for m in TWO:
        ar = primary["chosen"][m]
        for seed in SEEDS:
            matches = [
                rid
                for rid, c in models.items()
                if c["fit"]["branch"] == m
                and c["fit"]["seed"] == seed
                and c["fit"]["head_lr"] == ar["head_lr"]
                and c["fit"]["backbone_lr"] == ar["backbone_lr"]
            ]
            require(len(matches) == 1, "paired seed lacks a single selected fit")
            chosen[m, seed] = matches[0]
    own = {rid: defaultdict(dict) for rid in models}
    domains = {m: {d: defaultdict(dict) for d in ("own", "five_common", "seven_common")} for m in METHODS}
    original_values = {m: defaultdict(list) for m in FIVE}
    paired = {seed: {m: defaultdict(dict) for m in ("L1-full", *TWO)} for seed in SEEDS}
    retained = {rid: defaultdict(list) for rid in models}
    v_counts = defaultdict(int)
    shared = {seed: [] for seed in SEEDS}
    for row in ds["views"]:
        f, v = row["family"], row["view"]
        t = packed(
            file_for(dsroot, row["full"]["truth"]), ("reference_valid", "rgb_evaluable", "rgb_high_error")
        )
        E = t["rgb_evaluable"]
        Y = t["rgb_high_error"] == 1
        V = int(t["reference_valid"].sum())
        v_counts[f] += V
        base_scores, _ = read_scores(row, base)
        common_five = E.copy()
        for m in FIVE:
            common_five &= np.isfinite(base_scores[m])
            original_values[m][f].append(base_scores[m][E & np.isfinite(base_scores[m])])
        allowed = (
            packed(file_for(dsroot, row["full"]["deploy"]), ("scoreable",))["scoreable"]
            if row["full"]["deploy"]
            else np.zeros(E.shape, bool)
        )
        neural = {}
        for rid, c in models.items():
            s = neural_grid(dsroot, row, sources[rid], c["fit"]["threshold"])
            ok = np.isfinite(s)
            require(not (ok & ~allowed).any(), "candidate scored unsupported camera/geometry pixels")
            ownmask = common_five & ok
            rank = ap_readback(s[ownmask], Y[ownmask])
            old = c["dev"]["views"][f][v]
            require(
                same_ap(rank["ap"], old["ap"])
                and old["V"] == V
                and old["E"] == int(E.sum())
                and rank["n"] == old["n"]
                and rank["positive"] == old.get("positive", 0),
                "original candidate AP or support is not reproduced",
            )
            own[rid][f][v] = rank
            retained[rid][f].append(s[E & ok])
            neural[rid] = s
        support_hash = stage2_parity.supports(neural)
        for seed in SEEDS:
            scores = {m: neural[chosen[m, seed]] for m in TWO}
            mask = common_five & np.isfinite(scores[TWO[0]]) & np.isfinite(scores[TWO[1]])
            if seed == SEEDS[0]:
                primary_scores = {**base_scores, **scores}
                for m, risk in primary_scores.items():
                    support = E & np.isfinite(risk)
                    own_rank = ap_readback(risk[support], Y[support])
                    seven_rank = (
                        own_rank if np.array_equal(support, mask) else ap_readback(risk[mask], Y[mask])
                    )
                    domains[m]["own"][f][v] = own_rank
                    domains[m]["seven_common"][f][v] = seven_rank
                    if m in FIVE:
                        domains[m]["five_common"][f][v] = (
                            seven_rank
                            if np.array_equal(common_five, mask)
                            else ap_readback(risk[common_five], Y[common_five])
                        )
            scores["L1-full"] = base_scores["L1-full"]
            for m, s in scores.items():
                paired[seed][m][f][v] = (
                    domains[m]["seven_common"][f][v] if seed == SEEDS[0] else ap_readback(s[mask], Y[mask])
                )
            shared[seed].append(
                {
                    "family": f,
                    "view": v,
                    "V": V,
                    "E": int(E.sum()),
                    "S5": int(common_five.sum()),
                    "S7": int(mask.sum()),
                    "S7Y": int(Y[mask].sum()),
                    "mask_sha256": hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(),
                    "candidate_support_sha256": support_hash,
                }
            )
    individual = {}
    for rid, c in models.items():
        ap = macro_ap(own[rid])
        require(
            same_ap(ap["value"], c["fit"]["best_development_family_equal_ap"]), "candidate macro AP differs"
        )
        per = {f: np.concatenate(a) for f, a in retained.pop(rid).items()}
        threshold = threshold_readback(per, v_counts)
        require(
            threshold["tau"] == c["fit"]["threshold"]
            and threshold["coverage"] == c["dev"]["threshold"]["coverage"],
            "canonical original development threshold or coverage differs",
        )
        individual[rid] = {
            "branch": c["fit"]["branch"],
            "seed": c["fit"]["seed"],
            "head_lr": c["fit"]["head_lr"],
            "backbone_lr": c["fit"]["backbone_lr"],
            "best_epoch": c["fit"]["best_epoch"],
            "ap": ap,
            "threshold": threshold,
            "views": own[rid],
            "scores_sha256": sources[rid][2]["sha256"],
        }
    independent_winners = stage2_parity.independent_choice(individual, primary)
    compared_masks = stage2_parity.paired_domains(shared, len(ds["views"]))
    pairs = {seed: {m: macro_ap(paired[seed][m]) for m in paired[seed]} for seed in SEEDS}
    return {
        "candidates": individual,
        "independent_winners": independent_winners,
        "matched_48_three_seed_domains": compared_masks,
        "primary_domains": {
            m: {d: {"macro": macro_ap(views), "views": views} if views else None for d, views in by.items()}
            for m, by in domains.items()
        },
        "original_thresholds": {
            m: threshold_readback({f: np.concatenate(x) for f, x in z.items()}, v_counts, m == "L0")
            for m, z in original_values.items()
        },
        "paired": pairs,
        "paired_domain": shared,
        "chosen": {str(seed): {m: chosen[m, seed] for m in TWO} for seed in SEEDS},
    }


def cpu_numbers(canonical, actual, tau):
    a, b = canonical, actual
    require(
        a.dtype == b.dtype == np.float32
        and a.shape == b.shape
        and not np.isinf(a).any()
        and not np.isinf(b).any(),
        "cross-device risk shape or dtype differs",
    )
    va, vb = np.isfinite(a), np.isfinite(b)
    both = va & vb
    x, y = a[both], b[both]
    require(np.all((x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)), "out of range device output")
    diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
    # Sigmoid scores are nonnegative, so their unsigned IEEE-754 ordering is monotone.
    ulp = np.abs(x.view(np.uint32).astype(np.int64) - y.view(np.uint32).astype(np.int64))
    return {
        "total": int(a.size),
        "common_finite": len(x),
        "support_disagreement": int(np.count_nonzero(va != vb)),
        "unequal_finite": int(np.count_nonzero(ulp)),
        "max_ulps": int(ulp.max()) if len(x) else None,
        "max_abs": float(diff.max()) if len(x) else None,
        "sum_abs": float(diff.sum()),
        "threshold_crossings": int(
            np.count_nonzero((x.astype(np.float64) < tau) != (y.astype(np.float64) < tau))
        ),
    }


def cost_evidence(roots, pin, family, c, ds, scores):
    run = find_run(roots, compact_pin(pin))
    rid = c["fit"]["run_id"]
    require(
        run.run["execution_status"] == "succeeded"
        and run.run["run_kind"] in ("archive_verify", "model_score"),
        "CPU cost requires a completed full-native recorder run",
    )
    result, sha = json_output(run, "native-cost.json")
    method = c["fit"]["branch"]
    require(
        result["schema"] == "el-vision-native-cost-1"
        and result["model_id"] == rid
        and result["model_sha256"] == c["fit"]["model_sha256"]
        and result["method"] == method
        and result["device"] == "cpu"
        and result["all_measured_views_512"] is True
        and result["all_measured_views_scored"] is True
        and result["warm_views"] == 18,
        "cost receipt does not bind the primary six-target/three-warm-pass CPU measurements",
    )
    rows = {r["view"]: r for r in ds["views"] if r["family"] == family and r["condition"] == "B0"}
    require(set(rows) == set(TARGETS), "cost family is not an authorized development B0 case")
    observed = {(r["pass"], r["view"]): r for r in result["views"]}
    require(
        len(observed) == len(result["views"]) == 24
        and set(observed) == {(p, v) for p in range(4) for v in TARGETS},
        "cost result is missing an original measurement",
    )
    warm = [r["wall_through_npz_seconds"] for r in result["views"] if r["pass"] > 0]
    require(
        all(np.isfinite(x) and x > 0 for x in warm)
        and result["cold_process_to_first_output_seconds"] > 0
        and np.isfinite(result["cold_process_to_first_output_seconds"])
        and type(result["process_tree_peak_rss_bytes"]) is int
        and result["process_tree_peak_rss_bytes"] > 0
        and np.isclose(result["hot_p50_seconds"], np.percentile(warm, 50), rtol=1e-12)
        and np.isclose(result["hot_p95_seconds"], np.percentile(warm, 95), rtol=1e-12),
        "CPU timings or RSS cannot be reproduced",
    )
    from vision_sources import signed

    original = rows[TARGETS[0]]
    choice = signed(
        roots[0] / "trials" / original["trial_id"] / "trial_selection.json", original["selection_sha256"]
    )
    require(
        choice["scene_family_id"] == family and choice["condition_id"] == "B0",
        "unexpected benchmark condition",
    )
    render = find_run(roots, compact_pin(choice["selected"]["render"]))
    prediction, render_sha = json_output(render, "prediction.json")
    require(
        str(result["platform"]).startswith(("macOS", "Darwin")),
        "benchmark must measure the registered Mac CPU",
    )
    comparisons = {}
    for (rep, v), r in sorted(observed.items()):
        require(
            r["status"] == "scored"
            and r["height"] == r["width"] == 512
            and r["inputs"]["input_png"]["sha256"] == rows[v]["source_input_sha256"]
            and r["producer_run_id"] == render.run["run_id"]
            and r["inputs"]["cameras"]["sha256"] == prediction["camera_sha256"]
            and r["inputs"]["prediction_metadata"]["sha256"] == render_sha
            and r["inputs"]["prediction/" + v]["sha256"] == prediction["views"][v]["sha256"],
            "CPU benchmark used another original source or target",
        )
        location = ("cold" if rep == 0 else f"warm-{rep}") + "/" + v + ".npz"
        path, digest = payload(run, location)
        require(digest == r["output_sha256"], "cost output hash differs")
        if rep:
            continue
        canonical = neural_grid(None, rows[v], scores, c["fit"]["threshold"])
        a = packed(path, ("risk", "valid", "state"))
        require(
            np.array_equal(a["valid"], np.isfinite(a["risk"]))
            and np.array_equal(
                a["state"],
                np.where(a["valid"], a["risk"].astype(np.float64) >= c["fit"]["threshold"], 255).astype(
                    np.uint8
                ),
            ),
            "CPU states disagree with actual package threshold",
        )
        comparisons[v] = cpu_numbers(canonical, a["risk"], c["fit"]["threshold"])
    from evidence_lens.stage2_authority import external_cost_fields

    process, process_sha = json_output(run, "process-cost.json")
    full_process = external_cost_fields(
        result, sha, process, process_sha, observed[0, "v01"]["output_sha256"]
    )
    return {
        "pin": compact_pin(pin),
        "native_cost_sha256": sha,
        "model_sha256": c["fit"]["model_sha256"],
        "family": family,
        **full_process,
        "hot_p50_seconds": result["hot_p50_seconds"],
        "hot_p95_seconds": result["hot_p95_seconds"],
        "canonical_win_to_mac": comparisons,
    }


def deploy_rule(paired, ft_cost):
    frozen, ft = TWO
    values = [tuple(paired[s][m]["value"] for m in ("L1-full", frozen, ft)) for s in SEEDS]
    complete = all(all(x is not None and np.isfinite(x) for x in vals) for vals in values)
    ds = [vals[2] - vals[1] for vals in values] if complete else []
    av = float(np.mean(ds)) if ds else None
    sd = float(np.std(ds, ddof=1)) if ds else None
    tested = {
        "three_paired_seeds": complete,
        "ft_exceeds_lr_every_seed": complete and all(v[2] > v[0] for v in values),
        "paired_mean_exceeds_sample_sd": complete and av > sd,
        "hot_p95_at_most_5s": ft_cost["hot_p95_seconds"] <= 5,
        "rss_at_most_2gib": ft_cost["rss_bytes"] <= 2147483648,
    }
    return {
        "default": ft if all(tested.values()) else "L1-full",
        "checks": tested,
        "paired_ft_minus_frozen": ds,
        "mean_difference": av,
        "sample_sd_difference": sd,
        "primary_seed": 1729,
        "calibration": "Ranking score for the registered RGB error event. Probability calibration requires a separate evaluation.",
    }


def export_lr(fit, thresholds, dev_root, dev_sha, output, run_id):
    import shutil
    from evidence_lens.l1 import load_package
    from evidence_lens.scoring_io import file_hash
    from vision_data import open_dataset

    fitted = load_package(fit, require_frozen=False)
    require(fitted["index"]["status"] == "fit", "expected an actual refitted LR package")
    d = open_dataset(dev_root, dev_sha, "development")
    t = json.loads(Path(thresholds).read_bytes())
    require(
        t["schema"] == "el-vision-baseline-thresholds-1"
        and t["development_dataset_sha256"] == dev_sha
        and t["method_sources"]["original-five"]["model_sha256"] == file_hash(fit / "model.json"),
        "baseline operating points belong to other learning data",
    )
    base = fitted["index"]
    params = {m: t["thresholds"][m] for m in ("L1-full", "L1-geometry", "constant", "1-alpha")}
    require(t["thresholds"]["L0"] == 0.5, "L0 fixed rule changed")
    with new_output(output) as out:
        folder = out / "model"
        folder.mkdir()
        for name in base["files"]:
            shutil.copyfile(fit / name, folder / name)
        write_json(
            folder / "thresholds.json",
            {
                "schema": "el-l1-thresholds-1",
                "threshold_run_id": run_id,
                "fit_run_id": base["fit_run_id"],
                "split_manifest_sha256": base["split_manifest_sha256"],
                "thresholds": params,
                "target": 0.8,
                "low_risk_rule": "finite score < tau",
                "selection": "whole ties, nearest family-macro coverage, lower coverage on ties",
            },
        )
        write_json(
            folder / "model.json",
            {
                **base,
                "status": "frozen",
                "model_id": run_id,
                "threshold_run_id": run_id,
                "development_sources": {r["family"]: r["source_input_sha256"] for r in d["views"]},
                "files": {**base["files"], "thresholds.json": file_hash(folder / "thresholds.json")},
            },
        )
        frozen = load_package(folder)
        require(
            frozen["models"] == fitted["models"] and frozen["thresholds"] == params,
            "LR deployment parameters differ from refit",
        )
        write_json(
            out / "lr-export.json",
            {
                "schema": "el-vision-lr-export-1",
                "model_id": run_id,
                "model_sha256": file_hash(folder / "model.json"),
                "original_fit_run_id": base["fit_run_id"],
                "original_fit_model_sha256": file_hash(fit / "model.json"),
                "baseline_thresholds_sha256": file_hash(thresholds),
                "development_dataset_sha256": dev_sha,
                "thresholds": t["thresholds"],
            },
        )


def freeze(
    roots,
    roster_path,
    roster_sha,
    study_path,
    study_sha,
    catalog_path,
    outpath,
    run_id,
    parity_path,
    original_catalog_path,
):
    from evidence_lens_vision.selection import choose
    from vision_sources import signed
    from evidence_lens.scoring_io import file_hash

    roster = signed(roster_path, roster_sha)
    study = signed(study_path, study_sha)
    catalog = signed(catalog_path, study["catalog_sha256"])
    parity_sha = roster["inference_parity_protocol_sha256"]
    parity_protocol = signed(parity_path, parity_sha)
    stage2_parity.protocol(parity_protocol, parity_sha, study_sha, study["catalog_sha256"])
    from vision_pipeline import ORIGINAL_CATALOG_SHA

    old_catalog = signed(original_catalog_path, ORIGINAL_CATALOG_SHA)
    require(
        roster["schema"] == "el-vision-postfit-roster-1"
        and roster["catalog_sha256"] == study["catalog_sha256"]
        and roster["study_sha256"] == study_sha
        and set(roster["cpu_costs"]) == set(TWO)
        and len(roster["candidate_runs"]) == 8,
        "incorrect post-fit study or budget",
    )
    pins = roster["stages"]

    def stage(role, name):
        run = find_run(roots, compact_pin(pins[role]))
        require(run.run["execution_status"] == "succeeded", "incomplete postfit input")
        p, h = payload(run, name)
        return run, p, h

    _, train, train_sha = stage("train_data", "dataset.json")
    _, dev, dev_sha = stage("dev_data", "dataset.json")
    d = open_dataset(dev.parent, dev_sha, "development")
    split_check = stage2_parity.split_families(
        open_dataset(train.parent, train_sha, "train"), d, catalog, old_catalog
    )
    lr, lr_export, lr_export_sha = stage("lr_export", "lr-export.json")
    lr_model, lr_sha = json_output(lr, "model/model.json")
    five, scorep, score_sha = stage("baseline_scores", "scores.json")
    base_pin = (scorep, score_sha)
    th, threshold_p, threshold_sha = stage("thresholds", "thresholds.json")
    thresholds = json.loads(threshold_p.read_bytes())
    ev, eval_p, eval_sha = stage("development_evaluation", "evaluation.json")
    e = json.loads(eval_p.read_bytes())
    require(
        lr_model["status"] == "frozen"
        and lr_model["training_manifest_sha256"] == train_sha
        and e["schema"] == "el-vision-comparison-1"
        and e["split"] == "development"
        and e["dataset_sha256"] == dev_sha
        and thresholds["development_dataset_sha256"] == dev_sha
        and e["thresholds_sha256"] == threshold_sha
        and thresholds["method_sources"]["original-five"] == {"sha256": score_sha, "model_sha256": lr_sha}
        and e["thresholds"] == thresholds["thresholds"]
        and e["family_count"] == 8
        and e["planned_target_views"] == 48,
        "wrong development score, frozen baseline, or seven-method numerical report",
    )
    qa_run = find_run(roots, compact_pin(roster["numerical_qa"]))
    qa, qa_sha = json_output(qa_run, "readback.json")
    require(
        qa_run.run["execution_status"] == "succeeded"
        and qa_run.run["run_kind"] == "archive_verify"
        and qa["schema"] == "el-vision-five-development-qa-1"
        and qa["run_id"] == qa_run.run["run_id"]
        and qa["source_sha256"]["development"] == dev_sha
        and all(
            compact_pin(qa["stages"][a]) == compact_pin(pins[b])
            for a, b in (
                ("development", "dev_data"),
                ("scores", "fitted_baseline_scores"),
                ("thresholds", "fitted_baseline_thresholds"),
            )
        )
        and len(qa["views"]) == 48
        and qa["max_abs_ap_difference"] <= 1e-12
        and all(qa["thresholds"][m]["tau"] == thresholds["thresholds"][m] for m in FIVE),
        "independent five-method development QA differs",
    )
    cands = [candidate(roots, p, train_sha, dev_sha) for p in roster["candidate_runs"]]
    selected = choose([c["fit"] for c in cands], "primary")
    by_id = {c["fit"]["run_id"]: c for c in cands}
    replay = independent_development(dev.parent, d, dev_sha, base_pin, cands, selected)
    fit, fit_path, fit_sha = stage("baseline_fit", "model/model.json")
    oldscore, oldsp, oldscore_sha = stage("fitted_baseline_scores", "scores.json")
    _, origthpath, origth_sha = stage("fitted_baseline_thresholds", "thresholds.json")
    _, newthpath, newth_sha = stage("baseline_thresholds", "thresholds.json")
    require(
        oldscore_sha == qa["source_sha256"]["scores"] and origth_sha == qa["source_sha256"]["thresholds"],
        "original-five QA score or threshold receipt differs",
    )
    baseline_export = stage2_parity.original_scores(
        oldscore,
        five,
        json.loads(oldsp.read_bytes()),
        json.loads(scorep.read_bytes()),
        json.loads(origthpath.read_bytes()),
        json.loads(newthpath.read_bytes()),
        fit_sha,
        lr_sha,
        qa,
        replay,
    )
    fitted_index = json.loads(fit_path.read_bytes())
    member_sha = {}
    for name in ("L1-full.json", "L1-geometry.json", "feature_schema.json"):
        a, ah = payload(fit, "model/" + name)
        b, bh = payload(lr, "model/" + name)
        require(
            ah == bh == fitted_index["files"][name] == lr_model["files"][name],
            "L1 coefficient/preprocessing/schema changed on packaging",
        )
        member_sha[name] = ah
    baseline_export["coefficient_preprocessing_schema_sha256"] = member_sha
    baseline_export.update(
        fitted_scores_sha256=oldscore_sha,
        frozen_scores_sha256=score_sha,
        fitted_thresholds_sha256=origth_sha,
        frozen_thresholds_sha256=newth_sha,
    )
    cost = {}
    cpu_family = roster["cost_family"]
    require(any(r["family"] == cpu_family for r in d["views"]), "cost family must be a preregistered dev row")
    for m in TWO:
        c = by_id[selected["chosen"][m]["run_id"]]
        entry = score_sources(d, dev_sha, [c["score_pin"]], (m,))[m]
        require(
            thresholds["method_sources"][m] == entry[2]
            and thresholds["thresholds"][m] == c["fit"]["threshold"],
            "joint freeze differs from winning package's producer-domain threshold",
        )
        cost[m] = cost_evidence(roots, roster["cpu_costs"][m], cpu_family, c, d, entry)
    verified_native = stage2_parity.parity(cost, parity_protocol)
    macro, _ = json_output(ev, "macro.json")
    require(
        len(macro) == 1 and macro[0]["condition"] == "B0" and macro[0]["cohort"] == "all",
        "wrong primary development family aggregation",
    )
    for m in FIVE:
        independent = replay["original_thresholds"][m]
        registered = thresholds["choices"][m]
        require(
            independent["tau"] == registered["tau"]
            and np.isclose(independent["coverage"], registered["coverage"], atol=1e-15, rtol=0),
            "independent original threshold changed " + m,
        )
    for m, domains in replay["primary_domains"].items():
        for name, result in domains.items():
            match = macro[0]["methods"][m][name + "_ap"]
            require(
                (result is None and match["defined_families"] == 0)
                or (result and same_ap(result["macro"]["value"], match["value"])),
                "independent development primary domain differs " + m + "/" + name,
            )
    for m in ("L1-full", *TWO):
        require(
            same_ap(macro[0]["methods"][m]["seven_common_ap"]["value"], replay["paired"][1729][m]["value"]),
            "independent primary AP and native all-seven report differ",
        )
    decision = deploy_rule(replay["paired"], cost[TWO[1]])
    training = []
    for c in cands:
        r = c["fit"]
        training.append(
            {
                **{
                    k: r[k]
                    for k in (
                        "run_id",
                        "manifest_sha256",
                        "model_sha256",
                        "seed",
                        "head_lr",
                        "backbone_lr",
                        "best_epoch",
                    )
                },
                "method": r["branch"],
                "training_json_sha256": c["training_receipt_sha256"],
                "development_json_sha256": c["development_receipt_sha256"],
                "weights_sha256": c["weights_sha256"],
                "trainable_parameters": c["trainable_parameters"],
            }
        )
    models = {}
    for m in ("L1-full", "L1-geometry"):
        models[m] = {
            **compact_pin(pins["lr_export"]),
            "model_sha256": lr_sha,
            "files": [
                model_file(lr, name)
                for name in (
                    "model/model.json",
                    "model/" + m + ".json",
                    "model/thresholds.json",
                    "model/feature_schema.json",
                )
            ],
        }
    for m in TWO:
        c = by_id[selected["chosen"][m]["run_id"]]
        models[m] = {
            **compact_pin(c["fit"]),
            "model_sha256": c["fit"]["model_sha256"],
            "files": [
                model_file(c["run"], name)
                for name in (
                    "package/model.json",
                    "package/weights.safetensors",
                    "package/architecture.json",
                    "package/preprocessing.json",
                    "package/base.json",
                )
            ],
        }
    doc = {
        "schema": "el-vision-method-freeze-1",
        "run_id": run_id,
        "catalog_sha256": study["catalog_sha256"],
        "study_sha256": study_sha,
        "test_family_ids": sorted(r["scene_family_id"] for r in catalog["families"] if r["split"] == "test"),
        "methods": list(METHODS),
        "models": models,
        "thresholds": {
            m: {
                "tau": thresholds["thresholds"][m],
                "domain": "development-B0",
                "coverage": thresholds["choices"][m]["coverage"],
                "selection": "fixed L0"
                if m == "L0"
                else "nearest 0.8 family-equal L/V, whole ties, lower coverage on tie",
            }
            for m in METHODS
        },
        "display_rule": "v01",
        "seeds": list(SEEDS),
        "formal_worker_index_sha256": FORMAL_WORKER_INDEX_SHA,
        "primary_seed": 1729,
        "training_receipts": training,
        "deployment_rule": decision,
        "cpu_costs": cost,
        "train_dataset_sha256": train_sha,
        "development_dataset_sha256": dev_sha,
        "development_evaluation_sha256": eval_sha,
        "thresholds_sha256": threshold_sha,
        "input_roster_sha256": roster_sha,
        "baseline_development_qa": {
            **compact_pin(roster["numerical_qa"]),
            "readback_sha256": qa_sha,
            **{
                k: qa[k]
                for k in (
                    "V",
                    "E",
                    "finite_E",
                    "defined_view_domain_ap_comparisons",
                    "max_abs_ap_difference",
                    "historical_development",
                )
            },
        },
        "inference_parity_protocol_sha256": parity_sha,
        "original_catalog_sha256": ORIGINAL_CATALOG_SHA,
        "native_inference_parity": verified_native,
        "learning_split_audit": split_check,
        "original_five_export": baseline_export,
        "fit_reuse": "Training used 16 original and 12 new families. Development selection used four original and four new families. The four historical regression families form a separate evaluation.",
    }
    with new_output(outpath) as out:
        write_json(out / "development-replay.json", replay)
        write_json(out / "selection.json", selected)
        doc["development_replay_sha256"] = file_hash(out / "development-replay.json")
        doc["selection_sha256"] = file_hash(out / "selection.json")
        write_json(out / "method-freeze.json", doc)
    return doc


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("export-lr")
    for n in ("fit", "thresholds", "development"):
        s.add_argument("--" + n, type=Path, required=True)
    s.add_argument("--dev-sha", required=True)
    s = sub.add_parser("freeze")
    for n in ("store", "shelf", "roster", "study", "catalog", "parity", "original-catalog"):
        s.add_argument("--" + n, type=Path, required=True)
    for n in ("roster-sha", "study-sha"):
        s.add_argument("--" + n, required=True)
    for cmd in sub.choices.values():
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--run-id", required=True)
    a = p.parse_args()
    if a.cmd == "export-lr":
        export_lr(a.fit, a.thresholds, a.development, a.dev_sha, a.output, a.run_id)
    else:
        freeze(
            [a.store.resolve(), a.shelf.resolve()],
            a.roster,
            a.roster_sha,
            a.study,
            a.study_sha,
            a.catalog,
            a.output,
            a.run_id,
            a.parity,
            a.original_catalog,
        )


if __name__ == "__main__":
    main()
