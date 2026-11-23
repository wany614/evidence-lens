"""Record independent four-fit tensor checks and full native CPU costs on the fixed development example."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve()
sys.path[:0] = [str(HERE.parent), str(HERE.parents[1] / "src")]

from evidence_lens_vision.contract import digest, file_record, output_dir, put, read, require
from evidence_lens_multirisk.data import VIEWS
from evidence_lens_multirisk.model import B_INDEX, B_RUN, B_WEIGHTS, METHODS, SEEDS
from evidence_lens_multirisk.proof import verify
from evidence_lens_multirisk.train import CATALOG, SCHEMA, STUDY
from vision_gpu_job import original, parse_pin

B_PARENT = B_RUN + "=0100b3f27e79c4498d51854d249312fc54cd7c80291b812e68f42c1dbfd88583"


def candidate(store, shelf, pin):
    from stage3_terminal_recovery import AFTER, EXPERIMENT, recovery_identity
    from evidence_lens.stage3_decision import validate_fit_origin

    run = original(store, shelf, pin)
    require(
        (run.run["run_kind"], run.run["experiment_id"])
        in (("model_fit", "stage3-risk-neural-fit"), ("derivation", EXPERIMENT))
        and run.run["dataset_hash"] == CATALOG,
        "requires an original successful Stage C formal neural fit",
    )
    validate_fit_origin(run, store, shelf)
    fit_path = run.open("payload/training.json")
    fit = read(fit_path)
    index = read(run.open("payload/package/model.json"))
    require(
        fit["schema"] == "el-multisource-risk-training-1"
        and fit["phase"] == index["phase"] == "formal"
        and fit["run_id"] == pin["run_id"] == index["model_id"]
        and fit["method"] in METHODS
        and fit["method"] == index["method"]
        and fit["seed"] == index["seed"] in SEEDS
        and fit["threshold"] == index["threshold"]
        and index["catalog_sha256"] == CATALOG
        and index["protocol_sha256"] == STUDY
        and index["feature_schema_sha256"] == SCHEMA
        and fit["model_index_sha256"] == run.files["payload/package/model.json"]["sha256"]
        and fit["weights_sha256"] == run.files["payload/package/weights.safetensors"]["sha256"]
        and fit["optimizer_steps"] > 0
        and fit["best_epoch"] >= 1,
        "candidate has the wrong scope, model parameters or operating point",
    )
    require(index["worker_zip_sha256"] == AFTER, "unknown corrected export worker")
    if run.run["run_kind"] == "derivation":
        recovery_identity(run.run, fit, index, read(run.open("payload/recovery.json")))
    return run, fit, index


def paired_evidence(runs, b, train, dev, dest, threads, store, shelf):
    from stage3_terminal_recovery import AFTER, BEFORE, fetch_failure, source_equivalence

    failed = fetch_failure(store, shelf)
    old_source, new_source, changes = source_equivalence(
        failed.open("inputs/worker/worker.zip"), runs[0][0].open("inputs/worker/worker.zip")
    )
    from stage3_fit_audit import Inputs, checked_development, experiment_setup, first_step_replay, verify_base

    b_index, b_weights = b.open("payload/package/model.json"), b.open("payload/package/weights.safetensors")
    base = verify_base(b_index, b_weights)
    put(dest / "source-B-tensor-roster.json", base)
    data = Inputs(train, dev)
    candidates = {}
    for run, fit, idx in runs:
        key = (fit["seed"], fit["method"])
        require(key not in candidates, "duplicate registered fit")
        for name in (
            "setup.json",
            "initialization.json",
            "parameter-changes.json",
            "gradient-witnesses.json",
            "initial-head.safetensors",
            "package/weights.safetensors",
        ):
            run.open("payload/" + name)
        original = run.open("payload/independent-proof/verified-updates.json")
        verify_dir = dest / str(fit["seed"]) / fit["method"]
        check = verify(
            b_index, b_weights, run.root / "payload", verify_dir, run.files["payload/training.json"]["sha256"]
        )
        require(
            check == read(original), "independent Mac tensor comparison differs from the recorded original"
        )
        require(
            fit["training_dataset_sha256"] == data.train_sha
            and fit["development_dataset_sha256"] == data.dev_sha,
            "independent data source differs",
        )
        exact = {
            "method": fit["method"],
            "seed": fit["seed"],
            "source": run.pin,
            "model_index_sha256": fit["model_index_sha256"],
            "weights_sha256": fit["weights_sha256"],
            "setup": experiment_setup(run, fit, data),
            "development": checked_development(run, fit, data),
            "gradients": first_step_replay(run, fit, b_index, b_weights, data, threads),
        }
        numerical = verify_dir / "scientific-readback.json"
        put(numerical, exact)
        require(
            exact["gradients"]["within_numerical_check"],
            "first-step independent numerical evidence differs; retained scientific-readback.json",
        )
        curves = read(run.open("payload/curves.json"))
        witnesses = read(run.open("payload/gradient-witnesses.json"))
        heads = read(run.open("payload/setup.json"))["initial_tensors"]
        require(
            len(curves) == fit["epochs"]
            and curves[-1]["optimizer_steps"] == fit["optimizer_steps"]
            and fit["best_epoch"]
            == next(
                e["epoch"]
                for e in curves
                if e["development"]["family_equal_ap"]
                == max(v["development"]["family_equal_ap"] for v in curves)
            ),
            "best epoch or counted updates differ from original history",
        )
        underlying = BEFORE if run.run["run_kind"] == "derivation" else AFTER
        declared = old_source if underlying == BEFORE else new_source
        require(
            digest(run.open("inputs/worker/worker.zip")) == AFTER
            and idx["code_files"] == new_source
            and all(declared.get(n) == h for n, h in fit["source_sha256"].items())
            and all(
                new_source.get(n) == h
                for n, h in fit.get("export_source_sha256", fit["source_sha256"]).items()
            ),
            "optimizer or shipped runtime differs from its verified executable",
        )
        comparable = {
            k: v
            for k, v in idx.items()
            if k
            in (
                "authority_sha256",
                "catalog_sha256",
                "development_dataset_sha256",
                "feature_schema_sha256",
                "phase",
                "preprocessing_sha256",
                "protocol_sha256",
                "training_dataset_sha256",
            )
        }
        candidates[key] = {
            "source": run.pin,
            "training_sha256": check["training_sha256"],
            "original_proof": {
                **run.pin,
                "member": "payload/independent-proof/verified-updates.json",
                "sha256": digest(original),
            },
            "check": file_record(verify_dir / "verified-updates.json", dest),
            "scientific_readback": file_record(numerical, dest),
            "proof": check,
            "method": fit["method"],
            "seed": fit["seed"],
            "threshold": fit["threshold"],
            "initial_tensors_sha256": hashlib.sha256(json.dumps(heads, sort_keys=True).encode()).hexdigest(),
            "view_order": [e["view_order_sha256"] for e in curves],
            "first_minibatches": [
                (r["family"], r["condition"], r["view"], r["samples_sha256"]) for r in witnesses
            ],
            "optimizer_worker_sha256": underlying,
            "registered_design": comparable,
            "source_keys": sorted(fit["source_sha256"]),
            "worker_sha256": idx["worker_zip_sha256"],
        }
    require(set(candidates) == {(s, m) for s in SEEDS for m in METHODS}, "requires all four registered fits")
    all_rows = list(candidates.values())
    first = all_rows[0]
    require(
        all(
            r["registered_design"] == first["registered_design"]
            and r["source_keys"] == first["source_keys"]
            and r["worker_sha256"] == first["worker_sha256"]
            and r["proof"]["paired_design_sha256"] == first["proof"]["paired_design_sha256"]
            for r in all_rows
        ),
        "four candidates do not share the same experiment and code",
    )
    pairs = {}
    for s in SEEDS:
        fro, ft = (candidates[s, m] for m in METHODS)
        n = min(len(fro["view_order"]), len(ft["view_order"]))
        require(
            n > 0
            and fro["view_order"][:n] == ft["view_order"][:n]
            and fro["first_minibatches"] == ft["first_minibatches"]
            and fro["proof"]["initial_head_sha256"] == ft["proof"]["initial_head_sha256"]
            and fro["initial_tensors_sha256"] == ft["initial_tensors_sha256"]
            and fro["proof"]["backbone_changed_scalars"] == 0 < ft["proof"]["backbone_changed_scalars"],
            "paired initialization, data order, frozen backbone or true suffix update differs",
        )
        pairs[str(s)] = {
            "matched_order_epochs": n,
            "matched_logged_minibatches": len(fro["first_minibatches"]),
            "initial_tensors_sha256": ft["initial_tensors_sha256"],
            "initial_head_sha256": ft["proof"]["initial_head_sha256"],
            "fits": {
                v["method"]: {k: v[k] for k in ("source", "training_sha256", "check", "proof", "threshold")}
                for v in (fro, ft)
            },
        }
    require(
        pairs[str(SEEDS[0])]["initial_head_sha256"] != pairs[str(SEEDS[1])]["initial_head_sha256"],
        "sensitivity seeds share one head",
    )
    return {
        "schema": "el-stage3-neural-comparison-1",
        "run_id": os.environ["EL_RUN_ID"],
        "worker_sha256": first["worker_sha256"],
        "source_equivalence": changes,
        "optimization_workers_by_fit": [
            {"method": r["method"], "seed": r["seed"], "worker_sha256": r["optimizer_worker_sha256"]}
            for _, r in sorted(candidates.items())
        ],
        "catalog_sha256": CATALOG,
        "study_sha256": STUDY,
        "feature_schema_sha256": SCHEMA,
        "B_parent": b.pin,
        "B_index_sha256": B_INDEX,
        "B_weights_sha256": B_WEIGHTS,
        "B_roster": file_record(dest / "source-B-tensor-roster.json", dest),
        "training_dataset": train.pin,
        "development_dataset": dev.pin,
        "training_dataset_sha256": data.train_sha,
        "development_dataset_sha256": data.dev_sha,
        "cuda_bitwise_reproduction_established": False,
        "primary_seed": SEEDS[0],
        "fits": [
            {
                "method": v["method"],
                "seed": v["seed"],
                **v["source"],
                "model_index_sha256": v["proof"]["model_index_sha256"],
                "weights_sha256": v["proof"]["weights_sha256"],
                "original_proof": v["original_proof"],
                "independent_readback": v["check"],
                "scientific_readback": v["scientific_readback"],
            }
            for _, v in sorted(candidates.items())
        ],
        "pairs": pairs,
    }


def compare_maps(measured, canonical, tau):
    import numpy as np

    def read_grid(p):
        with np.load(p, allow_pickle=False) as z:
            require(set(z.files) == {"risk", "valid", "state"}, "unexpected native score fields")
            risk, valid, state = (z[k] for k in ("risk", "valid", "state"))
        require(
            risk.dtype == np.float32
            and valid.dtype == bool
            and state.dtype == np.uint8
            and risk.shape == valid.shape == state.shape == (256, 256)
            and np.array_equal(valid, np.isfinite(risk))
            and np.all(np.isnan(risk[~valid]))
            and np.array_equal(state[valid], risk[valid].astype(np.float64) >= tau)
            and np.all(state[~valid] == 255),
            "invalid native risk, unknown support or threshold decision",
        )
        return risk, valid, state

    x, v, s = read_grid(measured)
    y, u, t = read_grid(canonical)
    require(np.array_equal(v, u), "native CPU/GPU support mismatch")
    err = np.abs(x[v].astype(np.float64) - y[u])
    maximum = float(err.max(initial=0.0))
    require(
        maximum <= 0.0002,
        "native CPU and selected Windows scores exceed the registered FP32 export tolerance",
    )
    return {
        "valid": int(v.sum()),
        "unknown": int((~v).sum()),
        "max_abs_error": maximum,
        "mean_abs_error": float(err.mean()) if len(err) else None,
        "bitwise_equal_on_support": int((x[v] == y[u]).sum()),
        "threshold_crossings": int(np.count_nonzero(s != t)),
    }


def validate_cpu(runs, dev, out, threads):
    from evidence_lens_multirisk.inference import costs

    dsfile = dev.open("payload/dataset.json")
    data = read(dsfile)
    dsha = digest(dsfile)
    require(
        data["cohort"] == "formal"
        and data["split"] == "development"
        and data["catalog_sha256"] == CATALOG
        and dev.run["run_kind"] == "dataset_build"
        and dev.run["experiment_id"] == "stage3-risk-development-data",
        "registered formal development data required for CPU validation",
    )
    fams = sorted(data["families"])
    family = fams[0]
    require(
        len(fams) == 4 and family.startswith("scene_057_"),
        "fixed representative case is the first development family",
    )
    reps = {}
    for run, fit, idx in runs:
        name = fit["method"]
        require(
            fit["seed"] == SEEDS[0] and name not in reps and fit["development_dataset_sha256"] == dsha,
            "wrong primary model/development source",
        )
        for f in idx["files"]:
            run.open("payload/package/" + f)
        sp = run.open("payload/development-scores/scores.json")
        sc = read(sp)
        require(
            sc["dataset_sha256"] == dsha
            and sc["model_index_sha256"] == fit["model_index_sha256"]
            and sc["threshold"] == idx["threshold"]
            and sc["method"] == name,
            "incorrect original development scores",
        )
        case_results = {}
        for condition in ("M2", "M4"):
            selected = [r for r in data["views"] if (r["family"], r["condition"]) == (family, condition)]
            require(
                {r["view"] for r in selected} == set(VIEWS)
                and len(selected) == 6
                and all(r["status"] == "ready" for r in selected),
                "fixed CPU example is incomplete; preserve the failed case",
            )
            cs = selected[0]["case"]
            require(all(r["case"] == cs for r in selected), "views use different calibrated geometry")
            casefile = dev.open("payload/" + cs["path"])
            require(digest(casefile) == cs["sha256"], "different original case")
            meta = read(casefile)
            prefix = Path("payload") / Path(cs["path"]).parent
            for member in [
                meta["prediction"],
                meta["source_cameras"],
                meta["render_cameras"],
                *meta["source_images"].values(),
                *meta["renders"].values(),
            ]:
                local = dev.open((prefix / member["path"]).as_posix())
                require(
                    dev.files[(prefix / member["path"]).as_posix()]["sha256"] == member["sha256"]
                    and local.stat().st_size == member["bytes"],
                    "unbound original native projection input",
                )
            dest = out / name / condition
            args = SimpleNamespace(
                model=run.root / "payload/package",
                model_sha256=fit["model_index_sha256"],
                case=casefile.parent,
                case_sha256=cs["sha256"],
                threads=threads,
                scopes=["one", "two", "all"],
                repetitions=1,
                output=dest,
            )
            cost = costs(args, HERE.with_name("multi_risk_worker.py"))
            warm = read(dest / "warm/warm.json")
            truth_free = {}
            for row in warm["views"]:
                v, s = row["view"], row["scope"]
                matches = [
                    r
                    for r in sc["views"]
                    if (r["family"], r["condition"], r["view"]) == (family, condition, v)
                ]
                require(
                    len(matches) == 1 and matches[0]["status"] == "ready",
                    "original lacks the corresponding family/condition/target",
                )
                stored = matches[0]["modes"][s]["file"]
                path = run.open("payload/development-scores/" + stored["path"])
                require(digest(path) == stored["sha256"], "different map in original development score table")
                measured = dest / "warm" / row["score"]["path"]
                truth_free[v + "/" + s] = {
                    "canonical_platform": "macOS CPU recovery"
                    if run.run["run_kind"] == "derivation"
                    and (s == "one" or (condition == "M4" and s == "two"))
                    else "Windows CUDA",
                    "canonical_sha256": stored["sha256"],
                    "measured_sha256": row["score"]["sha256"],
                    **compare_maps(measured, path, fit["threshold"]),
                }
            case_results[condition] = {
                "case_sha256": cs["sha256"],
                "cost": file_record(dest / "costs.json", out),
                "comparison": truth_free,
                "cold_seconds": {
                    k: v["wall_started_before_python_seconds"] for k, v in cost["cold_v01_by_scope"].items()
                },
                "warm": cost["warm_views"],
                "warm_peak_rss_bytes": cost["warm_process"]["sampled_peak_process_tree_rss_bytes"],
            }
        reps[name] = {
            "model_id": run.pin["run_id"],
            "source": run.pin,
            "model_index_sha256": fit["model_index_sha256"],
            "weights_sha256": fit["weights_sha256"],
            "threshold": fit["threshold"],
            "native_cases": case_results,
        }
    require(set(reps) == set(METHODS), "both fixed-seed neural models are required")
    return {
        "schema": "el-stage3-neural-cpu-validation-1",
        "run_id": os.environ["EL_RUN_ID"],
        "catalog_sha256": CATALOG,
        "study_sha256": STUDY,
        "development_dataset": dev.pin,
        "dataset_sha256": dsha,
        "family": family,
        "models": reps,
        "cases": [
            {
                "method": name,
                "model_sha256": r["model_index_sha256"],
                "condition": c,
                "case_sha256": v["case_sha256"],
                "original_cost": v["cost"],
            }
            for name, r in sorted(reps.items())
            for c, v in sorted(r["native_cases"].items())
        ],
        "threads": threads,
        "native_scope": "one/two/all",
    }


def main(a):
    require(
        os.environ.get("EL_RUN_KIND") == "archive_verify" and os.environ.get("EL_RUN_ID") == a.run_id,
        "run as an original archive_verify",
    )
    if a.action == "base":
        from stage3_fit_audit import verify_base

        out = output_dir(a.output)
        parent = parse_pin(B_PARENT)
        sources = read(out.parent / "inputs.json")
        require(
            sources["parents"] == [parent]
            and all(v["parent_run_id"] == parent["run_id"] for v in sources["inputs"]),
            "original published B producer required for the independent roster",
        )
        result = {
            "schema": "el-stage3-source-backbone-roster-1",
            "run_id": a.run_id,
            "B_parent": parent,
            **verify_base(a.B_index, a.B_weights),
        }
        put(out / "source-B-tensor-roster.json", result)
        print(json.dumps({"run_id": a.run_id, "sha256": digest(out / "source-B-tensor-roster.json")}))
        return
    pins = [parse_pin(s) for s in a.candidate]
    runs = [candidate(a.store, a.shelf, p) for p in pins]
    required_pins = [*pins, parse_pin(a.development)]
    if a.action == "compare":
        required_pins.extend((parse_pin(a.train), parse_pin(B_PARENT)))
    out = output_dir(a.output)
    parents = read(out.parent / "inputs.json")["parents"]
    require(
        all(any(all(q[k] == v for k, v in p.items()) for q in parents) for p in required_pins),
        "original recorder is missing a scientific parent",
    )
    if a.action == "compare":
        j = paired_evidence(
            runs,
            original(a.store, a.shelf, parse_pin(B_PARENT)),
            original(a.store, a.shelf, parse_pin(a.train)),
            original(a.store, a.shelf, parse_pin(a.development)),
            out,
            a.threads,
            a.store,
            a.shelf,
        )
        name = "tensor-comparison.json"
    else:
        j = validate_cpu(runs, original(a.store, a.shelf, parse_pin(a.development)), out, a.threads)
        name = "cpu-validation.json"
    put(out / name, j)
    print(json.dumps({"sha256": digest(out / name), "run_id": a.run_id, "receipt": name}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    first = sub.add_parser("base")
    for name in ("B-index", "B-weights", "output"):
        first.add_argument("--" + name, type=Path, required=True)
    first.add_argument("--run-id", required=True)
    for k in ("compare", "cpu"):
        s = sub.add_parser(k)
        for n in ("store", "shelf", "output"):
            s.add_argument("--" + n, type=Path, required=True)
        s.add_argument("--run-id", required=True)
        s.add_argument("--candidate", action="append", required=True)
        s.add_argument("--development", required=True)
        s.add_argument("--threads", type=int, default=4)
        if k == "compare":
            s.add_argument("--train", required=True)
    main(parser.parse_args())
