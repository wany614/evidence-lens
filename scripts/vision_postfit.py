"""Record deployable LR, joint development metrics and a single opening-ready method freeze."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from evidence_lens._bundle import load
from evidence_lens.formal_inputs import compact_pin, require
from evidence_lens.scoring_io import file_hash
from vision_pipeline import EVAL, DEFAULT_STUDY_SHA, ORIGINAL_CATALOG_SHA
from vision_record import Recorder, parse_pin, publish
from vision_sources import signed
from vision_method import candidate, TWO


def stage_input(rec, pin, name):
    p, s = rec.asset(pin, name)
    return p, s, (p, pin["run_id"])


def original_stages(state):
    keys = {
        "train_data": "full-train-combined",
        "dev_data": "full-development-combined",
        "baseline_fit": "full-lr-fit",
        "fitted_baseline_scores": "full-dev-baseline-scores",
        "fitted_baseline_thresholds": "full-dev-baseline-thresholds",
    }
    return {role: load(state / name / "selected.json") for role, name in keys.items()}


def finalize_lr(rec, pins, orig, new):
    origins = {"original_catalog": (orig, None), "catalog": (new, None)}
    cats = [
        "--catalog",
        "{input:original_catalog}",
        ORIGINAL_CATALOG_SHA,
        "--catalog",
        "{input:catalog}",
        rec.catalog_sha,
    ]
    fit, fit_sha, fi = stage_input(rec, pins["baseline_fit"], "model/model.json")
    d, ds, di = stage_input(rec, pins["dev_data"], "dataset.json")
    th, th_sha, ti = stage_input(rec, pins["fitted_baseline_thresholds"], "thresholds.json")
    lr = rec.stage(
        "postfit-lr-deploy",
        "threshold_select",
        [pins[k] for k in ("baseline_fit", "dev_data", "fitted_baseline_thresholds")],
        {"fitted_model": fi, "development": di, "thresholds": ti},
        EVAL + ["vision_method.py", "vision_replay.py"],
        [
            "vision_method.py",
            "export-lr",
            "--fit",
            str(fit.parent),
            "--thresholds",
            "{input:thresholds}",
            "--development",
            str(d.parent),
            "--dev-sha",
            ds,
            "--run-id",
            "{run_id}",
            "--output",
            "{out}",
        ],
        "lr-export.json",
        200000,
    )
    model, model_sha, mi = stage_input(rec, lr, "model/model.json")
    sc = rec.stage(
        "postfit-dev-five-scores",
        "model_score",
        [lr, pins["dev_data"]],
        {"model": mi, "dataset": di},
        EVAL,
        [
            "vision_baselines.py",
            "score",
            "--model",
            str(model.parent),
            "--model-sha256",
            model_sha,
            "--dataset",
            str(d.parent),
            "--sha256",
            ds,
            "--split",
            "development",
            "--output",
            "{out}",
        ],
        "scores.json",
        220000000,
    )
    sp, ss, si = stage_input(rec, sc, "scores.json")
    newth = rec.stage(
        "postfit-dev-five-thresholds",
        "threshold_select",
        [sc, pins["dev_data"]],
        {"scores": si, "dataset": di, **origins},
        EVAL,
        [
            "vision_evaluate.py",
            "freeze-dev",
            "--baseline-only",
            "--scores",
            str(sp),
            ss,
            "--dataset",
            str(d.parent),
            "--sha256",
            ds,
            *cats,
            "--run-id",
            "{run_id}",
            "--output",
            "{out}",
        ],
        "thresholds.json",
        200000,
    )
    newp, _ = rec.asset(newth, "thresholds.json")
    before, after = load(th), load(newp)
    require(
        before["thresholds"] == after["thresholds"] and before["choices"] == after["choices"],
        "deployable LR packaging changed original full-field results",
    )
    return {**pins, "lr_export": lr, "baseline_scores": sc, "baseline_thresholds": newth}


def candidates_for(rec, raw, sha, pins):
    from evidence_lens_vision.selection import choose

    source = signed(raw, sha)
    require(
        source["schema"] == "el-vision-candidate-roster-1"
        and source["stage"] == "primary"
        and len(source["runs"]) == 8,
        "all eight genuine formal training pins required",
    )
    _, train_sha = rec.asset(pins["train_data"], "dataset.json")
    _, dev_sha = rec.asset(pins["dev_data"], "dataset.json")
    allc = [candidate(rec.roots, ref, train_sha, dev_sha) for ref in source["runs"]]
    selected = choose([c["fit"] for c in allc], "primary")
    by = {c["fit"]["run_id"]: c for c in allc}
    return [compact_pin(c["fit"]) for c in allc], {m: by[selected["chosen"][m]["run_id"]] for m in TWO}


def dev_metrics(rec, pins, chosen, orig, new):
    d, ds, di = stage_input(rec, pins["dev_data"], "dataset.json")
    five, fsha, fi = stage_input(rec, pins["baseline_scores"], "scores.json")
    prior, psha, pi = stage_input(rec, pins["baseline_thresholds"], "thresholds.json")
    args = [
        "--dataset",
        str(d.parent),
        "--sha256",
        ds,
        "--scores",
        str(five),
        fsha,
        "--catalog",
        "{input:original_catalog}",
        ORIGINAL_CATALOG_SHA,
        "--catalog",
        "{input:catalog}",
        rec.catalog_sha,
    ]
    inputs = {"dataset": di, "original_scores": fi, "original_catalog": (orig, None), "catalog": (new, None)}
    parents = [pins["dev_data"], pins["baseline_scores"]]
    for i, m in enumerate(TWO):
        c = chosen[m]
        p, s = c["score_pin"]
        par = compact_pin(c["fit"])
        parents.append(par)
        inputs[f"vision_scores_{i}"] = (p, par["run_id"])
        args += ["--scores", str(p), s]
    freeze = rec.stage(
        "postfit-dev-seven-thresholds",
        "threshold_select",
        [*parents, pins["baseline_thresholds"]],
        {**inputs, "prior": pi},
        EVAL,
        [
            "vision_evaluate.py",
            "freeze-dev",
            *args,
            "--baseline-freeze",
            "{input:prior}",
            psha,
            "--run-id",
            "{run_id}",
            "--output",
            "{out}",
        ],
        "thresholds.json",
        200000,
    )
    threshold, tsha, ti = stage_input(rec, freeze, "thresholds.json")
    t = load(threshold)
    for m in TWO:
        require(
            t["thresholds"][m] == chosen[m]["fit"]["threshold"],
            "original device development threshold differs: " + m,
        )
    evaluation = rec.stage(
        "postfit-dev-seven-evaluation",
        "evaluation",
        [*parents, freeze],
        {**inputs, "thresholds": ti},
        EVAL,
        [
            "vision_evaluate.py",
            "evaluate",
            *args,
            "--split",
            "development",
            "--thresholds",
            "{input:thresholds}",
            tsha,
            "--output",
            "{out}",
        ],
        "evaluation.json",
        20000000,
    )
    return {**pins, "thresholds": freeze, "development_evaluation": evaluation}


def recorded_method(rec, roster, study, catalog, parents):
    """The opening verifier requires this exact experiment identity and direct recorder parents."""
    dr = rec.state / "postfit-method-freeze"
    dr.mkdir(parents=True, exist_ok=True)
    rp = dr / "input-roster.json"
    publish(rp, roster)
    body = {
        "schema": "el-stage2-method-freeze-config-1",
        "experiment_id": "stage2-method-freeze",
        "trial_id": "v11-complete-method-freeze",
        "dataset_hash": rec.catalog_sha,
        "roster_sha256": file_hash(rp),
        "study_sha256": rec.study_sha,
        "parents": sorted((compact_pin(p) for p in parents), key=lambda p: p["run_id"]),
        "recording": {"expected_output_bytes": 3000000},
    }
    conf = dr / "config.json"
    publish(conf, body)
    out = dr / "selected.json"
    with (dr / "lock").open("ab") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if out.exists():
            val = load(out)
            p, sha = rec.asset(val, "method-freeze.json")
            require(
                val["method_sha256"] == sha and load(p)["input_roster_sha256"] == body["roster_sha256"],
                "freeze selection differs",
            )
            return val
        attempts = sorted(dr.glob("attempt-*.log"))
        retry = None
        completed = None
        if attempts:
            try:
                completed = json.loads(attempts[-1].read_text().strip().splitlines()[-1])
            except (ValueError, IndexError) as exc:
                raise ValueError("inspect interrupted freeze recorder " + str(attempts[-1])) from exc
            if completed["execution_status"] != "succeeded":
                retry = completed["run_id"]
                require(
                    rec.retry.get("postfit-method-freeze") == retry,
                    "A recorded freeze failed. Supply --retry postfit-method-freeze=" + retry,
                )
        log = attempts[-1] if completed and not retry else dr / f"attempt-{len(attempts) + 1:02d}.log"
        if not log.exists():
            argv = [
                sys.executable,
                "-m",
                "evidence_lens.recorder",
                "--root",
                str(rec.store),
                "--source-root",
                str(rec.project),
                "--kind",
                "report",
                "--config",
                str(conf),
                "--input",
                "method_roster=" + str(rp),
                "--input",
                "study=" + str(study),
                "--input",
                "catalog=" + str(catalog),
            ]
            if retry:
                argv += ["--retry-of", retry]
            for p in body["parents"]:
                argv += ["--parent", p["run_id"] + "=" + p["manifest_sha256"]]
            argv += [
                "--",
                "{python}",
                "{source}/scripts/vision_method.py",
                "freeze",
                "--store",
                str(rec.store),
                "--shelf",
                str(rec.shelf),
                "--roster",
                "{input:method_roster}",
                "--roster-sha",
                body["roster_sha256"],
                "--study",
                "{input:study}",
                "--study-sha",
                rec.study_sha,
                "--catalog",
                "{input:catalog}",
                "--run-id",
                "{run_id}",
                "--output",
                "{out}",
            ]
            with log.open("xb") as stream:
                subprocess.run(
                    argv,
                    cwd=rec.project,
                    env={**os.environ, "PYTHONPATH": str(rec.project / "src")},
                    stdout=stream,
                    stderr=stream,
                    pass_fds=(lock.fileno(),),
                )
        try:
            done = json.loads(log.read_text().strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise ValueError("inspect the interrupted recorder output " + str(log)) from exc
        require(done["execution_status"] == "succeeded", "method freeze did not complete: " + done["run_id"])
        pin = compact_pin(done)
        p, sha = rec.asset(pin, "method-freeze.json")
        require(load(p)["input_roster_sha256"] == body["roster_sha256"], "sealed method uses another roster")
        val = {**pin, "method_sha256": sha, "method_member": "payload/method-freeze.json"}
        publish(out, val)
        return val


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("finalize-lr", "development", "freeze", "open"))
    p.add_argument("--project", type=Path, default=Path.cwd())
    p.add_argument("--store", type=Path, default=Path(".research"))
    p.add_argument("--shelf", type=Path)
    p.add_argument("--study-sha", default=DEFAULT_STUDY_SHA)
    p.add_argument("--candidate-roster", nargs=2, metavar=("CANDIDATES_JSON", "SHA256"))
    p.add_argument("--cost-frozen")
    p.add_argument("--cost-ft")
    p.add_argument("--cost-family")
    p.add_argument(
        "--baseline-qa",
        default="20261123T131443217184Z-0d6f62ab5d23456d=fa7456507bd9ae0a3e22f8b474de1a232a95ad5b75c2411db4881903fd058d9b",
        help="independent selected 48-view baseline numerical readback",
    )
    p.add_argument("--retry", action="append", default=[], help="stage=failed run ID")
    a = p.parse_args()
    project = a.project.resolve()
    store = a.store.resolve()
    study = project / "protocol/stage2-training-v1.json"
    sc = signed(study, a.study_sha)
    catalog = project / "protocol/stage2-families-v1.json"
    signed(catalog, sc["catalog_sha256"])
    state = store / "stage2" / sc["catalog_sha256"] / "learning"
    rec = Recorder(
        project,
        store,
        a.shelf or store.parent / "cold-shelf",
        state,
        sc["catalog_sha256"],
        a.study_sha,
        dict(x.split("=", 1) for x in a.retry),
    )
    if a.action == "open":
        from stage2_opening import open_cohort

        pin = load(state / "postfit-method-freeze/selected.json")
        result = open_cohort(store, catalog, rec.catalog_sha, pin["run_id"], pin["manifest_sha256"])
    else:
        old = project / "protocol/families-v1.json"
        stages = finalize_lr(rec, original_stages(state), old, catalog)
        if a.action == "finalize-lr":
            result = stages["lr_export"]
        else:
            require(a.candidate_roster, "supply all eight original candidate pins")
            allpins, chosen = candidates_for(rec, *a.candidate_roster, stages)
            stages = dev_metrics(rec, stages, chosen, old, catalog)
            if a.action == "development":
                result = {k: stages[k] for k in ("thresholds", "development_evaluation")}
            else:
                require(
                    a.cost_frozen and a.cost_ft and a.cost_family,
                    "two original primary CPU cost runs and dev family required",
                )
                costs = dict(zip(TWO, [parse_pin(a.cost_frozen), parse_pin(a.cost_ft)]))
                lst = {
                    "schema": "el-vision-postfit-roster-1",
                    "study_sha256": rec.study_sha,
                    "catalog_sha256": rec.catalog_sha,
                    "stages": {k: compact_pin(v) for k, v in stages.items()},
                    "candidate_runs": allpins,
                    "cpu_costs": costs,
                    "numerical_qa": parse_pin(a.baseline_qa),
                    "cost_family": a.cost_family,
                }
                pars = {
                    v["run_id"]: v for v in [*stages.values(), *allpins, *costs.values(), lst["numerical_qa"]]
                }
                result = recorded_method(rec, lst, study, catalog, list(pars.values()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
