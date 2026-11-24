"""Record an actual final scoring failure and count the independently observable outcomes after its H opening."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dual_records import bind, resolve, selected, stage
from dtu_eval import read_pinned
from evidence_lens._bundle import RUN_ID, load, save, sha
from evidence_lens._verify import require
from many_data_access import EvaluatorTruth
from many_data_prepare import confirmation_terminals


def registered(store, scene, count):
    registration = selected(store, "b-data-registration")
    spec = load(resolve((store,), registration))
    row = next((r for r in spec["originals"] + spec["external"]["roster"] if r["id"] == scene), None)
    require(
        row and count in ((16, 128) if "seed" in row else [row["R"]]),
        "unregistered family or reconstruction count",
    )
    return row, spec, registration


def native_failure(store, rid):
    require(isinstance(rid, str) and RUN_ID.fullmatch(rid), "invalid failed native run id")
    roots = (store, store / "history")
    for base in roots:
        for bucket in ("runs", "quarantine"):
            d = base / bucket / rid
            if not d.is_dir():
                continue
            completion = load(d / "completion.json")
            pin = {
                "run_id": rid,
                "manifest_sha256": completion["manifest_sha256"],
                "member": "run.json",
                "sha256": sha(d / "run.json"),
            }
            native, path = read_pinned(roots, pin, success=False)
            cfg = native.root / "config.resolved.yaml"
            require(
                native.files["config.resolved.yaml"]["sha256"] == sha(cfg),
                "different failed scoring configuration",
            )
            return pin, load(path), load(cfg)
    raise ValueError("specified unsuccessful native run is unavailable")


def cutoff_condition(doc, scene, count, rid, limits):
    """An observed hard stop or insufficient remaining resources for one further registered attempt."""
    units = {"cpu": "core_seconds", "gpu": "worker_seconds", "archive": "bytes", "score_rss": "bytes"}
    keys = {
        "cpu": "max_cpu_core_seconds",
        "gpu": "max_gpu_seconds",
        "archive": "new_archive_bytes",
        "score_rss": "cpu_scoring_rss_bytes",
    }
    require(
        doc.get("schema") == "el-many-terminal-cutoff-1"
        and doc.get("family") == scene
        and doc.get("R") == count
        and doc.get("failed_native_run_id") == rid
        and doc.get("final_irrecoverable") is True
        and isinstance(doc.get("reason"), str)
        and bool(doc["reason"].strip()),
        "no final native resource disposition",
    )
    b = doc["budget"]
    kind = b["resource"]
    require(
        kind in units and b["unit"] == units[kind] and b["cap"] == limits[keys[kind]],
        "wrong registered resource limit",
    )
    used, minimum = b["observed_used"], b["minimum_for_next_valid_attempt"]
    require(
        all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x) and x >= 0
            for x in (used, minimum)
        )
        and (used > b["cap"] if kind == "score_rss" else used >= b["cap"] or used + minimum > b["cap"]),
        "remaining budget permits another registered attempt",
    )
    require(
        isinstance(doc.get("evidence"), dict)
        and all(k in doc["evidence"] for k in ("run_id", "manifest_sha256", "member", "sha256")),
        "missing native budget evidence",
    )
    return b


def verify_cutoff(store, pin, scene, count, rid, limits):
    path = resolve((store, store / "history"), pin)
    doc = load(path)
    cutoff_condition(doc, scene, count, rid, limits)
    parents = {
        (p["run_id"], p["manifest_sha256"]) for p in load(path.parents[1] / "manifest.json")["parents"]
    }
    evidence = doc["evidence"]
    require(
        (evidence["run_id"], evidence["manifest_sha256"]) in parents,
        "budget measurement is not a sealed cutoff parent",
    )
    measured = load(resolve((store, store / "history"), evidence))
    require(
        measured.get("schema") == "el-many-failure-budget-observation-1"
        and measured.get("failed_native_run_id") == rid
        and measured.get("resource_observations", {}).get(doc["budget"]["resource"])
        == doc["budget"]["observed_used"],
        "different observed resource use in the native budget evidence",
    )
    minimum = doc["budget"]["minimum_for_next_valid_attempt"]
    if minimum:
        require(
            measured.get("minimum_next_valid_attempt_by_resource", {}).get(doc["budget"]["resource"])
            == minimum,
            "estimated retry lower bound has no matching recorded measurement",
        )
    return doc


def failed_context(store, scene, count, rid):
    row, _, reg = registered(store, scene, count)
    slug = scene.replace("_", "-") + "-r" + str(count)
    matrix = "b-matrix-" + slug
    control = store / "work/controller" / matrix
    require(not (control / "selected.json").exists(), "a complete score result already exists")
    logs = sorted(control.glob("attempt-*.jsonl"))
    require(
        logs and json.loads(logs[-1].read_text().splitlines()[-1])["run_id"] == rid,
        "failed scoring id is not the latest native checkpoint attempt",
    )
    pin, native, cfg = native_failure(store, rid)
    freeze = selected(store, "b-model-freeze")
    bank = selected(store, "b-bank-" + slug)
    require(
        native["run_kind"] == "model_score"
        and native["experiment_id"] == "dual-track-b"
        and native["trial_id"] == matrix
        and cfg["sources"]["model_freeze"] == freeze
        and cfg["sources"]["bank"] == bank
        and cfg["scene"] == scene
        and cfg["R"] == count
        and cfg["split"] == row["split"],
        "actual failed score used a different bank, role or frozen model",
    )
    pins = {
        "registration": reg,
        "study": selected(store, "b-study-registration"),
        "freeze": freeze,
        "bank": bank,
    }
    return row, slug, pin, pins


def receive_score(store, scene, count, rid, cutoff_label):
    row, slug, pin, pins = failed_context(store, scene, count, rid)
    pins["cutoff"] = selected(store, cutoff_label)
    limits = load(resolve((store,), pins["study"]))["study"]["budget"]
    verify_cutoff(store, pins["cutoff"], scene, count, rid, limits)
    cmd = [
        "{python}",
        "-B",
        "{source}/scripts/many_data_terminal.py",
        "score-worker",
        "--store",
        str(store),
        "--config",
        "{config}",
        "--out",
        "{out}",
    ]
    return stage(
        store,
        "b-terminal-failure-" + slug,
        "archive_verify",
        pins,
        {
            "family": scene,
            "R": count,
            "role": row["split"],
            "failed_native_run": pin,
            "latest_score_attempt": rid,
        },
        cmd,
        "payload/failure.json",
        15000,
    )


def record_cutoff(a):
    """Snapshot the actual B resource ledger, then record an explicit hard-stop decision."""
    s = a.store
    row, slug, pin, pins = failed_context(s, a.scene, a.count, a.failed_run)
    require(
        a.resource and a.final_reason and a.final_reason.strip(),
        "actual limiting resource and final reason required",
    )
    command = ["{python}", "-B", "{source}/scripts/many_data_terminal.py"]
    tail = ["--store", str(s), "--config", "{config}", "--out", "{out}"]
    details = {"family": row["id"], "R": a.count, "failed_native_run": pin}
    stop = stage(
        s,
        "b-score-stop-observation-" + slug,
        "report",
        pins,
        details,
        [*command, "meter-worker", *tail],
        "payload/observation.json",
        10000,
    )
    pins["evidence"] = stop
    return stage(
        s,
        "b-score-stop-" + slug,
        "archive_verify",
        pins,
        {**details, "resource": a.resource, "final_reason": a.final_reason},
        [*command, "cutoff-worker", *tail],
        "payload/cutoff.json",
        10000,
    )


def budget_worker(a):
    from many_resources import account

    cfg = load(a.config)
    pin, run, _ = native_failure(a.store, cfg["failed_native_run"]["run_id"])
    require(pin == cfg["failed_native_run"], "failure changed after resource observation")
    if a.action == "meter-worker":
        use = account(a.store)
        data = {
            "cpu": use["cpu_sampled_tree_core_seconds"],
            "gpu": use["gpu_fit_worker_seconds"],
            "archive": use["unique_payload_bytes"],
            "score_rss": run["sampled_peak_tree_rss_bytes"],
        }
        save(
            a.out / "observation.json",
            {
                "schema": "el-many-failure-budget-observation-1",
                "run_id": os.environ["EL_RUN_ID"],
                "failed_native_run_id": pin["run_id"],
                "resource_observations": data,
                "scope": "Native B process tree sampled CPU, recorded fit wall, unique B payload bytes, and this unsuccessful score's sampled peak RSS.",
            },
        )
        return
    units = {"cpu": "core_seconds", "gpu": "worker_seconds", "archive": "bytes", "score_rss": "bytes"}
    keys = {
        "cpu": "max_cpu_core_seconds",
        "gpu": "max_gpu_seconds",
        "archive": "new_archive_bytes",
        "score_rss": "cpu_scoring_rss_bytes",
    }
    limits = load(resolve((a.store,), cfg["sources"]["study"]))["study"]["budget"]
    observed = load(resolve((a.store,), cfg["sources"]["evidence"]))
    k = cfg["resource"]
    decision = {
        "schema": "el-many-terminal-cutoff-1",
        "run_id": os.environ["EL_RUN_ID"],
        "family": cfg["family"],
        "R": cfg["R"],
        "failed_native_run_id": pin["run_id"],
        "final_irrecoverable": True,
        "reason": cfg["final_reason"],
        "evidence": cfg["sources"]["evidence"],
        "budget": {
            "resource": k,
            "unit": units[k],
            "cap": limits[keys[k]],
            "observed_used": observed["resource_observations"][k],
            "minimum_for_next_valid_attempt": 0,
        },
    }
    cutoff_condition(decision, cfg["family"], cfg["R"], pin["run_id"], limits)
    save(a.out / "cutoff.json", decision)


def score_worker(a):
    cfg = load(a.config)
    pins = cfg["sources"]
    s = a.store
    pin, run, original = native_failure(s, cfg["failed_native_run"]["run_id"])
    require(
        pin == cfg["failed_native_run"]
        and original["sources"]["bank"] == pins["bank"]
        and original["sources"]["model_freeze"] == pins["freeze"],
        "failed native identity changed",
    )
    study = load(resolve((s,), pins["study"]))["study"]
    end = verify_cutoff(s, pins["cutoff"], cfg["family"], cfg["R"], pin["run_id"], study["budget"])
    bank = load(resolve((s,), pins["bank"]))
    require(
        bank["scene_id"] == cfg["family"] and len(bank["reconstruction_ids"]) == cfg["R"],
        "different failed-score bank",
    )
    save(
        a.out / "failure.json",
        {
            "schema": "el-many-terminal-failure-1",
            "run_id": os.environ["EL_RUN_ID"],
            "stage": "model_score",
            "family": cfg["family"],
            "R": cfg["R"],
            "role": cfg["role"],
            "target_ids": bank["target_ids"],
            "model_freeze_sha256": pins["freeze"]["sha256"],
            "status": run["execution_status"],
            "reason": run["failure_reason"],
            "failed_native_run": pin,
            "cutoff": pins["cutoff"],
            "resource_disposition": end["budget"],
            "independent_reference_reads": 0,
            "bank": pins["bank"],
            "checkpoint_sha256": bank["checkpoint_sha256"],
        },
    )


def evaluate_terminal(s, scene, count):
    row, _, reg = registered(s, scene, count)
    slug = scene.replace("_", "-") + "-r" + str(count)
    terminal = selected(s, "b-terminal-failure-" + slug)
    end = load(resolve((s,), terminal))
    freeze = selected(s, "b-model-freeze")
    require(
        end["schema"] == "el-many-terminal-failure-1"
        and end["family"] == scene
        and end["R"] == count
        and (
            end.get("model_freeze_sha256") == freeze["sha256"]
            or (row["split"] != "confirmation" and end.get("model_freeze_sha256") is None)
        ),
        "wrong terminal family, count or frozen model",
    )
    require(
        not (s / "work/controller" / ("b-matrix-" + slug) / "selected.json").exists(),
        "complete ranks must use the registered standard evaluator",
    )
    native, original_run, _ = native_failure(s, end["failed_native_run"]["run_id"])
    require(native == end["failed_native_run"], "wrong original failure reference")
    opening = "b-confirmation-h-opening" if row["split"] == "confirmation" else "b-train-dev-h-opening"
    grant = selected(s, opening)
    gate = load(resolve((s,), grant))
    require(end["target_ids"] == gate["allowed_targets"].get(scene), "scope of H permission differs")
    if row["split"] == "confirmation":
        confirmation_terminals(gate, load(resolve((s,), reg)))
        status = gate["terminals"][scene][str(count)]
        require(
            status["status"] == "failed"
            and status["receipt"] == terminal
            and status["native_failure"] == native,
            "terminal is outside the prior unified H permission",
        )
    else:
        require(gate["phase"] == "train-development", "wrong H phase")
    truth = selected(s, "b-truth-" + scene.replace("_", "-"))
    pins = {
        "terminal": terminal,
        "freeze": freeze,
        "registration": reg,
        "opening": grant,
        "truth": truth,
        "truth_archive": bind(s, truth["run_id"], "payload/truth.zip"),
        "runtime": selected(s, "b-score-runtime"),
    }
    bankpath = s / "work/controller" / ("b-bank-" + slug) / "selected.json"
    if bankpath.exists():
        pins["bank"] = selected(s, "b-bank-" + slug)
    if end.get("stage") == "model_score":
        require(pins.get("bank") == end["bank"], "missing independent predicted RGB after scoring failure")
        require(original_run["run_kind"] == "model_score", "different original unsuccessful scoring kind")
    else:
        pins["fit_ticket"] = end["ticket"]
        dispatch = load(resolve((s,), pins["fit_ticket"]))
        source = load(resolve((s,), truth))["inputs"]["source"]
        require(
            original_run["run_kind"] == "model_fit"
            and original_run["trial_id"] == dispatch["trial"]
            and dispatch["scene_id"] == scene
            and len(dispatch["R"]) == count
            and dispatch["H_poses_only"] == end["target_ids"]
            and dispatch["data"]["run_id"] == source["run_id"]
            and dispatch["data"]["manifest_sha256"] == source["manifest_sha256"],
            "failed native fit used a different source acquisition or H roster",
        )
    runtime = load(resolve((s,), pins["runtime"]))
    cmd = [
        str(s / runtime["relative_executable"]),
        "-I",
        "-B",
        "{source}/scripts/many_data_terminal.py",
        "eval-worker",
        "--store",
        str(s),
        "--config",
        "{config}",
        "--out",
        "{out}",
    ]
    return stage(
        s,
        "b-eval-" + slug,
        "evaluation",
        pins,
        {
            "family": scene,
            "R": count,
            "role": row["split"],
            "terminal_stage": end.get("stage", "model_fit"),
            "scope": "real independently decoded H, include any surviving verified native predictions",
        },
        cmd,
        "payload/evaluation.json",
        15000000,
    )


def failed_view(reference, prediction, pred_hash, ref_hash, reason):
    from evidence_lens_many.evaluation import labels, mask_id

    rgb = prediction["rgb"] if prediction is not None else np.full_like(reference, np.nan)
    ref, valid, errors, mae = labels(rgb, reference)
    n = int(valid.sum())
    entry = {
        "status": "terminal_ranks_unavailable",
        "reference": int(ref.sum()),
        "labelled": n,
        "errors": int(errors.sum()),
        "prediction_unavailable": int((ref & ~valid).sum()),
        "error_rate": int(errors.sum()) / n if n else None,
        "mae_sum": float(mae[valid].sum(dtype=np.float64)),
        "prediction_sha256": pred_hash,
        "source_reference_sha256": ref_hash,
        "labels_sha256": mask_id(errors),
        "valid_labels_sha256": mask_id(valid),
        "p1": None,
        "p2": None,
        "priority": {},
        "queries": {},
        "mechanism": {},
        "visibility": None,
        "nearest_depth_visibility_approximation": None,
        "missing_reason": reason,
        "ranks_unavailable": int(ref.sum()),
        "domain_status": "unavailable_final_native_failure",
    }
    return entry, ref, valid, errors


def unavailable_domains(view, count, ref, valid, errors, reason):
    from evidence_lens_many.evaluation import ranking_metrics

    rank = np.full(ref.shape, np.nan, np.float32)
    m = ranking_metrics(rank, ref, valid, errors)
    common = ["coverage", "alpha_missing", "residual", "B-CPU-LR"]
    budgets = sorted({4, 8, 16, min(32, count), count})
    cells = ["coverage-" + str(k) for k in budgets] + ["fixed-4", "nearest-4"]
    return [
        {
            "target": view,
            "cell": cell,
            "method": name,
            "domain": domain,
            "domain_status": "terminal_ranks_unavailable",
            "terminal_reason": reason,
            "unknown_reasons": {"255": int(ref.size)},
            **m,
        }
        for cell in cells
        for name in (common + ["MS-L1", "MS-ft"] if cell == "coverage-4" else common)
        for domain in ("own", "same_query_methods", "all_checkpoint_ranks")
    ]


def eval_worker(a):
    from evidence_lens_many import Bank

    cfg = load(a.config)
    p = cfg["sources"]
    s = a.store
    end = load(resolve((s,), p["terminal"]))
    freeze = load(resolve((s,), p["freeze"]))
    truthpath = resolve((s,), p["truth"])
    meta = load(truthpath)
    grant = load(resolve((s,), p["opening"]))
    require(
        meta["opening_sha256"] == p["opening"]["sha256"]
        and meta["split"] == cfg["role"]
        and meta["scene_id"] == cfg["family"]
        and meta["H"] == end["target_ids"] == grant["allowed_targets"][cfg["family"]],
        "different reference roster or permission",
    )
    if cfg["role"] == "confirmation":
        confirmation_terminals(grant, load(resolve((s,), p["registration"])))
        require(
            grant["terminals"][cfg["family"]][str(cfg["R"])]["receipt"] == p["terminal"], "unopened terminal"
        )
    bank = Bank(resolve((s,), p["bank"]).parent, p["bank"]["sha256"]) if "bank" in p else None
    archive = EvaluatorTruth(truthpath, p["truth"]["sha256"], bank=bank)
    rows = {}
    domains = []
    for view in meta["H"]:
        reference = archive.target_rgb(view)
        prediction = None
        pred_hash = None
        reason = end["reason"]
        if bank is not None:
            try:
                prediction = bank.target(view)
                if prediction is not None:
                    pred_hash = bank.meta["targets"][view]["render"]["sha256"]
            except (OSError, ValueError) as exc:
                reason = reason + "; original target prediction unavailable: " + type(exc).__name__
        observed, ref, valid, errors = failed_view(
            reference, prediction, pred_hash, meta["targets"][view]["analysis_rgb"]["sha256"], reason
        )
        rows[view] = observed
        domains.extend(unavailable_domains(view, cfg["R"], ref, valid, errors, reason))
    save(
        a.out / "rank-domains.json",
        {
            "schema": "el-many-domain-metrics-1",
            "rows": domains,
            "special_unknown_reason_codes": {"255": "entire finalized ranking job unavailable"},
        },
    )
    save(
        a.out / "evaluation.json",
        {
            "schema": "el-many-checkpoint-evaluation-1",
            "run_id": os.environ["EL_RUN_ID"],
            "scene": cfg["family"],
            "role": cfg["role"],
            "R": cfg["R"],
            "mode": "matrix",
            "rule": freeze["strong_simple_rule"],
            "status": "terminal_failure",
            "failure_stage": cfg["terminal_stage"],
            "terminal_receipt": p["terminal"],
            "failed_native_run": end["failed_native_run"],
            "native_reason": end["reason"],
            "scores_sha256": None,
            "bank_sha256": None if bank is None else bank.sha256,
            "truth_sha256": p["truth"]["sha256"],
            "opening_sha256": p["opening"]["sha256"],
            "views": rows,
            "domain_results_sha256": sha(a.out / "rank-domains.json"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "score-terminal",
            "evaluate",
            "score-worker",
            "eval-worker",
            "record-cutoff",
            "cutoff-worker",
            "meter-worker",
        ),
    )
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--scene")
    parser.add_argument("--count", type=int)
    parser.add_argument("--failed-run")
    parser.add_argument("--cutoff-label")
    parser.add_argument("--resource", choices=("cpu", "gpu", "archive", "score_rss"))
    parser.add_argument("--final-reason")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    a = parser.parse_args()
    a.store = a.store.resolve()
    if a.action == "score-terminal":
        print(json.dumps(receive_score(a.store, a.scene, a.count, a.failed_run, a.cutoff_label)), flush=True)
    elif a.action == "record-cutoff":
        print(json.dumps(record_cutoff(a)), flush=True)
    elif a.action == "evaluate":
        print(json.dumps(evaluate_terminal(a.store, a.scene, a.count)), flush=True)
    else:
        require(os.environ.get("EL_RUN_ID"), "score terminal and evaluation must be recorded")
        if a.action in ("cutoff-worker", "meter-worker"):
            budget_worker(a)
        else:
            (score_worker if a.action == "score-worker" else eval_worker)(a)
