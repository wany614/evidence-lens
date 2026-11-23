"""Record the model fit, development thresholds and condition-specific CPU scoring; freeze a method once."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from evidence_lens._bundle import RUN_ID, encoded, load, save, sha
from evidence_lens._index import writer_gate
from evidence_lens._verify import require, verify_bundle
from evidence_lens.formal_inputs import PinnedRun, hash_json, protocol
from evidence_lens.l1 import load_package
from evidence_lens.scoring_io import TARGETS
from formal_batch import dataset
from formal_lineage import VerifiedRuns, family_record, heldout_method
from mac_trial import select_files
from make_handoff import publish


ROOT = Path(__file__).resolve().parents[1]
METHOD_SOURCES = ("scripts/train_l1.py", "scripts/formal_learning.py", "src/evidence_lens/evaluation.py",
                  "src/evidence_lens/features.py", "src/evidence_lens/risk_model.py",
                  "src/evidence_lens/scoring_io.py", "src/evidence_lens/l1.py", "src/evidence_lens/l1_data.py",
                  "src/evidence_lens/l1_fit.py", "src/evidence_lens/l1_freeze.py")


def run_pin(ref):
    return {k: ref[k] for k in ("run_id", "manifest_sha256")}


def sealed_stage(store, ref, kind):
    rid = ref["run_id"]
    require(isinstance(rid, str) and RUN_ID.fullmatch(rid), "invalid selected run ID")
    checked = verify_bundle(store / "runs" / rid)
    require(checked["run"]["execution_status"] == "succeeded" and checked["run"]["run_kind"] == kind
            and checked["manifest_sha256"] == ref["manifest_sha256"], "selected stage is not the pinned successful run")
    return checked


def source_digests():
    return {name: sha(ROOT / name) for name in METHOD_SOURCES}


def record_once(store, pointer, config, parents, inputs, worker, *, retry=None, adopt=None):
    """Publish one immutable selection and recover an interrupted publication by its explicit run ID."""
    trial, experiment, kind = (config[k] for k in ("trial_id", "experiment_id", "run_kind"))
    request = hash_json(config)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pending = pointer.with_suffix(".intent.json")
    with writer_gate(pointer.parent / ".locks" / pointer.stem):
        if pointer.exists():
            selected = load(pointer)
            require(selected["request_sha256"] == request and (not adopt or adopt == selected["run_id"]),
                    "the first recorded selection uses a different request")
            sealed_stage(store, selected, kind)
            return selected
        intent = {"schema": "el-scientific-stage-intent-1", "trial_id": trial, "experiment_id": experiment,
                  "run_kind": kind, "request_sha256": request}
        if pending.exists():
            require(load(pending) == intent, "conflicting scientific stage intent")
        if adopt:
            require(RUN_ID.fullmatch(adopt), "invalid recovery run ID")
            rid = adopt
        else:
            # Only prior run metadata is searched, only after a durable same-request reservation.
            # A worker that completed but lost its controller can be adopted without rerunning it.
            prior = []
            if pending.exists():
                for filename in (store / "work").glob("*/config.resolved.yaml"):
                    require(load(filename).get("request_sha256") != request,
                            f"the matching recorder worker is still unsealed under {filename.parent.name}")
                for filename in (store / "runs").glob("*/run.json"):
                    candidate = load(filename)
                    if (candidate["trial_id"], candidate["experiment_id"], candidate["run_kind"]) == (trial, experiment, kind):
                        cfg = load(filename.parent / "config.resolved.yaml")
                        require(cfg.get("request_sha256") == request, "conflicting earlier recorded request")
                        prior.append(candidate)
            success = [r for r in prior if r["execution_status"] == "succeeded"]
            require(len(success) <= 1, "multiple successful runs require an explicit provenance review")
            if success:
                rid = success[0]["run_id"]
            else:
                failed = [r for r in prior if r.get("worker_exit_code") is not None]
                require((not failed and retry is None) or (len(failed) == 1 and failed[0]["run_id"] == retry),
                        "an executed failure requires its explicit retry ID")
                publish(pending, encoded(intent))
                with tempfile.TemporaryDirectory(dir=store, prefix="learning-recorder-") as temp:
                    cfg = Path(temp) / "config.json"
                    save(cfg, dict(config, request_sha256=request))
                    cmd = [sys.executable, "-m", "evidence_lens.recorder", "--root", str(store), "--source-root", str(ROOT),
                           "--config", str(cfg), "--kind", kind, "--experiment", experiment, "--trial", trial]
                    for p in parents:
                        cmd.extend(["--parent", p["run_id"] + "=" + p["manifest_sha256"]])
                    for role, (path, parent) in inputs.items():
                        cmd.extend(["--input", role + "=" + str(path)])
                        if parent:
                            cmd.extend(["--input-parent", role + "=" + parent])
                    if retry:
                        cmd.extend(["--retry-of", retry])
                    result = subprocess.run([*cmd, "--", *worker], cwd=ROOT, text=True, stdout=subprocess.PIPE)
                    require(result.returncode == 0, f"{kind} failed. Recorder result: {result.stdout.strip()}")
                    rid = json.loads(result.stdout)["run_id"]
        run_dir = store / "runs" / rid
        ref = {"schema": "el-scientific-stage-selection-1", "run_id": rid, "manifest_sha256": sha(run_dir / "manifest.json"),
               "run_kind": kind, "trial_id": trial, "experiment_id": experiment, "request_sha256": request}
        checked = sealed_stage(store, ref, kind)
        require(checked["run"]["trial_id"] == trial and checked["run"]["experiment_id"] == experiment
                and load(run_dir / "config.resolved.yaml").get("request_sha256") == request
                and sorted(checked["inputs"]["parents"], key=lambda r: r["run_id"]) == sorted(parents, key=lambda r: r["run_id"]),
                "recorded identity or parent pins disagree with this selection")
        receipt_name, id_key, statuses = {
            "model_fit": ("fit.json", "fit_run_id", {"fitted"}),
            "threshold_select": ("development.json", "run_id", {"frozen"}),
            "model_score": ("l1_score.json", "run_id", {"scored", "unsupported_l1_depth", "unsupported"}),
            "archive_verify": ("development_parity.json", "run_id", {"passed"}),
        }[kind]
        receipt = load(run_dir / "payload" / receipt_name)
        require(receipt[id_key] == rid and receipt["status"] in statuses, "scientific receipt did not complete")
        ref["receipt_sha256"] = sha(run_dir / "payload" / receipt_name)
        if kind in ("model_fit", "threshold_select"):
            model_hash = sha(run_dir / "payload/model/model.json")
            require(receipt["model_index_sha256"] == model_hash, "model index not bound in scientific receipt")
            ref["model_index_sha256"] = model_hash
        publish(pending, encoded(intent))
        publish(pointer, encoded(ref))
        return ref


def record_learning(store, catalog, freeze_file, split, python, *, fit_run=None, retry=None, adopt=None):
    digest = sha(catalog)
    control = store / "formal" / digest
    directory = control / "learning"
    experiment, kind = ("p4-model-fit", "model_fit") if split == "train" else ("p5-threshold-select", "threshold_select")
    cmd = [str(python.absolute()), "{source}/scripts/train_l1.py", "fit" if split == "train" else "freeze",
           "--index", "{input:index}", "--catalog", "{input:catalog}", "--output", "{out}", "--run-id", "{run_id}"]
    with tempfile.TemporaryDirectory(dir=store, prefix="learning-split-") as temp:
        compiled = Path(temp) / "dataset"
        result = dataset(store, catalog, freeze_file, split, compiled)
        index = load(compiled / "dataset.json")
        identity = {"dataset_id": load(catalog)["dataset_id"], "dataset_hash": digest,
                    "split": "train" if split == "train" else "development", "condition_id": "B0"}
        inputs = {"index": (compiled / "dataset.json", None), "catalog": (catalog, None), "protocol-freeze": (freeze_file, None)}
        parents, sources = [], {}
        for n, item in enumerate(index["families"]):
            name = item["scene_family_id"]
            ref = load(control / "extracts" / (name + ".json"))
            require(ref["extract_manifest_sha256"] == item["extract_manifest_sha256"], "split and producer disagree")
            parents.append(run_pin(ref))
            role = f"extract-{n:02d}"
            sources[name] = run_pin(ref)
            inputs[role] = (compiled / name, ref["run_id"])
            cmd.extend(["--extract", name + "={input:" + role + "}"])
        config = {"schema": "el-recorded-learning-1", "identity": identity, "run_kind": kind, "trial_id": experiment + "-v1",
                  "experiment_id": experiment, "dataset_sha256": result["dataset_sha256"], "formal_freeze_sha256": sha(freeze_file), "extract_parents": sources,
                  "method_source_sha256": source_digests()}
        if split == "dev":
            fit = load(directory / "fit.json")
            require(fit["run_id"] == fit_run, "threshold selection must use the explicitly named selected fit")
            sealed_stage(store, fit, "model_fit")
            model = store / "runs" / fit_run / "payload/model"
            load_package(model, require_frozen=False, expected_index_sha256=fit["model_index_sha256"])
            config["fit"] = run_pin(fit)
            parents.append(run_pin(fit))
            inputs["fit"] = (model, fit_run)
            cmd.extend(["--fit", "{input:fit}"])
        return record_once(store, directory / ("fit.json" if split == "train" else "threshold.json"),
                           config, parents, inputs, cmd, retry=retry, adopt=adopt)


def freeze_method(store, catalog, freeze_file, study, rid, code_commit):
    digest = sha(catalog)
    selection = load(store / "formal" / digest / "learning/threshold.json")
    require(rid == selection["run_id"] and re.fullmatch(r"[0-9a-f]{40}", code_commit), "invalid selected threshold or method commit")
    threshold = sealed_stage(store, selection, "threshold_select")
    package_dir = store / "runs" / rid / "payload/model"
    package = load_package(package_dir, expected_index_sha256=selection["model_index_sha256"])
    index = package["index"]
    study_hash = sha(study)
    lock = load(freeze_file)
    require(lock["catalog_sha256"] == digest == index["split_manifest_sha256"] and lock["preregistration_sha256"] == study_hash
            and threshold["run"]["dataset_hash"] == digest and index["threshold_run_id"] == rid,
            "method differs from the registered dataset, thresholds or study")
    comparison = protocol(ROOT)
    sources = source_digests()
    registered_files = {"protocol/families-v1.json": digest, "configs/formal-freeze-v1.json": sha(freeze_file)}
    for name, expected in {**sources, **comparison["files"], **registered_files}.items():
        tracked = subprocess.check_output(["git", "-C", str(ROOT), "show", code_commit + ":" + name])
        require(hashlib.sha256(tracked).hexdigest() == expected, f"method commit differs from the working {name}")
    fit = load(store / "formal" / digest / "learning/fit.json")
    require(fit["run_id"] == index["fit_run_id"] and run_pin(fit) in threshold["inputs"]["parents"], "thresholds do not bind the selected fit")
    for pinned in (fit, selection):
        stage = store / "runs" / pinned["run_id"]
        cfg = load(stage / "config.resolved.yaml")
        require(cfg["method_source_sha256"] == sources and cfg["formal_freeze_sha256"] == registered_files["configs/formal-freeze-v1.json"]
                and all(sha(stage / "source" / name) == expected for name, expected in sources.items()),
                "executed learning source differs from the method freeze")
    method = {"schema": "el-method-freeze-1", "catalog_sha256": digest, "study_sha256": study_hash,
              "formal_freeze_sha256": registered_files["configs/formal-freeze-v1.json"],
              "model_id": rid, "threshold_select_run_id": rid, "threshold_select_manifest_sha256": selection["manifest_sha256"],
              "model_fit_run_id": fit["run_id"], "model_fit_manifest_sha256": fit["manifest_sha256"],
              "model_index_sha256": selection["model_index_sha256"], "thresholds_sha256": index["files"]["thresholds.json"],
              "method_code_commit": code_commit, "method_source_sha256": sources, "comparison_protocol_sha256": comparison["comparison_protocol_sha256"],
              "comparison_files": comparison["files"], "development_receipt_sha256": selection["receipt_sha256"]}
    output = store / "protocol/method-freeze-v1.json"
    publish(output, encoded(method))
    return {"method_freeze": str(output), "method_freeze_sha256": sha(output), "model_id": rid}


def record_score(store, catalog, freeze_file, study, method_file, family_id, condition, python, *,
                 retry=None, adopt=None, repeat=None, development_parity=False):
    family = family_record(catalog, freeze_file, family_id)
    if development_parity:
        first = min(r["scene_family_id"] for r in load(catalog)["families"] if r["split"] == "development")
        require(family_id == first and condition == "B0" and repeat is None, "deployment parity uses the first registered development family")
    else:
        require(family["split"] == "test" and condition in ("B0", "R1", "Q1"), "formal comparison uses a registered held-out condition")
    if repeat is not None:
        timing = load(study)["timing_repeats"]
        test_names = sorted(r["scene_family_id"] for r in load(catalog)["families"] if r["split"] == "test")
        require(timing["family"] == "lexicographically first test family" and family_id == test_names[0]
                and repeat in range(1, timing["repetitions"] + 1) and condition in timing["conditions"], "unregistered timing replicate")
    # Check the method before opening the selected trial or any condition inputs.
    gate = heldout_method(VerifiedRuns(store), family, method_file, study, freeze_file)
    method = load(method_file)
    require(method["formal_freeze_sha256"] == sha(freeze_file) and method["method_source_sha256"] == source_digests()
            and method["comparison_protocol_sha256"] == protocol(ROOT)["comparison_protocol_sha256"],
            "deployable method or comparator changed after freeze")
    row = next(r for r in load(catalog)["families"] if r["scene_family_id"] == family_id)
    trial = row["trials"][condition]
    selected_file = store / "trials" / trial / "trial_selection.json"
    s = load(selected_file)
    schema = "el-p2-trial-selection-1" if condition == "B0" else "el-condition-selection-1"
    require(s.get("schema") == schema and s["trial_id"] == trial and s["selection_status"] == "selected" and s["scene_family_id"] == family_id
            and s["scene_id"] == family["scene_id"] and s["condition_id"] == condition, "condition differs from pinned trial")
    render_trial = trial
    render_ref = run_pin(s["selected"]["render"])
    producer = s["selected"]["generation" if condition == "B0" else "producer"]["run_id"]
    timing_file = None
    if repeat is not None:
        render_trial = f"{trial}-timing-{repeat:02}"
        timing_file = store / "formal" / family["dataset_hash"] / "conditions/timing" / (render_trial + ".json")
        timing_marker = load(timing_file)
        require(timing_marker.get("schema") == "el-timing-repeat-1" and timing_marker["trial_id"] == render_trial
                and timing_marker["timing_repeat"] == repeat and timing_marker["scene_family_id"] == family_id
                and timing_marker["condition_id"] == condition and timing_marker["catalog_sha256"] == family["dataset_hash"]
                and timing_marker["method_freeze_sha256"] == gate["method_freeze_sha256"]
                and timing_marker["included_in_quality_or_training"] is False
                and timing_marker["warmup"]["selection_sha256"] == sha(selected_file)
                and timing_marker["render"]["run_id"] != render_ref["run_id"],
                "timed CPU scoring requires this repeat's independent Windows render")
        render_ref = run_pin(timing_marker["render"])
        producer = timing_marker["producer"]["run_id"]
    source_ref = run_pin(s["selected"]["preparation" if condition == "R1" else "reference"])
    model_ref = {"run_id": gate["threshold_select_run_id"], "manifest_sha256": gate["threshold_select_manifest_sha256"]}
    rr, sr, mr = (PinnedRun(store, pin) for pin in (render_ref, source_ref, model_ref))
    meta = load(rr.open("payload/prediction.json"))
    image, cameras = sr.open("payload/input.png"), sr.open("payload/cameras.json")
    expected_source_kind = "derivation" if condition == "R1" else "scene_render"
    require(sr.run["run_kind"] == expected_source_kind and sr.run["scene_family_id"] == family_id
            and sr.run["dataset_hash"] == family["dataset_hash"] and sr.run["scene_id"] == family["scene_id"]
            and set(rr.run["parent_run_ids"]) == {source_ref["run_id"], producer}, "render has different source or producer parents")
    if condition in ("R1", "Q1"):
        b0 = store / "trials" / row["trials"]["B0"] / "trial_selection.json"
        require(s["dataset_hash"] == family["dataset_hash"] and s["method_freeze_sha256"] == gate["method_freeze_sha256"]
                and s["b0_selection_sha256"] == sha(b0), "stress condition differs from its method or B0 selection")
        for role, path in (("input_png", image), ("cameras", cameras)):
            source_pin = s["sources"][role]
            require(run_pin(source_pin) == source_ref and source_pin["path"] == "payload/" + path.name
                    and source_pin["sha256"] == sha(path), "stress source asset disagrees with condition selection")
    require(rr.run["run_kind"] == "render" and rr.run["execution_status"] == sr.run["execution_status"] == "succeeded"
            and all(rr.run[k] == v for k, v in {"dataset_hash":family["dataset_hash"], "scene_family_id":family_id, "trial_id":render_trial,
                    "scene_id": family["scene_id"], "condition_id":condition, "split":family["split"]}.items())
            and meta["producer_run_id"] == render_ref["run_id"] and meta["input_sha256"] == sha(image) and meta["camera_sha256"] == sha(cameras),
            "condition source or camera disagrees with the selected render")
    model = mr.open("payload/model/model.json").parent
    load_package(model, expected_index_sha256=method["model_index_sha256"])
    identity = {**{k: family[k] for k in ("dataset_id", "dataset_hash", "scene_id", "scene_family_id", "scene_version", "split")},
                "camera_set_hash": sha(cameras), "condition_id": condition, "model_id": method["model_id"], "model_hash": method["model_index_sha256"]}
    experiment = "p5-development-deploy-parity" if development_parity else ("p6-frozen-model-score" if repeat is None else f"p6-timing-{repeat}-model-score")
    config = {"schema": "el-heldout-l1-score-1", "run_kind": "model_score", "trial_id": trial, "experiment_id": experiment,
              "timing_repeat": repeat, "quality_result": repeat is None and not development_parity,
              "identity": identity, "render": render_ref, "source": source_ref, "model": model_ref, "selection_sha256": sha(selected_file), **gate}
    if timing_file is not None:
        config["timing_marker_sha256"] = sha(timing_file)
    with tempfile.TemporaryDirectory(dir=store, prefix="five-method-inputs-") as tmp:
        files = ["prediction.json", *(v + ".npz" for v in ("source", *TARGETS) if v not in meta.get("failed_views", {}))]
        for name in files:
            rr.open("payload/" + name)
        pred = select_files(rr.root / "payload", Path(tmp) / "predictions", files)
        inputs = {"source": (image, source_ref["run_id"]), "cameras": (cameras, source_ref["run_id"]),
                  "predictions": (pred, render_ref["run_id"]), "model": (model, model_ref["run_id"]), "method": (method_file, None)}
        if timing_file is not None:
            inputs["timing-marker"] = (timing_file, None)
        cmd = [str(python.absolute()), "-m", "evidence_lens.l1", "--input", "{input:source}", "--cameras", "{input:cameras}",
               "--predictions", "{input:predictions}", "--model", "{input:model}", "--model-sha256", method["model_index_sha256"],
               "--output", "{out}", "--run-id", "{run_id}"]
        directory = store / "formal" / family["dataset_hash"] / "learned-scores"
        if development_parity:
            directory = directory / "development"
        pointer = (directory if repeat is None else directory / "timing" / str(repeat)) / (trial + ".json")
        return record_once(store, pointer, config, [source_ref, render_ref, model_ref], inputs, cmd, retry=retry, adopt=adopt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for stage in ("fit", "threshold", "freeze-method", "score"):
        p = commands.add_parser(stage)
        p.add_argument("--store", required=True, type=Path)
        p.add_argument("--families", type=Path, default=ROOT / "protocol/families-v1.json")
        p.add_argument("--freeze", type=Path, default=ROOT / "configs/formal-freeze-v1.json")
        if stage in ("freeze-method", "score"):
            p.add_argument("--study", type=Path, default=ROOT / "protocol/study-v1.json")
        if stage == "freeze-method":
            p.add_argument("--threshold-run", required=True)
            p.add_argument("--method-code-commit", required=True)
        else:
            p.add_argument("--retry-of")
            p.add_argument("--adopt-run")
            p.add_argument("--worker-python", required=True, type=Path)
        if stage == "threshold":
            p.add_argument("--fit-run", required=True)
        if stage == "score":
            p.add_argument("--method-freeze", type=Path)
            p.add_argument("--family", required=True)
            p.add_argument("--condition", required=True, choices=("B0", "R1", "Q1"))
            p.add_argument("--timing-repeat", type=int, choices=(1, 2, 3), help="registered CPU score repeat, excluded from quality")
    a = parser.parse_args(argv)
    store, catalog, lock = a.store.resolve(), a.families.resolve(), a.freeze.resolve()
    if a.stage in ("fit", "threshold"):
        result = record_learning(store, catalog, lock, "train" if a.stage == "fit" else "dev", a.worker_python,
                                 fit_run=getattr(a, "fit_run", None), retry=a.retry_of, adopt=a.adopt_run)
    elif a.stage == "freeze-method":
        result = freeze_method(store, catalog, lock, a.study, a.threshold_run, a.method_code_commit)
    else:
        result = record_score(store, catalog, lock, a.study, a.method_freeze or store / "protocol/method-freeze-v1.json",
                              a.family, a.condition, a.worker_python, retry=a.retry_of, adopt=a.adopt_run, repeat=a.timing_repeat)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
