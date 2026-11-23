"""Record the four actual C models and their selected operating points before a single test opening."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from evidence_lens._bundle import HASH, encoded, load, save, sha
from evidence_lens._verify import parent_directory, require, verify_bundle
from evidence_lens.formal_inputs import PinnedRun
from evidence_lens.stage3_gate import CONDITIONS, MEMBER, verify_method_sources


def run_pin(word):
    rid, sep, digest = word.partition("=")
    require(sep and HASH.fullmatch(digest), "Provide the recorded RUN_ID=MANIFEST_SHA256")
    return {"run_id": rid, "manifest_sha256": digest}


def assemble(store, shelf, pins, run_id):
    sources = {}
    direct = {}

    def original(pin):
        rid, digest = pin["run_id"], pin["manifest_sha256"]
        require(direct.setdefault(rid, digest) == digest, "Conflicting seals for one run")
        if rid not in sources:
            path = parent_directory(store, pin, shelf)
            sources[rid] = PinnedRun(path.parent.parent, pin)
        return sources[rid]

    def spec(pin, member):
        r = original(pin)
        require(r.run["execution_status"] == "succeeded" and member in r.files, "Missing original method evidence")
        return {**pin, "member": member, "sha256": r.files[member]["sha256"]}

    def document(role):
        item = components[role]
        return load(original(item).open(item["member"]))

    components = {key: spec(pins[producer], member) for key, producer, member in (
        ("catalog", "references", "payload/catalog.json"),
        ("study", "study", "payload/protocol/study.json"),
        ("feature_schema", "study", "payload/protocol/features.json"),
        ("reconstruction_protocol", "references", "payload/optimizer.json"),
        ("production_budget", "budget", "payload/production-budget.json"),
        ("learning_protocol", "train", "inputs/learning/stage3-learning-v1.json"),
        ("training_dataset", "train", "payload/dataset.json"),
        ("development_dataset", "development", "payload/dataset.json"),
        ("lightweight_development", "light", "payload/development.json"),
        ("neural_selection", "selection", "payload/selection.json"),
        ("thresholds", "selection", "payload/thresholds.json"),
        ("tensor_comparison", "comparison", "payload/tensor-comparison.json"),
        ("cpu_validation", "cpu", "payload/cpu-validation.json"),
    )}
    selected, study, cutoffs = (document(k) for k in ("neural_selection", "study", "thresholds"))
    transfer = document("lightweight_development")["transfer"]["run"]
    primary = study["neural"]["release_seed"]
    locations = {
        "B-L1-transfer": (transfer, "payload/model/model.json", "payload/model/L1-full.json"),
        "MS-L1": (pins["light"], "payload/package/model.json", "payload/package/coefficients.json"),
    }
    for fit in document("tensor_comparison")["fits"]:
        p = {k: fit[k] for k in ("run_id", "manifest_sha256")}
        original(p)
        if fit["seed"] == primary:
            locations[fit["method"]] = (p, "payload/package/model.json", "payload/package/weights.safetensors")
    models = {}
    for name in study["methods"]:
        meta = selected["models"][name]
        parent, index, weights = locations[name]
        models[name] = {
            **{k: meta[k] for k in ("model_id", "weights_sha256", "model_index_sha256", "threshold")},
            "index": spec(parent, index), "weights": spec(parent, weights),
        }
    frozen = {
        "schema": "el-stage3-method-freeze-1", "run_id": run_id,
        "catalog_sha256": components["catalog"]["sha256"], "study_sha256": components["study"]["sha256"],
        "feature_schema_sha256": components["feature_schema"]["sha256"], "components": components,
        "models": models, "thresholds": {m: x["tau"] for m, x in cutoffs["methods"].items()},
        "default_method": selected["decision"]["default"], "primary_seed": primary,
        "scorer_scopes": ["one", "two", "all"], "conditions": CONDITIONS,
        "test_families": document("catalog")["test_families"], "targets": study["targets"],
    }
    verify_method_sources(store, shelf, frozen, direct)
    return frozen, direct


def main():
    project = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    record = sub.add_parser("record")
    names = ("references", "study", "budget", "train", "development", "light", "selection", "comparison", "cpu")
    for name in names:
        record.add_argument("--"+name, required=True, type=run_pin)
    record.add_argument("--retry-of")
    worker = sub.add_parser("work")
    worker.add_argument("--request", required=True, type=Path)
    worker.add_argument("--out", required=True, type=Path)
    for cmd in (record, worker):
        cmd.add_argument("--store", required=True, type=Path)
        cmd.add_argument("--shelf", type=Path)
    a = p.parse_args()
    store = a.store.resolve()
    shelf = (a.shelf or store.parent/"cold-shelf").resolve()
    if a.action == "work":
        require(os.environ.get("EL_RUN_KIND") == "report", "Method freeze requires its recorder")
        frozen, direct = assemble(store, shelf, load(a.request)["source_runs"], os.environ["EL_RUN_ID"])
        run_inputs = load(a.out.parent/"inputs.json")["parents"]
        require({r["run_id"]: r["manifest_sha256"] for r in run_inputs} == direct, "Recorded parents are incomplete")
        save(a.out/"method-freeze.json", frozen)
        return
    pins = {name: getattr(a, name) for name in names}
    preview, direct = assemble(store, shelf, pins, None)
    config = {"experiment_id": "stage3-method-freeze", "trial_id": "development-selection-freeze",
              "dataset_hash": preview["catalog_sha256"], "study_sha256": preview["study_sha256"],
              "source_runs": pins, "recording": {"expected_output_bytes": 600000}}
    with tempfile.TemporaryDirectory(dir=store/"work", prefix="stage3-freeze-") as tmp:
        request = Path(tmp)/"freeze-inputs.json"
        save(request, config)
        cmd = [sys.executable,"-m","evidence_lens.recorder","--root",str(store),"--source-root",str(project),
               "--storage-policy",str(project/"configs/storage.json"),"--kind","report","--config",str(request),
               "--input","authority="+str(request)]
        for rid, digest in sorted(direct.items()):
            cmd.extend(("--parent",rid+"="+digest))
        if a.retry_of:
            cmd.extend(("--retry-of", a.retry_of))
        cmd.extend(("--","{python}","{source}/scripts/freeze_stage3_method.py","work","--request","{input:authority}",
                    "--out","{out}","--store",str(store),"--shelf",str(shelf)))
        proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, text=True)
        if proc.returncode:
            raise SystemExit(proc.stdout or proc.returncode)
        outcome = json.loads(proc.stdout.strip().splitlines()[-1])
    output = Path(outcome["path"])
    seal = verify_bundle(output)
    require(seal["run"]["execution_status"] == "succeeded", "Method recording failed")
    print(json.dumps({"run_id": output.name, "manifest_sha256": seal["manifest_sha256"], "method_sha256": sha(output/MEMBER)}))


if __name__ == "__main__":
    main()
