"""Explicit manifest and immutable selected-file staging for the private formal comparison."""

import hashlib
import json
import shutil
from pathlib import Path

from ._bundle import HASH, RUN_ID, load, local_path
from .scoring_io import TARGETS, file_hash, new_output, write_json

PROTOCOL_FILES = ("protocol/study-v1.json", "protocol/analysis-addendum-v1.json", "scripts/report_study.py",
                  *(f"src/evidence_lens/{name}.py" for name in
                    ("formal_evaluation", "formal_metrics", "formal_board", "formal_inputs", "formal_report",
                     "evaluation", "comparison_board", "labels", "scoring_io", "features", "l1", "risk_model", "_bundle")))
CONDITIONS = ("B0", "R1", "Q1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def protocol(root):
    sources = {name: file_hash(Path(root) / name) for name in sorted(PROTOCOL_FILES)}
    return {"schema": "el-comparison-protocol-1", "files": sources, "comparison_protocol_sha256": hash_json(sources)}


def regular(root, rel):
    rel = local_path(rel)
    root = Path(root).resolve()
    path = root / rel
    require(path.is_file() and path.resolve() == path, f"missing or linked file: {rel}")
    return path


def signature(path):
    return {"bytes": path.stat().st_size, "sha256": file_hash(path)}


def verified_file(root, rel, entry):
    path = regular(root, rel)
    require(signature(path) == {k: entry[k] for k in ("bytes", "sha256")}, f"SHA-256/size mismatch: {rel}")
    return path


class PinnedRun:
    """Authenticate the seal and each consumed file. Full-store verification is a separate recorder operation."""

    def __init__(self, store, pin):
        rid = pin.get("run_id")
        require(isinstance(rid, str) and RUN_ID.fullmatch(rid) and HASH.fullmatch(pin["manifest_sha256"]), "invalid run pin")
        self.pin = pin
        self.root = Path(store) / "runs" / rid
        manifest = regular(self.root, "manifest.json")
        seal = load(regular(self.root, "completion.json"))
        require(file_hash(manifest) == pin["manifest_sha256"] == seal["manifest_sha256"] and seal["run_id"] == rid,
                f"wrong seal for {rid}")
        data = load(manifest)
        require(data["run_id"] == rid, "manifest identifies another run")
        self.files = {local_path(row["path"]): row for row in data["files"]}
        require(len(self.files) == len(data["files"]), "duplicate file in parent manifest")
        self.run = load(self.open("run.json"))
        require(self.run["run_id"] == rid and self.run["execution_status"] == seal["execution_status"], "wrong run metadata")

    def open(self, rel):
        require(rel in self.files, f"{self.pin['run_id']}: unlisted file {rel}")
        return verified_file(self.root, rel, self.files[rel])


class Staged:
    def __init__(self, directory, name="comparison.json"):
        self.path = Path(directory)
        self.meta = load(regular(self.path, name))
        self.files = self.meta["files"]
        self.checked = set()

    def open(self, rel):
        if rel in self.checked:
            return regular(self.path, rel)
        path = verified_file(self.path, rel, self.files[rel])
        self.checked.add(rel)
        return path

    def json(self, rel):
        return load(self.open(rel))


def stage(store, selection, destination, checkout):
    selection = Path(selection).resolve()
    sel = load(selection)
    require(sel.get("schema") == "el-formal-comparison-selection-1" and sel["kind"] in ("heldout_study", "synthetic_fixture"),
            "invalid comparison selection")
    family, condition = sel["scene_family_id"], sel["condition_id"]
    require(condition in CONDITIONS, "unregistered condition")
    pins = sel["parents"]
    whole_failure = sel.get("render_failure_reason")
    roles = {"reference", "prediction", "labels", "model"}
    if not whole_failure:
        roles |= {"rules", "learned"}
    if condition == "Q1":
        roles.add("paired_prediction")
        if not sel.get("paired_render_failure_reason"):
            roles |= {"paired_rules", "paired_learned"}
    require(roles <= set(pins) <= roles | {"condition", "preparation", "readback", "generation", "derivation"}, "unexpected or missing parent role")
    intervention = next((r for r in ("preparation", "condition", "derivation") if r in pins), None)
    if sel["kind"] == "heldout_study" and condition != "B0":
        require(intervention is not None, "condition requires its sealed preparation")
    pfiles = sel["protocol_files"]
    require(set(pfiles) == {"catalog", "study", "method_freeze"}, "missing protocol pins")
    pin_paths = {key: (selection.parent / spec["path"]).resolve() for key, spec in pfiles.items()}
    for key, path in pin_paths.items():
        require(not path.is_symlink() and file_hash(path) == pfiles[key]["sha256"], f"{key} differs from selection")
    catalog, study, method = (load(pin_paths[k]) for k in ("catalog", "study", "method_freeze"))
    rows = [r for r in catalog["families"] if r["scene_family_id"] == family]
    require(len(rows) == 1 and rows[0]["split"] == "test" and rows[0]["example_view"] == "v01", "family outside fixed held-out split")
    row = rows[0]
    trial = row["trials"][condition]
    proto = protocol(checkout)
    require(method["schema"] == "el-method-freeze-1" and method["catalog_sha256"] == pfiles["catalog"]["sha256"]
            and method["study_sha256"] == pfiles["study"]["sha256"] and method["comparison_protocol_sha256"] == proto["comparison_protocol_sha256"]
            and method["threshold_select_run_id"] == pins["model"]["run_id"]
            and method["threshold_select_manifest_sha256"] == pins["model"]["manifest_sha256"], "P5 did not freeze this method and comparison")
    require(study["test_methods"] == ["constant", "1-alpha", "L0", "L1-full", "L1-geometry"], "unexpected scientific methods")
    runs = {r: PinnedRun(store, p) for r, p in pins.items()}
    partial_ids = {"source": sel["source"], "cameras": sel["cameras"]}
    with new_output(Path(destination)) as out:
        files = {}
        def include(rel, source, provenance):
            path = out / local_path(rel)
            path.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(source, path)
            files[rel] = signature(path) | provenance
        def parent_file(role, name, rel):
            run = runs[role]
            name = local_path(name)
            include(rel, run.open(name), {"parent_role": role, "parent_file": name})
        for key, path in pin_paths.items():
            include("protocol/" + key + ".json", path, {"protocol_role": key})
        for key, spec in partial_ids.items():
            parent_file(spec["parent_role"], spec["path"], "source/" + ("input.png" if key == "source" else "cameras.json"))
        if intervention is not None and condition in ("R1","Q1"):
            receipt = "source_derivation.json" if condition == "R1" else "generation.json"
            parent_file(intervention, "payload/"+receipt, "intervention.json")
            if condition == "R1":
                parent_file(intervention, "payload/reference.json","reference/condition_metadata.json")
        parent_file("model", "payload/model/model.json", "model/model.json")
        parent_file("model", "payload/model/thresholds.json", "model/thresholds.json")
        parent_file("reference", "payload/reference.json", "reference/reference.json")
        parent_file("labels", "payload/evaluation.json", "labels/evaluation.json")
        for v in TARGETS:
            parent_file("reference", f"payload/reference/{v}.npz", f"reference/{v}.npz")
            parent_file("labels", f"payload/labels/{v}.npz", f"labels/{v}.npz")
        for prefix in (["", "paired_"] if condition == "Q1" else [""]):
            failed = sel.get(prefix + "render_failure_reason")
            rr = runs[prefix + "prediction"]
            if failed:
                require(isinstance(failed, str) and failed.strip() and rr.run["execution_status"] != "succeeded",
                        "whole-render failure must pin an actual failed run")
                continue
            sources = (("prediction", "prediction.json", "sha256"), ("rules", "score.json", "score_sha256"),
                       ("learned", "l1_score.json", "sha256"))
            for name, receipt, digest_key in sources:
                role = prefix + name
                run = runs[role]
                require(run.run["execution_status"] == "succeeded", f"{role} is not sealed successful")
                parent_file(role, "payload/" + receipt, role + "/" + receipt)
                metadata = load(out / role / receipt)
                planned = ("source", *TARGETS) if condition == "Q1" and name == "prediction" else TARGETS
                for view in planned:
                    view_row = metadata.get("views", {}).get(view, {})
                    if view_row.get(digest_key):
                        parent_file(role, f"payload/{view}.npz", f"{role}/{view}.npz")
                        require(files[f"{role}/{view}.npz"]["sha256"] == view_row[digest_key], f"inconsistent {role} view")
        for role in {"prediction", "rules", "learned", "labels", "paired_prediction", "paired_rules", "paired_learned"} & set(runs):
            meta = runs[role].run
            c = "B0" if role.startswith("paired_") else condition
            require(meta.get("trial_id") == row["trials"][c] and meta.get("scene_family_id") == family
                    and meta.get("scene_id") == row["scene_id"] and meta.get("condition_id") == c
                    and meta.get("dataset_hash") == pfiles["catalog"]["sha256"], f"wrong scientific identity for {role}")
        require(runs["labels"].run["execution_status"] == "succeeded" and runs["reference"].run["execution_status"] == "succeeded"
                and runs["model"].run["run_kind"] == "threshold_select" and runs["model"].run["execution_status"] == "succeeded",
                "unaccepted labels, reference or model")
        meta = {"schema": "el-formal-comparison-input-1", "kind": sel["kind"], "scene_family_id": family,
                "scene_id": row["scene_id"], "trial_id": trial, "condition_id": condition, "dataset_id": catalog["dataset_id"],
                "dataset_hash": pfiles["catalog"]["sha256"], "study_sha256": pfiles["study"]["sha256"],
                "method_freeze_sha256": pfiles["method_freeze"]["sha256"], "comparison_protocol_sha256": proto["comparison_protocol_sha256"],
                "model_id": method["model_id"], "parents": pins, "source_selection_sha256": file_hash(selection),
                "render_failure_reason": whole_failure, "paired_render_failure_reason": sel.get("paired_render_failure_reason"),
                "files": files}
        write_json(out / "comparison.json", meta)
    return meta
