"""Explicit manifest and immutable selected-file staging for the private formal comparison."""

import hashlib
import json
import shutil
from pathlib import Path

from ._bundle import HASH, RUN_ID, load, local_path
from .scoring_io import TARGETS, file_hash, new_output, write_json

PROTOCOL_FILES = (
    "protocol/study-v1.json",
    "protocol/analysis-addendum-v1.json",
    "protocol/operating-points-v1.json",
    "scripts/report_study.py",
    *(
        f"src/evidence_lens/{name}.py"
        for name in (
            "formal_evaluation",
            "formal_metrics",
            "formal_board",
            "formal_inputs",
            "formal_report",
            "formal_roster",
            "evaluation",
            "comparison_board",
            "labels",
            "scoring_io",
            "features",
            "l1",
            "risk_model",
            "_bundle",
        )
    ),
)
CONDITIONS = ("B0", "R1", "Q1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def protocol(root):
    sources = {name: file_hash(Path(root) / name) for name in sorted(PROTOCOL_FILES)}
    return {
        "schema": "el-comparison-protocol-1",
        "files": sources,
        "comparison_protocol_sha256": hash_json(sources),
    }


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
        require(
            isinstance(rid, str) and RUN_ID.fullmatch(rid) and HASH.fullmatch(pin["manifest_sha256"]),
            "invalid run pin",
        )
        self.pin = pin
        self.root = Path(store) / "runs" / rid
        manifest = regular(self.root, "manifest.json")
        seal = load(regular(self.root, "completion.json"))
        require(
            file_hash(manifest) == pin["manifest_sha256"] == seal["manifest_sha256"]
            and seal["run_id"] == rid,
            f"wrong seal for {rid}",
        )
        data = load(manifest)
        require(data["run_id"] == rid, "manifest identifies another run")
        self.files = {local_path(row["path"]): row for row in data["files"]}
        require(len(self.files) == len(data["files"]), "duplicate file in parent manifest")
        self.run = load(self.open("run.json"))
        require(
            self.run["run_id"] == rid and self.run["execution_status"] == seal["execution_status"],
            "wrong run metadata",
        )

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


def compact_pin(entry):
    pin = {key: entry[key] for key in ("run_id", "manifest_sha256")}
    require(
        RUN_ID.fullmatch(pin["run_id"]) and HASH.fullmatch(pin["manifest_sha256"]),
        "invalid selected run identity",
    )
    return pin


def quality_parents(data):
    require(
        data.get("schema") in ("el-p2-trial-selection-1", "el-condition-selection-1")
        and data.get("selection_status") == "selected",
        "scientific condition has no canonical selected chain",
    )
    selected = data["selected"]
    rename = {
        "reference": "reference",
        "readback": "readback",
        "render": "prediction",
        "score": "rules",
        "evaluation": "labels",
        "generation": "generation",
        "preparation": "preparation",
    }
    return {target: compact_pin(selected[k]) for k, target in rename.items() if k in selected}


def selection_from_quality(
    quality,
    learned,
    method_path,
    catalog_path,
    study_path,
    destination,
    paired_quality=None,
    paired_learned=None,
):
    inputs = {
        k: Path(path).resolve()
        for k, path in (("method_freeze", method_path), ("catalog", catalog_path), ("study", study_path))
    }
    quality = Path(quality).resolve()
    data = load(quality)

    def selected_pin(value):
        rid, sep, sha = value.partition("=")
        require(sep == "=", "supply run_id=manifest_sha256")
        return compact_pin({"run_id": rid, "manifest_sha256": sha})

    frozen = load(inputs["method_freeze"])
    parents = quality_parents(data)
    parents["learned"] = selected_pin(learned)
    parents["model"] = compact_pin(
        {
            "run_id": frozen["threshold_select_run_id"],
            "manifest_sha256": frozen["threshold_select_manifest_sha256"],
        }
    )

    def file_pin(path):
        return {"path": str(path), "sha256": file_hash(path)}

    condition = data["condition_id"]
    if condition == "Q1":
        require(paired_quality is not None and paired_learned, "Q1 requires same-family selected B0 and L1")
        paired_quality = Path(paired_quality).resolve()
        b = load(paired_quality)
        require(
            b["scene_family_id"] == data["scene_family_id"] and b["condition_id"] == "B0",
            "Q1 baseline belongs to a different family or condition",
        )
        require(
            parents.get("generation") == quality_parents(b).get("generation")
            and "generation" in parents
            and parents.get("preparation") == compact_pin(data["selected"]["producer"])
            and data["b0_selection_sha256"] == file_hash(paired_quality),
            "Q1 must use the selected B0 generation and its own selected PLY derivation",
        )
        parents.update(
            {"paired_" + k: p for k, p in quality_parents(b).items() if k in ("prediction", "rules")}
        )
        parents["paired_learned"] = selected_pin(paired_learned)
    else:
        require(paired_quality is None and paired_learned is None, "B0/R1 do not require paired score files")

    def condition_asset(key, default):
        if condition == "B0":
            return {"parent_role": "reference", "path": "payload/" + default}
        asset = data["sources"][key]
        candidates = [name for name, p in parents.items() if p == compact_pin(asset)]
        require(candidates, "condition source owner lacks a selected run pin")
        return {"parent_role": candidates[0], "path": asset["path"], "sha256": asset["sha256"]}

    result = {
        "schema": "el-formal-comparison-selection-1",
        "kind": "heldout_study",
        "scene_family_id": data["scene_family_id"],
        "condition_id": condition,
        "parents": parents,
        "source": condition_asset("input_png", "input.png"),
        "cameras": condition_asset("cameras", "cameras.json"),
        "protocol_files": {k: file_pin(p) for k, p in inputs.items()},
        "quality_selection": file_pin(quality),
    }
    if paired_quality is not None:
        result["paired_quality_selection"] = file_pin(paired_quality)
    destination = Path(destination)
    with destination.open("x", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, allow_nan=False)
        fp.write("\n")
    return result


def q1_render_inputs(runs, pins, role, producer, failed):
    rec, owner = runs[role], runs[producer]
    require(
        set(rec.run["parent_run_ids"]) == {pins["reference"]["run_id"], pins[producer]["run_id"]},
        f"{role} must descend from its selected Q1/B0 geometry and source",
    )
    if not failed:
        q1_ply_inputs(rec, owner)
        pred = load(rec.open("payload/prediction.json"))
        require(
            pred["producer_run_id"] == rec.pin["run_id"]
            and pred["generation_run_id"] == owner.pin["run_id"]
            and pred["ply_sha256"] == owner.files["payload/scene.ply"]["sha256"],
            f"{role} does not identify its actual selected PLY",
        )


def q1_ply_inputs(child, owner):
    observed = {
        f["sha256"]
        for entry in load(child.open("inputs.json"))["inputs"]
        if entry.get("parent_run_id") == owner.pin["run_id"]
        for f in entry["files"]
    }
    require(
        {owner.files["payload/" + name]["sha256"] for name in ("scene.ply", "generation.json")} <= observed,
        "recorder inputs do not bind the selected PLY and generation metadata",
    )


def verify_q1_plys(pins, runs, sel, quality_files, row):
    require(
        "paired_quality_selection" in quality_files and "preparation" in pins and "generation" in pins,
        "Q1 requires its derivation and the canonical paired B0 generation",
    )
    base_sel = load(quality_files["paired_quality_selection"])
    require(
        quality_parents(base_sel).get("generation") == pins["generation"],
        "Q1 baseline generation differs from the selected B0 generation",
    )
    if "quality_selection" in quality_files:
        q = load(quality_files["quality_selection"])
        require(
            q["b0_selection_sha256"] == file_hash(quality_files["paired_quality_selection"])
            and compact_pin(q["selected"]["producer"]) == pins["preparation"],
            "Q1 quality selection uses another B0 marker or PLY derivation",
        )
    base, derived = runs["generation"], runs["preparation"]
    for owner, cond, kind in ((base, "B0", "generation"), (derived, "Q1", "derivation")):
        r = owner.run
        require(
            r["run_kind"] == kind
            and r["execution_status"] == "succeeded"
            and r["condition_id"] == cond
            and r["scene_family_id"] == row["scene_family_id"]
            and r["scene_id"] == row["scene_id"]
            and r["trial_id"] == row["trials"][cond],
            "incorrect scientific identity for Q1 PLY parent",
        )
        owner.open("payload/scene.ply")
    require(
        derived.run["parent_run_ids"] == [base.pin["run_id"]],
        "derivation has a different B0 generation parent",
    )
    q1_ply_inputs(derived, base)
    q1_render_inputs(runs, pins, "prediction", "preparation", sel.get("render_failure_reason"))
    q1_render_inputs(runs, pins, "paired_prediction", "generation", sel.get("paired_render_failure_reason"))
    binding = {
        name: owner.files["payload/scene.ply"]["sha256"] for name, owner in (("B0", base), ("Q1", derived))
    }
    verify_q1_metadata(
        binding,
        pins,
        load(base.open("payload/generation.json")),
        load(derived.open("payload/generation.json")),
    )
    return binding


def verify_q1_metadata(binding, pins, base, derived):
    require(
        base["schema"] == "el-generation-1"
        and base["producer_run_id"] == pins["generation"]["run_id"]
        and base["ply_sha256"] == binding["B0"]
        and derived["producer_run_id"] == pins["preparation"]["run_id"]
        and derived["schema"] == "el-gaussian-derivation-1"
        and derived["condition_id"] == "Q1"
        and derived["parent_generation_run_id"] == base["producer_run_id"]
        and derived["parent_ply_sha256"] == base["ply_sha256"]
        and derived["ply_sha256"] == binding["Q1"]
        and all(derived[k] == base[k] for k in ("input_sha256", "camera_sha256", "gaussian_count")),
        "Q1 parent PLY, original generation, or derived PLY metadata differs",
    )


def stage(store, selection, destination, checkout):
    selection = Path(selection).resolve()
    sel = load(selection)
    require(
        sel.get("schema") == "el-formal-comparison-selection-1"
        and sel["kind"] in ("heldout_study", "synthetic_fixture"),
        "invalid comparison selection",
    )
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
    require(
        roles <= set(pins) <= roles | {"condition", "preparation", "readback", "generation", "derivation"},
        "unexpected or missing parent role",
    )
    intervention = next((r for r in ("preparation", "condition", "derivation") if r in pins), None)
    if sel["kind"] == "heldout_study" and condition != "B0":
        require(intervention is not None, "condition requires its sealed preparation")
    pfiles = sel["protocol_files"]
    require(set(pfiles) == {"catalog", "study", "method_freeze"}, "missing protocol pins")
    pin_paths = {key: (selection.parent / spec["path"]).resolve() for key, spec in pfiles.items()}
    for key, path in pin_paths.items():
        require(
            not path.is_symlink() and file_hash(path) == pfiles[key]["sha256"],
            f"{key} differs from selection",
        )
    catalog, study, method = (load(pin_paths[k]) for k in ("catalog", "study", "method_freeze"))
    rows = [r for r in catalog["families"] if r["scene_family_id"] == family]
    require(
        len(rows) == 1 and rows[0]["split"] == "test" and rows[0]["example_view"] == "v01",
        "family outside fixed held-out split",
    )
    row = rows[0]
    trial = row["trials"][condition]
    proto = protocol(checkout)
    require(
        method["schema"] == "el-method-freeze-1"
        and method["catalog_sha256"] == pfiles["catalog"]["sha256"]
        and method["study_sha256"] == pfiles["study"]["sha256"]
        and method["comparison_protocol_sha256"] == proto["comparison_protocol_sha256"]
        and method["threshold_select_run_id"] == pins["model"]["run_id"]
        and method["threshold_select_manifest_sha256"] == pins["model"]["manifest_sha256"],
        "P5 did not freeze this method and comparison",
    )
    require(
        study["test_methods"] == ["constant", "1-alpha", "L0", "L1-full", "L1-geometry"],
        "unexpected scientific methods",
    )
    quality_files = {}
    for name, prefix in (("quality_selection", ""), ("paired_quality_selection", "paired_")):
        if name in sel:
            qpath = (selection.parent / sel[name]["path"]).resolve()
            require(file_hash(qpath) == sel[name]["sha256"], "changed canonical quality selection")
            quality = load(qpath)
            require(
                quality["scene_family_id"] == family
                and quality["condition_id"] == ("B0" if prefix else condition),
                "quality selection has the wrong scientific identity",
            )
            for role, pin in quality_parents(quality).items():
                if not prefix or role in ("prediction", "rules"):
                    require(
                        pins.get(prefix + role) == pin,
                        "comparison parents differ from the canonical scientific selection",
                    )
            quality_files[name] = qpath
    runs = {r: PinnedRun(store, p) for r, p in pins.items()}
    q1_binding = None
    if sel["kind"] == "heldout_study":
        require(
            catalog["dataset_id"] == "evidence-lens-families-v1"
            and method["comparison_files"] == proto["files"],
            "formal run requires the registered dataset and complete frozen file map",
        )
        if condition == "Q1":
            q1_binding = verify_q1_plys(pins, runs, sel, quality_files, row)
        for prefix in ("", "paired_"):
            key = prefix + "learned"
            if key not in runs:
                continue
            checked = runs[key]
            cfg = load(checked.open("config.resolved.yaml"))
            source_pin = pins["reference"] if prefix else pins[sel["source"]["parent_role"]]
            require(
                checked.run["experiment_id"] == "p6-frozen-model-score"
                and checked.run["run_kind"] == "model_score"
                and cfg["schema"] == "el-heldout-l1-score-1"
                and cfg["quality_result"] is True
                and cfg["timing_repeat"] is None
                and cfg["render"] == pins[prefix + "prediction"]
                and cfg["model"] == pins["model"]
                and cfg["source"] == source_pin
                and cfg["method_freeze_sha256"] == pfiles["method_freeze"]["sha256"],
                "non-scientific or mismatched L1 score",
            )
            quality_key = "paired_quality_selection" if prefix else "quality_selection"
            if quality_key in sel:
                require(
                    cfg["selection_sha256"] == sel[quality_key]["sha256"],
                    "L1 job used another canonical scientific selection",
                )
    else:
        require(
            catalog["dataset_id"] != "evidence-lens-families-v1",
            "the formal catalog cannot be relabeled as a synthetic fixture",
        )

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

        for name, path in quality_files.items():
            include("protocol/" + name + ".json", path, {"selection_role": name})
        for key, path in pin_paths.items():
            include("protocol/" + key + ".json", path, {"protocol_role": key})
        for key, spec in partial_ids.items():
            rel = "source/" + ("input.png" if key == "source" else "cameras.json")
            parent_file(spec["parent_role"], spec["path"], rel)
            if "sha256" in spec:
                require(
                    files[rel]["sha256"] == spec["sha256"], "condition source asset differs from selection"
                )
        if intervention is not None and condition in ("R1", "Q1"):
            receipt = "source_derivation.json" if condition == "R1" else "generation.json"
            parent_file(intervention, "payload/" + receipt, "intervention.json")
            if condition == "R1":
                parent_file(intervention, "payload/reference.json", "reference/condition_metadata.json")
        if q1_binding is not None:
            parent_file("generation", "payload/generation.json", "q1/b0_generation.json")
        parent_file("model", "payload/model/model.json", "model/model.json")
        parent_file("model", "payload/model/thresholds.json", "model/thresholds.json")
        if sel["kind"] == "heldout_study":
            require(
                files["model/model.json"]["sha256"] == method["model_index_sha256"]
                and files["model/thresholds.json"]["sha256"] == method["thresholds_sha256"],
                "P5 model/threshold bytes differ",
            )

        parent_file("reference", "payload/reference.json", "reference/reference.json")
        if condition == "R1" and sel["kind"] == "heldout_study":
            parent_file("reference", "payload/input.png", "reference/input.png")
            parent_file("reference", "payload/cameras.json", "reference/cameras.json")
        parent_file("labels", "payload/evaluation.json", "labels/evaluation.json")
        for v in TARGETS:
            parent_file("reference", f"payload/reference/{v}.npz", f"reference/{v}.npz")
            parent_file("labels", f"payload/labels/{v}.npz", f"labels/{v}.npz")
        for prefix in ["", "paired_"] if condition == "Q1" else [""]:
            failed = sel.get(prefix + "render_failure_reason")
            rr = runs[prefix + "prediction"]
            if failed:
                require(
                    isinstance(failed, str) and failed.strip() and rr.run["execution_status"] != "succeeded",
                    "whole-render failure must pin an actual failed run",
                )
                continue
            sources = (
                ("prediction", "prediction.json", "sha256"),
                ("rules", "score.json", "score_sha256"),
                ("learned", "l1_score.json", "sha256"),
            )
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
                        require(
                            files[f"{role}/{view}.npz"]["sha256"] == view_row[digest_key],
                            f"inconsistent {role} view",
                        )
        for role in {
            "prediction",
            "rules",
            "learned",
            "labels",
            "paired_prediction",
            "paired_rules",
            "paired_learned",
        } & set(runs):
            meta = runs[role].run
            c = "B0" if role.startswith("paired_") else condition
            require(
                meta.get("trial_id") == row["trials"][c]
                and meta.get("scene_family_id") == family
                and meta.get("scene_id") == row["scene_id"]
                and meta.get("condition_id") == c
                and meta.get("dataset_hash") == pfiles["catalog"]["sha256"],
                f"wrong scientific identity for {role}",
            )
        require(
            runs["labels"].run["execution_status"] == "succeeded"
            and runs["reference"].run["execution_status"] == "succeeded"
            and runs["model"].run["run_kind"] == "threshold_select"
            and runs["model"].run["execution_status"] == "succeeded",
            "unaccepted labels, reference or model",
        )
        meta = {
            "schema": "el-formal-comparison-input-1",
            "kind": sel["kind"],
            "scene_family_id": family,
            "scene_id": row["scene_id"],
            "trial_id": trial,
            "condition_id": condition,
            "dataset_id": catalog["dataset_id"],
            "dataset_hash": pfiles["catalog"]["sha256"],
            "study_sha256": pfiles["study"]["sha256"],
            "method_freeze_sha256": pfiles["method_freeze"]["sha256"],
            "comparison_protocol_sha256": proto["comparison_protocol_sha256"],
            "model_id": method["model_id"],
            "parents": pins,
            "q1_verified_ply_sha256": q1_binding,
            "source_selection_sha256": file_hash(selection),
            "render_failure_reason": whole_failure,
            "paired_render_failure_reason": sel.get("paired_render_failure_reason"),
            "files": files,
        }
        write_json(out / "comparison.json", meta)
    return meta
