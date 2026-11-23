"""Pinned family identities and verified references for the Mac formal controllers."""

from pathlib import Path

from evidence_lens._bundle import HASH, LABEL, RUN_ID, load, sha
from evidence_lens._verify import require, verify_bundle, verify_dependencies

EXPERIMENT = "p3-formal-b0"
VIEWS = ("source", *(f"v{i:02}" for i in range(1, 7)))


def family_record(catalog, freeze, family_id):
    catalog = Path(catalog)
    pin = load(freeze)
    digest = sha(catalog)
    require(pin.get("catalog_sha256") == digest, "family catalog differs from the protocol freeze")
    data = load(catalog)
    require(data.get("schema") == "el-formal-families-1" and LABEL.fullmatch(data["dataset_id"]), "invalid family catalog")
    rows = [row for row in data["families"] if row["scene_family_id"] == family_id]
    require(len(rows) == 1 and LABEL.fullmatch(family_id), "family missing or repeated in catalog")
    row = rows[0]
    require(row["trial_id"] == row["trials"]["B0"] and LABEL.fullmatch(row["trial_id"]) and
            row["split"] in {"train", "development", "test"}, "invalid frozen family assignment")
    require((type(row["cv_fold"]) is int and 0 <= row["cv_fold"] <= 3) if row["split"] == "train"
            else row["cv_fold"] is None, "invalid frozen family CV fold")
    return {key: row[key] for key in ("scene_family_id", "scene_id", "scene_version", "trial_id", "split", "cv_fold", "example_view", "camera_sha256", "recipe_sha256")} | {
        "dataset_id": data["dataset_id"], "dataset_hash": digest, "condition_id": "B0"}


class VerifiedRuns:
    """Hash each sealed package and shared object once during this invocation."""

    def __init__(self, store):
        self.store = Path(store)
        self.records = {}
        self.shared = {}

    def get(self, rid, kind=None, ancestors=()):
        require(isinstance(rid, str) and RUN_ID.fullmatch(rid) and rid not in ancestors, "invalid or cyclic run ID")
        if rid not in self.records:
            bundle = self.store / "runs" / rid
            result = verify_bundle(bundle)
            require(result["run"]["run_id"] == rid, "sealed run does not match its directory")
            deps = result["inputs"]
            for ref in [*deps["parents"], *([deps["retry_of"]] if deps.get("retry_of") else [])]:
                self.get(ref["run_id"], ancestors=(*ancestors, rid))
            unseen = []
            for obj in result["shared"]:
                digest = obj["sha256"]
                if digest in self.shared:
                    require(self.shared[digest] == obj["bytes"], "inconsistent shared asset length")
                else:
                    self.shared[digest] = obj["bytes"]
                    unseen.append(obj)
            verify_dependencies(dict(result, shared=unseen), self.store, self.records)
            self.records[rid] = result
        result = self.records[rid]
        if kind:
            require(result["run"]["run_kind"] == kind and result["run"]["execution_status"] == "succeeded",
                    f"expected successful {kind}: {rid}")
        return result


def run_pin(checked):
    return {"run_id": checked["run"]["run_id"], "manifest_sha256": checked["manifest_sha256"]}


def reference_identity(verified, reference_id, readback_id, family=None):
    ref = verified.get(reference_id, "scene_render")
    rb = verified.get(readback_id, "archive_verify")
    bundle = verified.store / "runs" / reference_id
    meta = load(bundle / "payload/reference.json")
    confirmation = load(verified.store / "runs" / readback_id / "payload/reference_readback.json")
    refhash = ref["artifacts"]["payload/reference.json"]["sha256"]
    bound = {r["sha256"] for i in rb["inputs"]["inputs"] if i["parent_run_id"] == reference_id for r in i["files"]}
    require(rb["inputs"]["parents"] == [run_pin(ref)] and rb["run"]["trial_id"] == ref["run"]["trial_id"]
            and refhash in bound and confirmation.get("schema") == "el-reference-readback-1" and
            confirmation.get("scientific_acceptance") is True and confirmation.get("integrity_checked") is True
            and confirmation.get("failed_scientific_checks") == [] and confirmation.get("reference_json_sha256") == refhash
            and set(confirmation["views"]) == set(VIEWS), "reference readback did not independently accept these seven views")
    require(meta.get("schema") == "el-reference-1" and meta.get("validation_passed") is True and set(meta["views"]) == set(VIEWS)
            and meta["input_sha256"] == ref["artifacts"]["payload/input.png"]["sha256"] and
            meta["camera_sha256"] == ref["artifacts"]["payload/cameras.json"]["sha256"] and
            HASH.fullmatch(meta["blend_sha256"]), "reference did not freeze the requested camera, source and scene")
    if family:
        require(meta["camera_sha256"] == family["camera_sha256"] and meta.get("formal", {}).get("recipe_sha256") == family["recipe_sha256"]
                and confirmation.get("verification_version") == "1.2" and confirmation.get("dataset_hash") == family["dataset_hash"],
                "camera, recipe or readback version differs from the family freeze")
        selected = load(verified.store / "formal-references" / family["dataset_hash"] / (family["scene_family_id"] + ".json"))
        require(selected.get("schema") == "el-formal-reference-selection-1" and
                all(selected.get(k) == family[k] for k in ("scene_family_id", "trial_id", "split", "dataset_hash", "cv_fold")) and
                selected["reference_run_id"] == reference_id and selected["readback_run_id"] == readback_id and
                selected["reference_manifest_sha256"] == ref["manifest_sha256"] and selected["readback_manifest_sha256"] == rb["manifest_sha256"] and
                all(selected[k] == meta[k] for k in ("camera_sha256", "input_sha256", "blend_sha256")),
                "reference/readback differs from the family's frozen reference selection")
        require(family["trial_id"] != "tabletop-v001-b0", "reuse the frozen engineering training selection for tabletop")
        for result in (ref, rb):
            run = result["run"]
            require(run["experiment_id"] == EXPERIMENT and all(run[k] == family[k] for k in
                    ("trial_id", "scene_family_id", "scene_id", "dataset_id", "dataset_hash", "split", "condition_id")),
                    "formal run differs from its frozen family assignment")
        require(all(meta[k] == family[k] for k in ("scene_id", "scene_family_id")), "reference metadata family differs")
    return meta, ref, rb


def heldout_method(verified, family, method_file, study_file, freeze):
    """Resolve the recorded P5 release before any held-out B0 evaluation."""
    import re

    method = load(method_file)
    study_hash = sha(study_file)
    require(method.get("schema") == "el-method-freeze-1" and method.get("catalog_sha256") == family["dataset_hash"] and
            method.get("study_sha256") == study_hash == load(freeze)["preregistration_sha256"] and
            HASH.fullmatch(method["comparison_protocol_sha256"]) and re.fullmatch(r"[0-9a-f]{40}", method["method_code_commit"]),
            "held-out method freeze does not pin this catalog and registered comparison")
    rid = method["threshold_select_run_id"]
    release = verified.get(rid, "threshold_select")
    require(release["manifest_sha256"] == method["threshold_select_manifest_sha256"], "threshold-selection manifest differs")
    path = verified.store / "runs" / rid
    config = load(path / "config.resolved.yaml")
    split_pin = config.get("identity", {}).get("dataset_hash") or config.get("dataset_hash") or config.get("split_manifest_sha256")
    dev, model = load(path / "payload/development.json"), load(path / "payload/model/model.json")
    require(split_pin == family["dataset_hash"] == dev["split_manifest_sha256"] == model["split_manifest_sha256"] and
            dev["schema"] == "el-development-freeze-1" and dev["status"] == "frozen" and dev["run_id"] == rid and
            model["schema"] == "el-l1-package-1" and model["status"] == "frozen" and
            method["model_id"] == model["model_id"] == model["threshold_run_id"] == rid and
            release["artifacts"]["payload/model/model.json"]["sha256"] == dev["model_index_sha256"],
            "held-out evaluation requires the matching frozen development model and thresholds")
    return {"method_freeze_sha256": sha(method_file), "threshold_select_run_id": rid,
            "threshold_select_manifest_sha256": release["manifest_sha256"], "frozen_model_id": method["model_id"]}
