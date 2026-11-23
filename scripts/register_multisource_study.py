"""Seal the multisource study, feature schema and roster before the first fit."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register(args):
    files = {name: getattr(args, name) for name in ("catalog", "study", "features", "phase1", "phase2", "public_replay")}
    docs = {name: json.loads(path.read_bytes()) for name, path in files.items()}
    catalog, study, features = (docs[name] for name in ("catalog", "study", "features"))
    if (catalog["schema"] != "el-stage3-family-roster-1" or study["schema"] != "el-stage3-study-1"
            or features["schema"] != "multisource-v1" or digest(files["catalog"]) != study["catalog_sha256"]
            or digest(files["features"]) != study["feature_schema_sha256"]):
        raise ValueError("Study, family roster or deployable feature schema differs")
    previous = docs["phase1"]["families"] + docs["phase2"]["families"]
    current = catalog["families"]
    ids = [f["scene_family_id"] for f in current]
    older = {f["scene_family_id"] for f in previous}
    if len(older) != 48 or len(set(ids)) != len(ids) or len(ids) != 16 or older.intersection(ids):
        raise ValueError("Repeated historical or current family identity")
    splits = {name: [f["scene_family_id"] for f in current if f["split"] == name]
              for name in ("train", "development", "test")}
    if ([len(splits[k]) for k in splits] != [8, 4, 4]
            or catalog["pilot_families"] != splits["train"][:2] or catalog["test_families"] != splits["test"]
            or catalog["source_sets"] != study["sources"] or catalog["target_ids"] != study["targets"]
            or set(catalog["source_ids"]).intersection(catalog["target_ids"])):
        raise ValueError("The fixed family split, paired cameras or targets differ")
    for key in ("geometry_fingerprint", "pair_structure_sha256"):
        new = [f[key] for f in current]
        known = {f[key] for f in previous if key in f}
        if len(set(new)) != len(new) or set(new).intersection(known):
            raise ValueError("Existing geometry or paired layout repeated")
    replay = docs["public_replay"]
    if (replay["code_commit"] != study["stage_b"]["code_commit"] or replay["status"] != "verified_full"
            or replay["release_download"]["credentials"] or replay["version"] != study["stage_b"]["release"]):
        raise ValueError("A fully verified public Stage B predecessor is required")
    out = args.output / "protocol"
    out.mkdir(parents=True, exist_ok=False)
    for name in ("catalog", "study", "features"):
        shutil.copyfile(files[name], out / (name + ".json"))
    receipt = {"schema": "el-stage3-study-registration-1", "run_id": os.environ["EL_RUN_ID"],
               "files": {k: digest(v) for k, v in files.items()}, "families": splits,
               "source_sets": study["sources"], "targets": study["targets"],
               "candidate_seeds": study["neural"]["candidate_seeds"],
               "formal_fit_limit": study["neural"]["max_formal_fits"], "feature_columns": len(features["columns"])}
    (args.output / "registration.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "study", "features", "phase1", "phase2", "public_replay", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    register(parser.parse_args())
