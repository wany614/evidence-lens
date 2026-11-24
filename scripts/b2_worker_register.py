"""Seal the registered batch-two reconstruction budget and the exact historical numerical engine."""

import argparse
import os
from pathlib import Path
import shutil
from t0_worker_pack import check, digest, encoded, load, unzip_worker


def register(config, roles, out):
    study = load(roles["study"])
    seed = config["optimizer_seed"]
    base = roles["recipe"].read_bytes()
    opt = study["optimization"]
    base_seed = opt["primary_seed"]
    check(
        study["schema"] == "el-b2-reconstruction-protocol-1"
        and seed in [base_seed, *opt["fixed_M4_sensitivity_seeds_per_development_object"]]
        and base_seed == 43171
        and digest(roles["recipe"]) == study["base_recipe_sha256"]
        and digest(roles["initrules"]) == study["source_initialization_sha256"],
        "Unknown seed or numerical recipe",
    )
    source = load(roles["recipe"])
    needle = b'"seed": 43171'
    check(base.count(needle) == 1, "Unknown original recipe serialization")
    raw = base if seed == base_seed else base.replace(needle, ('"seed": ' + str(seed)).encode())
    (out / "recipe.json").write_bytes(raw)
    check(
        load(out / "recipe.json") == dict(source, seed=seed)
        and source["stop_step"] == 5000
        and source["topology"]["cap"] == 20000,
        "More than the optimizer seed changed",
    )
    tree = unzip_worker(roles["worker"], study["worker_zip_sha256"])
    dg = load(roles["depth"])
    res = load(roles["resource"])
    check(
        tree == study["worker_tree_sha256"]
        and dg["passed"]
        and res["passed"]
        and res["geometry"]["actual"] == dg,
        "Missing original CUDA/ED verification",
    )
    historical = load(roles["historical"])
    rid = historical["run_id"]
    (out / "historical").mkdir()
    for role, dest in {
        "study": "protocol.json",
        "initrules": "init-protocol.json",
        "worker": "worker.zip",
        "depth": "depth-probe.json",
        "resource": "resource-probe.json",
        "historical": "historical/" + rid + ".json",
    }.items():
        shutil.copyfile(roles[role], out / dest)
    doc = {
        "schema": "el-b2-reconstruction-origin-1",
        "run_id": os.environ["EL_RUN_ID"],
        "study_sha256": digest(roles["study"]),
        "optimizer_seed": seed,
        "recipe_sha256": digest(out / "recipe.json"),
        "base_recipe_sha256": study["base_recipe_sha256"],
        "worker_tree_sha256": tree,
        "origins": {
            "depth-probe.json": config["sources"]["depth"],
            "resource-probe.json": config["sources"]["resource"],
        },
        "prior_fits": [
            {"run_id": rid, "manifest_sha256": config["sources"]["historical"]["manifest_sha256"]}
        ],
        "target_reference_files_opened": 0,
    }
    (out / "bridge.json").write_bytes(encoded(doc))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--input", action="append", nargs=2, required=True)
    a = p.parse_args()
    register(load(a.config), {k: Path(v) for k, v in a.input}, a.out)
