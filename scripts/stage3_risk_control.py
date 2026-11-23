"""Record and resume Stage C native dataset construction and 3A fit/development from pinned original data."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from evidence_lens._bundle import encoded, load
from evidence_lens.formal_inputs import compact_pin, hash_json, require
from evidence_lens.scoring_io import file_hash
from stage3_risk_sources import CATALOG, STUDY, LEARNING, TRANSFER, authorities, pin_selections
from vision_record import publish
from vision_sources import find_run, member

ROLES = {
    "data-pilot": ("dataset_build", "stage3-risk-pilot-data", "dataset.json", 350_000_000),
    "data-train": ("dataset_build", "stage3-risk-training-data", "dataset.json", 775_000_000),
    "data-development": ("dataset_build", "stage3-risk-development-data", "dataset.json", 1_700_000_000),
    "data-test": ("dataset_build", "stage3-risk-test-data", "dataset.json", 1_700_000_000),
    "fit": ("model_fit", "stage3-risk-light-fit", "fit.json", 4_000_000),
    "development": ("threshold_select", "stage3-risk-light-development", "development.json", 150_000_000),
    "score-test": ("model_score", "stage3-risk-test-baselines", "baselines.json", 150_000_000),
}


def chosen(roots, state, action):
    e = load(state / action / "selected.json")
    p = compact_pin(e)
    r = find_run(roots, p)
    k, experiment, receipt, _ = ROLES[action]
    require(
        r.run["run_kind"] == k
        and r.run["experiment_id"] == experiment
        and r.run["execution_status"] == "succeeded"
        and r.run["dataset_hash"] == CATALOG,
        "selected scientific stage mismatch",
    )
    path, _ = member(r, "payload/" + receipt, e["receipt_sha256"])
    return p, r, path


def run_stage(project, roots, action, opening=None, retry=None):
    store, shelf = roots
    state = store / "stage3" / CATALOG / "risk"
    folder = state / action
    folder.mkdir(parents=True, exist_ok=True)
    cat, study, learning = [
        project / "protocol" / n
        for n in ("stage3-families-v1.json", "stage3-study-v1.json", "stage3-learning-v1.json")
    ]
    kind, experiment, receipt, estimate = ROLES[action]
    parents = {}
    roles = {}
    settings = {"action": action}
    if action.startswith("data-"):
        name = action[5:]
        split = "train" if name == "pilot" else name
        cohort = "pilot" if name == "pilot" else "formal"
        _, _, rows, marker = authorities(cat, study, learning, split, cohort, roots, opening)
        selections = pin_selections(roots, rows, cohort)
        req = {
            "schema": "el-stage3-risk-source-index-1",
            "catalog_sha256": CATALOG,
            "study_sha256": STUDY,
            "learning_sha256": LEARNING,
            "split": split,
            "cohort": cohort,
            "selections": selections,
            "opening_sha256": opening,
        }
        publish(folder / "source-index.json", req)
        request = file_hash(folder / "source-index.json")
        reg = compact_pin(load(store / "stage3" / CATALOG / "registration.json")["registration"])
        parents = {p["run_id"]: p for ss in selections for p in ss["selected"].values()}
        for ss in selections:
            if ss["production_budget"]:
                p = compact_pin(ss["production_budget"])
                parents[p["run_id"]] = p
        for p in (reg, TRANSFER):
            parents[p["run_id"]] = p
        if marker:
            parents[marker["method"]["run_id"]] = marker["method"]
        roles = {
            "catalog": (cat, reg),
            "study": (study, None),
            "learning": (learning, None),
            "request": (folder / "source-index.json", None),
        }
        args = [
            "stage3_risk_data.py",
            "--store",
            str(store),
            "--shelf",
            str(shelf),
            "--catalog",
            "{input:catalog}",
            "--study",
            "{input:study}",
            "--learning",
            "{input:learning}",
            "--request",
            "{input:request}",
            "--request-sha256",
            request,
        ]
        settings.update(split=split, cohort=cohort, opening_sha256=opening)
    else:
        name = {"fit": "data-train", "development": "data-development", "score-test": "data-test"}[action]
        pin, r, dp = chosen(roots, state, name)
        parents[pin["run_id"]] = pin
        roles = {"dataset_receipt": (dp, pin)}
        args = [
            "stage3_risk_learning.py",
            {"fit": "fit", "development": "freeze", "score-test": "score-test"}[action],
            "--dataset",
            str(dp.parent),
            "--dataset-sha256",
            file_hash(dp),
        ]
        if action != "fit":
            name = "fit" if action == "development" else "development"
            pin, r, _ = chosen(roots, state, name)
            mp = r.open("payload/package/model.json")
            parents[pin["run_id"]] = pin
            roles["model_receipt"] = (mp, pin)
            args.extend(["--model", str(mp.parent), "--model-sha256", file_hash(mp)])
        if action == "score-test":
            from evidence_lens.stage3_gate import require_opening

            marker = require_opening(store, CATALOG, opening_sha256=opening, shelf=shelf)
            require(opening is not None, "explicit test opening SHA required")
            parents[marker["method"]["run_id"]] = marker["method"]
            args += ["--store", str(store), "--shelf", str(shelf), "--opening-sha256", opening]
    parent_map = {rid: p["manifest_sha256"] for rid, p in parents.items()}
    inputs = {
        k: {"sha256": file_hash(p), "parent": par["run_id"] if par else None} for k, (p, par) in roles.items()
    }
    reqhash = hash_json({"action": action, "parents": parent_map, "roles": inputs, "settings": settings})
    cfg = {
        "project_id": "evidence_lens",
        "schema": "el-stage3-risk-record-config-1",
        "experiment_id": experiment,
        "trial_id": experiment,
        "dataset_hash": CATALOG,
        "study_sha256": STUDY,
        "learning_sha256": LEARNING,
        "scientific_request_sha256": reqhash,
        "upstream": parent_map,
        "roles": inputs,
        "settings": settings,
        "recording": {"expected_output_bytes": estimate},
    }
    with (folder / "lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        publish(folder / "config.json", cfg)
        if (folder / "selected.json").exists():
            _, r, _ = chosen(roots, state, action)
            require(
                load(r.open("config.resolved.yaml"))["scientific_request_sha256"] == reqhash,
                "scientific request changed",
            )
            return load(folder / "selected.json")
        logs = sorted(folder.glob("attempt-*.log"))
        if logs:
            try:
                before = json.loads(logs[-1].read_text().strip().splitlines()[-1])
            except (ValueError, IndexError) as exc:
                raise ValueError("prior attempt has no sealed result; inspect " + str(logs[-1])) from exc
            if before["execution_status"] == "succeeded":
                return register(folder, roots, action, before, reqhash)
            require(before["run_id"] == retry, "previous failed attempt requires explicitly pinned retry")
        dest = folder / ("attempt-%02d.log" % (len(logs) + 1))
        cmd = [
            sys.executable,
            "-m",
            "evidence_lens.recorder",
            "--root",
            str(store),
            "--source-root",
            str(project),
            "--storage-policy",
            str(project / "configs/storage.json"),
            "--kind",
            kind,
            "--config",
            str(folder / "config.json"),
        ]
        for rid, sha in sorted(parent_map.items()):
            cmd += ["--parent", rid + "=" + sha]
        for role, (p, par) in roles.items():
            cmd += ["--input", role + "=" + str(p)]
            if par:
                cmd += ["--input-parent", role + "=" + par["run_id"]]
        if retry:
            cmd += ["--retry-of", retry]
        cmd += ["--", "{python}", "{source}/scripts/" + args[0], *args[1:], "--output", "{out}"]
        with dest.open("xb") as log:
            run = subprocess.run(
                cmd,
                cwd=project,
                stdout=log,
                stderr=log,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(project / "src")},
                pass_fds=(lock.fileno(),),
            )
        raw = dest.read_text().strip()
        require(run.returncode == 0, "recorded stage failed; see " + str(dest) + " " + raw[-450:])
        return register(folder, roots, action, json.loads(raw.splitlines()[-1]), reqhash)


def register(folder, roots, action, result, reqhash):
    pin = compact_pin(result)
    run = find_run(roots, pin)
    require(
        run.run["execution_status"] == "succeeded"
        and load(run.open("config.resolved.yaml"))["scientific_request_sha256"] == reqhash,
        "cannot select failed or unrelated run",
    )
    entry = ROLES[action][2]
    p = run.open("payload/" + entry)
    selected = {
        **pin,
        "receipt": "payload/" + entry,
        "receipt_sha256": file_hash(p),
        "scientific_request_sha256": reqhash,
        "experiment_id": run.run["experiment_id"],
    }
    publish(folder / "selected.json", selected)
    return selected


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=ROLES)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--shelf", type=Path)
    ap.add_argument("--opening-sha256")
    ap.add_argument("--retry-of")
    a = ap.parse_args()
    store = a.store.resolve()
    print(
        json.dumps(
            run_stage(
                a.project.resolve(),
                (store, (a.shelf or store.parent / "cold-shelf").resolve()),
                a.action,
                a.opening_sha256,
                a.retry_of,
            ),
            sort_keys=True,
        )
    )
