"""Verify and receive the two registered R0 readback deliveries from a named project return."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from evidence_lens._bundle import HASH, RUN_ID, encoded, load, sha
from evidence_lens._verify import require, verify_bundle
from evidence_lens.archive import receive
from make_handoff import publish

MAIN = "r0-dtu-checkpoints-01"
CENTRES = "r0-native-centres-01"


def read_original(folder):
    t = load(folder / "READY.json")
    name = t["batch"]
    require(name in (MAIN, CENTRES), "Unexpected R0 batch")
    pack = folder / t["zip"]
    require(sha(pack) == t["zip_sha256"] and pack.stat().st_size == t["zip_bytes"], "Incorrect local dispatch")
    jobs = {}
    with zipfile.ZipFile(pack) as z:
        if name == MAIN:
            body = z.read("batch.json")
            b = json.loads(body)
            for e in b["jobs"]:
                raw = z.read("jobs/" + e["object_id"] + "-" + e["condition"] + "/job.json")
                require(hashlib.sha256(raw).hexdigest() == e["sha256"], "Different job original")
                jobs[e["sha256"]] = (json.loads(raw), b["worker_files"])
        else:
            body = z.read("job.json")
            b = json.loads(body)
            jobs[t["batch_sha256"]] = (b, b["worker_files"])
        require(hashlib.sha256(body).hexdigest() == t["batch_sha256"], "R0 dispatch identity changed")
    return t, jobs


def intake(store, dispatch_folder, returns, limits):
    delivery, jobs = read_original(dispatch_folder)
    name = delivery["batch"]
    out = store / "r0/intake" / name
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for f in sorted(returns.glob("*.receipt.json")):
        rec = load(f)
        rid = rec["run"]
        h = rec["handoff_sha256"]
        require(
            RUN_ID.fullmatch(rid)
            and HASH.fullmatch(rec["manifest_sha256"])
            and rec["schema"] == "el-worker-return-1"
            and h in jobs
            and rec["run_kind"] == "archive_verify"
            and rec["package"] == rid + ".zip"
            and f.name == rid + ".receipt.json",
            "Unexpected R0 producer receipt",
        )
        j, source = jobs[h]
        cfg = j["config"]
        require(rec["trial_id"] == j["trial_id"], "Wrong trial")
        pack = returns / rec["package"]
        if not pack.is_file() or pack.stat().st_size != rec["bytes"]:
            continue
        require(sha(pack) == rec["sha256"], "Return container SHA differs")
        with zipfile.ZipFile(pack) as z:
            run = json.loads(z.read("run.json"))
            icfg = json.loads(z.read("config.resolved.yaml"))
            source_digest = {x["path"]: x["sha256"] for x in json.loads(z.read("code_snapshot.json"))["files"]}
            require(
                run["experiment_id"] == j["experiment_id"]
                and run["trial_id"] == j["trial_id"]
                and run["run_id"] == rid
                and run["execution_status"] == rec["status"]
                and icfg["formal_handoff_sha256"] == h
                and source_digest == source
                and all(icfg.get(k) == v for k, v in cfg.items())
                and icfg["optimization_steps"] == 0,
                "Wrong scientific execution, config or frozen source",
            )
            inp = json.loads(z.read("inputs.json"))
            require(
                inp["parents"] == list(cfg["originals"].values())
                and hashlib.sha256(z.read("manifest.json")).hexdigest() == rec["manifest_sha256"],
                "Unregistered parents or manifest",
            )
            if rec["status"] == "succeeded":
                expected_roles = j.get("roles", cfg.get("roles"))
                roles = {v["role"]: v for v in inp["inputs"]}
                require(
                    set(roles) == set(expected_roles)
                    and all(
                        len(roles[n]["files"]) == 1
                        and roles[n]["parent_run_id"] == e["parent"]["run_id"]
                        and roles[n]["files"][0]["sha256"] == e["sha256"]
                        for n, e in expected_roles.items()
                    ),
                    "Unregistered numerical/photo input",
                )
        marker = out / (rid + ".json")
        if marker.exists():
            cached = load(marker)
            require(
                cached["receipt_sha256"] == sha(f) and cached["container_sha256"] == rec["sha256"],
                "Previously seen receipt changed",
            )
        else:
            x = receive(pack, store, rec["sha256"], limits, shelf=store.parent / "cold-shelf")
            if x["status"] == "deferred_capacity":
                continue
            require(
                x["status"] in ("accepted", "identical") and x["manifest_sha256"] == rec["manifest_sha256"],
                "R0 archive receive failed: " + str(x.get("reason")),
            )
            cached = {
                "schema": "el-r0-intake-1",
                "run_id": rid,
                "manifest_sha256": rec["manifest_sha256"],
                "receipt_sha256": sha(f),
                "container_sha256": rec["sha256"],
                "container_bytes": rec["bytes"],
                "batch_sha256": delivery["batch_sha256"],
                "job_sha256": h,
                "trial_id": j["trial_id"],
                "status": rec["status"],
                "archive": x,
            }
            publish(marker, encoded(cached))
        original = verify_bundle(store / "runs" / rid)
        require(original["manifest_sha256"] == rec["manifest_sha256"], "Received archive changed")
        results.setdefault(j["trial_id"], []).append(cached)
    if all(any(r["status"] == "succeeded" for r in results.get(j["trial_id"], [])) for j, _ in jobs.values()):
        require(
            all(sum(r["status"] == "succeeded" for r in rr) == 1 for rr in results.values()),
            "Multiple successful checkpoints",
        )
        manifest = {
            "schema": "el-r0-return-selection-1",
            "batch": name,
            "delivery_sha256": sha(dispatch_folder / "READY.json"),
            "jobs": {
                k: {
                    "run_id": v["run_id"],
                    "manifest_sha256": v["manifest_sha256"],
                    "job_sha256": v["job_sha256"],
                }
                for k, rr in results.items()
                for v in rr
                if v["status"] == "succeeded"
            },
            "attempts": results,
        }
        if name == MAIN:
            subs = sorted(returns.glob(name + "-summary*.json"))
            if not subs:
                return results
            summary = next(
                (
                    s
                    for s in subs
                    if {v["run"] for v in load(s)["returns"]}
                    == {v["run_id"] for rr in results.values() for v in rr}
                ),
                None,
            )
            if not summary:
                return results
            claim = load(summary)
            require(
                claim["schema"] == "el-r0-windows-returns-1"
                and claim["batch_sha256"] == delivery["batch_sha256"]
                and all(load(returns / (v["run"] + ".receipt.json")) == v for v in claim["returns"]),
                "Incorrect Windows summary",
            )
            manifest["summary_sha256"] = sha(summary)
        publish(out / "selection.json", encoded(manifest))
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("store", "dispatch", "returns", "limits"):
        p.add_argument("--" + k, required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(intake(a.store.resolve(), a.dispatch, a.returns, a.limits), sort_keys=True))
