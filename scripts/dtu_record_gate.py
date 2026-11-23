"""Record and reuse one D1 freeze, source opening, or reference release."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from evidence_lens._bundle import load, sha
from evidence_lens._verify import verify_bundle, require


def verified_member(store, item):
    run = store / "runs" / item["run_id"]
    record = verify_bundle(run)
    require(record["manifest_sha256"] == item["manifest_sha256"] and
            record["run"]["execution_status"] == "succeeded", "selected parent not sealed and successful")
    name = item["member"]
    require(name.startswith("payload/") and name in {f["path"] for f in record["manifest"]["files"]},
            "selected material is outside sealed payload")
    path = run / name
    require(sha(path) == item["sha256"], "pinned member differs")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["freeze", "opening", "reference"])
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    a = p.parse_args()
    store, project = a.store.resolve(), a.project.resolve()
    registry = store / "dtu-authority" / "dtu-crop256-v1"
    registry.mkdir(parents=True, exist_ok=True)
    spec = load(a.selection)
    roles, parents, argv = {}, {}, []
    def add(role, ref, option=None):
        path = verified_member(store, ref)
        roles[role] = (path, ref["run_id"])
        parents[ref["run_id"]] = ref["manifest_sha256"]
        argv.extend(["--" + (option or role), "{input:" + role + "}"])

    if a.action == "freeze":
        protocol = project / "protocol/dtu-external-v1.json"
        cfg = {"protocol_sha256": sha(protocol), "diagnostic_decision": spec["diagnostic_decision"],
               "code_sha256": {n: sha(project / n) for n in spec["scientific_code"]},
               "required_receipts": spec["required_receipts"]}
        argv.extend(["--protocol","{source}/protocol/dtu-external-v1.json"])
        for role, label in (("view-set","view_set"), ("camera-plan","camera_plan"), ("release","candidate")):
            add(role, spec[role])
            cfg[label + "_sha256"] = spec[role]["sha256"]
        for ref in spec["required_receipts"].values():
            verified_member(store, ref)
            parents[ref["run_id"]] = ref["manifest_sha256"]
    else:
        add("freeze", spec["freeze"])
        cfg = {"freeze_sha256": spec["freeze"]["sha256"]}
        if a.action == "reference":
            add("opening", spec["opening"])
            cfg["opening_sha256"] = spec["opening"]["sha256"]
            cfg["terminal_cells"] = spec["terminal_cells"]
            for cond, cell in cfg["terminal_cells"].items():
                if cell["status"] == "ready":
                    add("evidence-" + cond, cell, "evidence")
                    add("case-" + cond, cell["case_provenance"], "provenance")
                else:
                    # A sealed failed computation is evidence, never a successful scientific parent.
                    run = store / "runs" / cell["run_id"]
                    record = verify_bundle(run)
                    path = run / "run.json"
                    require(record["run"]["execution_status"] in {"failed","cancelled","interrupted"}
                            and record["manifest_sha256"] == cell["manifest_sha256"]
                            and sha(path) == cell["sha256"], "unverified failure")
                    role = "failure-" + cond
                    roles[role] = (path, None)
                    argv.extend(["--evidence","{input:"+role+"}"])

    trial = "d1-dtu-" + a.action
    cfg.update(experiment_id="d1-dtu-authority", trial_id=trial, recording={"expected_output_bytes": 1024**2})
    control = store / "work" / "d1-control"
    control.mkdir(exist_ok=True, parents=True)
    reg_name = {"freeze":"freeze.json","opening":"opening.json","reference":"reference-gate.json"}[a.action]
    canonical = registry / reg_name
    if canonical.exists():
        rid = load(canonical)["run_id"]
        record = verify_bundle(store / "runs" / rid)
        out = store / "runs" / rid / "payload" / reg_name
        earlier = load(out.parent.parent / "config.resolved.yaml")
        require(record["run"]["execution_status"] == "succeeded" and sha(out) == sha(canonical)
                and all(k == "recording" or earlier[k] == v for k,v in cfg.items()), "conflicting earlier gate")
        print(json.dumps({"reused": True, "run_id": rid, "sha256": sha(out), "manifest_sha256":record["manifest_sha256"]}))
        return
    path = control / (trial + "-request.json")
    with path.open("w") as file:
        json.dump(cfg, file, indent=2)
    cmd = [sys.executable, "-m", "evidence_lens.recorder", "--root", str(store), "--source-root",str(project),
           "--kind","archive_verify","--config",str(path)]
    for rid, digest in parents.items():
        cmd.extend(["--parent", rid+"="+digest])
    for role, (origin, parent) in roles.items():
        cmd.extend(["--input", role+"="+str(origin)])
        if parent:
            cmd.extend(["--input-parent",role+"="+parent])
    cmd.extend(["--","{python}","{source}/scripts/dtu_gate.py",a.action,"--out","{out}","--source","{source}",
                "--request","{config}","--registry",str(registry),*argv])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
