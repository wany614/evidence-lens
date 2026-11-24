"""Collect a portable legacy deployment proof from explicitly linked successful recorder runs."""

import argparse
import hashlib
import json
from pathlib import Path

from evidence_lens_bundle.proof import DOCS, RIG_ROLES, RID


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check(yes, what):
    if not yes:
        raise ValueError(what)


def emit(folder, relative, raw):
    p = folder / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        f.write(raw)
    return {"path": relative, "bytes": len(raw), "sha256": sha(raw)}


def source_map(store, deployment):
    """Read only explicit manifests and named deployment/probe/camera JSON. Never enumerate runs."""
    sources = {}
    pins = {}
    seals = {}

    def load(role, origin):
        rid, expected = origin["run_id"], origin["manifest_sha256"]
        check(RID.fullmatch(rid) and len(expected) == 64, "invalid run identity")
        root = store / "runs" / rid
        seal = root / "manifest.json"
        raw = seal.read_bytes()
        manifest = json.loads(raw)
        check(sha(raw) == expected and manifest["run_id"] == rid, "original manifest SHA differs")
        members = {e["path"]: e for e in manifest["files"]}
        seals[rid] = expected
        sources["seal-" + rid] = seal
        sources["run-" + rid] = root / "run.json"
        path = DOCS[role]
        record = members[path]
        file = root / path
        data = file.read_bytes()
        check(
            sha(data) == record["sha256"] and len(data) == record["bytes"], "original proof document differs"
        )
        if "member" in origin:
            check((path, sha(data)) == (origin["member"], origin["sha256"]), "wrong original member")
        sources[role] = file
        pins[role] = {"run_id": rid, "manifest_sha256": expected, "member": path, "sha256": sha(data)}
        return json.loads(data)

    rid, ms = deployment.split("=")
    d = load("deployment", {"run_id": rid, "manifest_sha256": ms})
    if d["schema"] == "el-recovery-deployment-case-1":
        kind = "recovery"
        n = load("normalized", d["originals"]["outputs"])
        load("generator", n["original_generation"])
        load("renderer", n["original_render"])
        load("depth_probe", d["originals"]["depth_probe"])
        physical = load("original", d["original_provenance"])
    else:
        check(d["schema"] == "el-dtu-case-provenance-1", "unrecognized historical Case provenance")
        kind = "dtu"
        physical = d
    for role in sorted(RIG_ROLES):
        load(role, physical["rig_registration"])
    return {
        "schema": "el-bundle-proof-inputs-1",
        "kind": kind,
        "original_case_sha256": d["case_sha256"],
        "pins": pins,
        "seals": seals,
    }, sources


def pack(request, sources, out):
    records = {}
    seals = {}
    for role, pin in request["pins"].items():
        data = sources[role].read_bytes()
        check(sha(data) == pin["sha256"], "changed document")
        records[role] = {"origin": pin, "file": emit(out, "documents/" + role + ".json", data)}
    for rid, h in request["seals"].items():
        raw = sources["seal-" + rid].read_bytes()
        check(sha(raw) == h, "changed original seal")
        seals[rid] = {
            "manifest": emit(out, "seals/" + rid + "/manifest.json", raw),
            "run": emit(out, "seals/" + rid + "/run.json", sources["run-" + rid].read_bytes()),
        }
    obj = {
        "schema": "el-legacy-deploy-proof-1",
        "kind": request["kind"],
        "original_case_sha256": request["original_case_sha256"],
        "members": records,
        "seals": seals,
    }
    data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode()
    emit(out, "proof.json", data)
    return {"proof_sha256": sha(data), "original_case_sha256": request["original_case_sha256"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, help="Old recorder root for the local collector")
    p.add_argument("--deployment", help="Recorded deployment run_id=manifest_sha256")
    p.add_argument("--request", type=Path, help="Instead use explicit recorder-staged member inputs")
    p.add_argument("--input", action="append", default=[], metavar="ROLE=FILE")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    if a.store and a.deployment and not a.request and not a.input:
        req, files = source_map(a.store, a.deployment)
    elif a.request and a.input and not a.store and not a.deployment:
        req = json.loads(a.request.read_bytes())
        files = {k: Path(v) for k, v in (e.split("=", 1) for e in a.input)}
    else:
        p.error("choose a sealed deployment or explicit recorder-staged inputs")
    check(
        not a.out.is_symlink() and (not a.out.exists() or not any(a.out.iterdir())),
        "new or empty proof directory required",
    )
    a.out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(pack(req, files, a.out)))
