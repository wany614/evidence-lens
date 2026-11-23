"""Materialize one frozen formal family through the recorder, then independently read its sealed reference."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from evidence_lens._verify import verify_bundle, verify_dependencies
from evidence_lens.blender_reference.formal_catalog import CATALOG, digest, read, require, resolved, validate


ROOT = Path(__file__).resolve().parents[1]


def checked(store, run_id, kind, row, expected_sha=None):
    result = verify_bundle(store/"runs"/run_id)
    meta = result["run"]
    require(meta["run_id"] == run_id and meta["run_kind"] == kind and meta["execution_status"] == "succeeded"
            and meta["trial_id"] == row["trial_id"] and meta["scene_family_id"] == row["scene_family_id"] and
            meta["scene_id"] == row["scene_id"], f"Incorrect {kind} run identity")
    if expected_sha:
        require(result["manifest_sha256"] == expected_sha, f"{kind} checksum mismatch")
    verify_dependencies(result, store)
    return result


def evidence(store, row, reference, readback, dataset_hash):
    reused = row.get("reuse", {})
    a = checked(store, reference, "scene_render", row, reused.get("reference_manifest_sha256"))
    b = checked(store, readback, "archive_verify", row, reused.get("readback_manifest_sha256"))
    meta = read(store/"runs"/reference/"payload/reference.json")
    audit = read(store/"runs"/readback/"payload/reference_readback.json")
    require(meta["validation_passed"] is True and audit["integrity_checked"] is True and audit["scientific_acceptance"] is True,
            "Reference or independent readback failed scientific checks")
    require(audit["reference_json_sha256"] == digest(store/"runs"/reference/"payload/reference.json")
            and reference in b["run"]["parent_run_ids"] and meta["camera_sha256"] == row["camera_sha256"],
            "Readback or camera provenance differs")
    if reused:
        require(meta["blend_sha256"] == reused["blend_sha256"] and meta["input_sha256"] == reused["input_sha256"],
                "Engineering reference assets differ")
        require(digest(store/"trials"/row["trial_id"]/"trial_selection.json") == reused["selection_sha256"],
                "Engineering trial selection differs")
    else:
        require(all(m["dataset_hash"] == dataset_hash and m["split"] == row["split"] and m["condition_id"] == "B0"
                    for m in (a["run"], b["run"], meta)), "Recorded dataset, split or condition differs")
        require(audit["verification_version"] == "1.2" and audit["dataset_hash"] == dataset_hash and
                meta["formal"]["recipe_sha256"] == row["recipe_sha256"], "Readback or recipe differs from freeze")
    return {"schema":"el-formal-reference-selection-1", "scene_family_id": row["scene_family_id"],
            "trial_id":row["trial_id"], "split":row["split"], "cv_fold":row["cv_fold"],
            "dataset_hash":dataset_hash, "reused_engineering":bool(reused),
            "reference_run_id":reference, "reference_manifest_sha256":a["manifest_sha256"],
            "readback_run_id":readback, "readback_manifest_sha256":b["manifest_sha256"],
            "camera_sha256":meta["camera_sha256"], "input_sha256":meta["input_sha256"], "blend_sha256":meta["blend_sha256"]}


def attempts(store, row, catalog_hash, kind, reference=None):
    """Search only identity metadata to resume an interrupted orchestrator deterministically."""
    found = []
    for meta_path in sorted((store/"runs").glob("*/run.json")):
        m = read(meta_path)
        if (m["run_kind"] == kind and m["trial_id"] == row["trial_id"] and m["dataset_hash"] == catalog_hash
                and m["scene_family_id"] == row["scene_family_id"] and m["condition_id"] == "B0"
                and (reference is None or reference in m["parent_run_ids"])):
            found.append(m)
    return found


def record_stage(kind, store, interpreter, cfg_path, cfg, row, reference=None):
    prior = attempts(store, row, cfg["dataset_hash"], kind, reference)
    for meta in prior:
        if meta["execution_status"] == "succeeded":
            return meta["run_id"]
    cmd = [sys.executable, "-m", "evidence_lens.recorder", "--root", str(store), "--source-root", str(ROOT),
           "--kind",kind, "--config", str(cfg_path), "--experiment", cfg["experiment_id"], "--trial", row["trial_id"]]
    if prior:
        # A preflight rejection did not consume a scientific attempt. The recorder enforces the single
        # worker retry budget using its recorded events and claims.
        measured = [m for m in prior if m.get("worker_exit_code") is not None]
        if measured:
            require(len(measured) == 1 and not measured[0].get("retry_of_run_id"), f"Scientific retry limit reached for {kind}")
            cmd.extend(["--retry-of", measured[0]["run_id"]])
    if reference is None:
        worker = ["{source}/src/evidence_lens/blender_reference/produce.py", "--config", "{config}"]
    else:
        parent = checked(store, reference, "scene_render", row)
        cmd.extend(["--parent", reference+"="+parent["manifest_sha256"], "--input",
                    f"reference={store/'runs'/reference/'payload'}", "--input-parent", "reference="+reference])
        worker = ["{source}/src/evidence_lens/blender_reference/verify_reference.py", "--payload", "{input:reference}"]
    result = subprocess.run([*cmd, "--", str(interpreter), *worker, "--out", "{out}"],
                            cwd=ROOT, text=True, stdout=subprocess.PIPE)
    print(result.stdout.strip(), flush=True, file=sys.stderr)
    if result.returncode:
        raise SystemExit(result.returncode)
    return json.loads(result.stdout.splitlines()[-1])["run_id"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, default=ROOT/".research")
    p.add_argument("--family")
    p.add_argument("--validate-only", action="store_true", help="Check frozen inputs and all 24 authored identities without rendering")
    p.add_argument("--blender-python", type=Path, default=ROOT/".venv-reference/bin/python")
    args = p.parse_args()
    catalog, designs, base, lock = validate(ROOT)
    rows = {r["scene_family_id"]:r for r in catalog["families"]}
    if args.family and args.family not in rows:
        p.error("Family is not in the frozen catalog")
    if args.validate_only:
        print(json.dumps({"catalog": CATALOG, "catalog_sha256":lock["catalog_sha256"], "family_count":len(rows),
                          "representative_families":catalog["representative_families"],
                          "selected_family":args.family, "rendered":False}))
        return
    if not args.family:
        p.error("--family is required to generate or reuse a reference")
    row = rows[args.family]
    store = args.store.resolve()
    receipts = store/"formal-references"/lock["catalog_sha256"]
    receipts.mkdir(parents=True,exist_ok=True)
    path = receipts/(args.family+".json")
    with (receipts/(args.family+".lock")).open("a+") as gate:
        fcntl.flock(gate, fcntl.LOCK_EX)
        if path.exists():
            saved = read(path)
            receipt = evidence(store,row,saved["reference_run_id"],saved["readback_run_id"],lock["catalog_sha256"])
            require(receipt == saved, "Existing formal reference receipt differs")
        else:
            if "reuse" in row:
                reference, readback = row["reuse"]["reference_run_id"],row["reuse"]["readback_run_id"]
            else:
                cfg = resolved(row,catalog,designs,base,lock)
                # Keep the virtual-environment symlink for Python; resolving the symlink would select its base runtime.
                interpreter = args.blender_python.absolute()
                require(interpreter.is_file(), "Pinned reference Python executable unavailable")
                with tempfile.TemporaryDirectory(dir=store,prefix="formal-reference-config-") as tmp:
                    config_path = Path(tmp)/"reference.json"
                    config_path.write_text(json.dumps(cfg,indent=2,sort_keys=True)+"\n",encoding="utf-8")
                    reference = record_stage("scene_render",store,interpreter,config_path,cfg,row)
                    readback = record_stage("archive_verify",store,interpreter,config_path,cfg,row,reference)
            receipt = evidence(store,row,reference,readback,lock["catalog_sha256"])
            with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=receipts,delete=False) as tmp:
                json.dump(receipt,tmp,indent=2,sort_keys=True)
                tmp.write("\n")
            Path(tmp.name).replace(path)
            path.chmod(0o444)
    print(json.dumps(receipt,sort_keys=True),flush=True)


if __name__ == "__main__":
    main()
