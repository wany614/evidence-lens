"""Receive an original recovery worker archive and check native checkpoint predictions before evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

from evidence_lens._bundle import HASH, RUN_ID, asset_path, encoded, load, local_path, sha
from evidence_lens._snapshot import commit_shared
from evidence_lens._verify import require, verify_bundle
from evidence_lens import _capacity
from evidence_lens._index import writer_gate
from evidence_lens.archive import receive
from evidence_lens.stage3_gaussians import PARAMS, digest_values
from recovery_inputs import (
    INIT,
    SETS,
    TARGETS,
    copy_script,
    located,
    member,
    validate_regression_freeze,
)

STEPS = (0, 500, 1500, 3000, 5000)


def signature(path):
    return {"bytes": path.stat().st_size, "sha256": sha(path)}


def restore_shared(returns, store, receipt, archived_inputs):
    """Rebuild the producer's exact shared asset set before the ordinary recorder import."""
    listed = {
        v["sha256"]: v["bytes"]
        for row in archived_inputs["inputs"]
        if row["storage"] == "shared"
        for v in row["files"]
    }
    actual = receipt.get("shared_inputs", {})
    require(set(actual) == set(listed), "Unaccounted shared inputs in the returned original")
    origin_parents = {x["run_id"]: x["manifest_sha256"] for x in archived_inputs["parents"]}
    for digest, info in actual.items():
        require(HASH.fullmatch(digest) and info["bytes"] == listed[digest], "Wrong shared asset identity")
        dest = asset_path(store, digest)
        if dest.exists():
            require(
                not dest.is_symlink() and dest.stat().st_size == listed[digest] and sha(dest) == digest,
                "Changed local content-addressed object",
            )
            continue
        if "file" in info:
            require(info["file"] == f"assets/sha256/{digest[:2]}/{digest}", "Invalid shared resource path")
            source = returns / info["file"]
            require(
                not source.is_symlink() and source.stat().st_size == info["bytes"],
                "Incomplete returned shared object",
            )
            commit_shared(source, store, digest)
        else:
            name = info["package"]
            rid = name.removesuffix(".zip")
            require(
                name == rid + ".zip" and RUN_ID.fullmatch(rid) and HASH.fullmatch(info["package_sha256"]),
                "Wrong upstream checkpoint package",
            )
            upstream = load(returns / (rid + ".receipt.json"))
            require(
                upstream["run"] == rid
                and upstream["package"] == name
                and upstream["sha256"] == info["package_sha256"]
                and (rid not in origin_parents or upstream["manifest_sha256"] == origin_parents[rid]),
                "Different original checkpoint producer",
            )
            native = local_path(info["member"])
            require(native.startswith("payload/"), "Shared return must refer to a producer payload")
            # Normal order is generation first. Use the already verified source bytes to avoid a second extracted copy.
            base = located(store, {"run_id": rid, "manifest_sha256": upstream["manifest_sha256"]})
            source = base / native
            require(
                source.is_file()
                and not source.is_symlink()
                and source.stat().st_size == listed[digest]
                and sha(source) == digest,
                "Changed original producer member",
            )
            commit_shared(source, store, digest)
    return listed


def read_return(store, returns, receipt_path, receipt_sha, archive_sha, policy):
    """Check the sender's receipt, the exact ZIP and its verified recorder lineage independently."""
    require(
        HASH.fullmatch(receipt_sha) and HASH.fullmatch(archive_sha) and sha(receipt_path) == receipt_sha,
        "Different independent return receipt",
    )
    doc = load(receipt_path)
    rid = doc["run"]
    require(
        doc["schema"] in ("el-recovery-return-1", "el-worker-return-1")
        and RUN_ID.fullmatch(rid)
        and receipt_path.name == rid + ".receipt.json"
        and doc["package"] == rid + ".zip"
        and doc["sha256"] == archive_sha
        and HASH.fullmatch(doc["manifest_sha256"]),
        "Return identity differs from its receipt",
    )
    archive = returns / doc["package"]
    require(
        archive.is_file()
        and not archive.is_symlink()
        and signature(archive) == {"sha256": archive_sha, "bytes": doc["bytes"]},
        "Wrong return archive size or digest",
    )
    with zipfile.ZipFile(archive) as z:
        uncompressed = sum(entry.file_size for entry in z.infolist())
        require(
            hashlib.sha256(z.read("manifest.json")).hexdigest() == doc["manifest_sha256"],
            "Different producer seal",
        )
        run = json.loads(z.read("run.json"))
        archived_inputs = json.loads(z.read("inputs.json"))
        config = json.loads(z.read("config.resolved.yaml"))
        require(
            run["run_id"] == rid
            and run["execution_status"] == doc["status"]
            and run["run_kind"] == doc["run_kind"]
            and run["trial_id"] == doc["trial_id"],
            "File receipt and sealed run identities disagree",
        )
    if doc["schema"] == "el-recovery-return-1":
        require(
            doc.get("cell_sha256") == config.get("cell_sha256")
            and doc.get("request_sha256") == config.get("request_sha256"),
            "Windows receipt does not match the sealed request",
        )
    pin = {"run_id": rid, "manifest_sha256": doc["manifest_sha256"]}
    dest = store / "recovery" / "returns" / rid / "transport.json"
    if dest.exists():
        prior = load(dest)
        require(
            prior["pin"] == pin
            and prior["zip_sha256"] == archive_sha
            and prior["receipt_sha256"] == receipt_sha,
            "Attempt to replace an archived return",
        )
        require(
            verify_bundle(located(store, pin))["manifest_sha256"] == pin["manifest_sha256"],
            "Previously accepted recorder original no longer verifies",
        )
        return prior
    from make_handoff import publish

    if doc["schema"] == "el-recovery-return-1":
        with writer_gate(store):
            missing = sum(
                row["bytes"]
                for digest, row in doc["shared_inputs"].items()
                if not asset_path(store, digest).exists()
            )
            if missing:
                space = _capacity.check(
                    store, missing + uncompressed + doc["bytes"], _capacity.policy(policy, Path.cwd())
                )
                if space["reason"]:
                    return {
                        "status": "deferred_capacity",
                        "pin": pin,
                        "reason": space["reason"],
                        "shared_asset_preflight": space,
                    }
            restore_shared(returns, store, doc, archived_inputs)
    accepted = receive(archive, store, archive_sha, policy, shelf=store.parent / "cold-shelf")
    if accepted["status"] == "deferred_capacity":
        return accepted
    require(
        accepted["status"] in ("accepted", "identical")
        and accepted["run_id"] == rid
        and accepted["manifest_sha256"] == pin["manifest_sha256"],
        "Return rejected: " + str(accepted.get("reason")),
    )
    checked = verify_bundle(located(store, pin))
    require(
        checked["manifest_sha256"] == pin["manifest_sha256"]
        and checked["run"]["execution_status"] == run["execution_status"],
        "Independent local receipt differs",
    )
    proof = {
        "schema": "el-recovery-received-1",
        "pin": pin,
        "status": run["execution_status"],
        "receipt_sha256": receipt_sha,
        "zip_sha256": archive_sha,
        "zip_bytes": doc["bytes"],
        "worker_receipt": doc,
        "local_transfer": accepted,
    }
    publish(dest, encoded(proof))
    return proof


def authenticate_native(path):
    """Count native render validity separately from file, dtype, and shape integrity."""
    with np.load(path, allow_pickle=False) as frame:
        require(
            {"rgb", "z", "alpha"}.issubset(frame.files),
            "Complete checkpoint native RGB/expected camera-Z/alpha required",
        )
        rgb, z, alpha = (frame[n] for n in ("rgb", "z", "alpha"))
    require(
        rgb.shape == (256, 256, 3)
        and z.shape == alpha.shape == (256, 256)
        and rgb.dtype == z.dtype == alpha.dtype == np.dtype("float32"),
        "Wrong native render grid or precision",
    )
    color = np.isfinite(rgb).all(2) & ((rgb >= 0) & (rgb <= 1)).all(2)
    opacity = np.isfinite(alpha) & (alpha >= 0) & (alpha <= 1)
    depth = np.isfinite(z) & (z > 0)
    return {
        "pixels": int(z.size),
        "finite_display_rgb": int(color.sum()),
        "valid_opacity": int(opacity.sum()),
        "positive_finite_depth": int(depth.sum()),
        "positive_opacity_pixels": int((opacity & (alpha > 0)).sum()),
        "zero_opacity_pixels": int((opacity & (alpha == 0)).sum()),
        "zero_opacity_non_nan_depth": int((opacity & (alpha == 0) & ~np.isnan(z)).sum()),
        "nonzero_opacity_invalid_depth": int((opacity & (alpha > 0) & ~depth).sum()),
    }


def native_gaussians(path, n):
    with np.load(path, allow_pickle=False) as data:
        require(set(PARAMS).issubset(data.files), "Gaussian parameter arrays are missing")
        p = {k: data[k] for k in PARAMS}
    for key, tail in zip(PARAMS, ((3,), (3,), (4,), (), (3,)), strict=True):
        a = p[key]
        require(
            a.shape == (n, *tail) and a.dtype == np.dtype("float32") and np.isfinite(a).all(),
            "Invalid Gaussian parameter array: " + key,
        )
    return digest_values(p)


def check_grid(index, cond, stop, full_curve=True):
    require(
        index["status"] in {"completed", "failed", "truncated", "not_run"}
        and index["schedule_steps"] == 5000
        and type(index["completed_steps"]) is int
        and 0 <= index["completed_steps"] <= stop,
        "Invalid schedule or endpoint",
    )
    if index["status"] == "completed":
        require(index["completed_steps"] == stop, "Completed fit has wrong stop step")
    elif index["status"] != "not_run":
        require(index.get("failure_reason") or index.get("reason"), "Incomplete fit has no recorded reason")
    expected = [s for s in STEPS if s <= index["completed_steps"]]
    keys = set(index["checkpoints"])
    planned_keys = {str(s) for s in expected}
    needed = planned_keys if full_curve else {str(stop)}
    require(
        keys <= planned_keys and (index["status"] != "completed" or needed <= keys),
        "Unexpected checkpoint or incomplete successful trajectory",
    )
    views = set(SETS[cond] + TARGETS)
    for step, point in index["checkpoints"].items():
        require(
            str(int(step)) == step
            and point["step"] == int(step)
            and type(point["gaussians"]) is int
            and point["gaussians"] > 0
            and isinstance(point["phase"], str)
            and point["phase"]
            and HASH.fullmatch(point["checkpoint_sha256"])
            and set(point["views"]) == views,
            "Checkpoint count, source/target roster or identity differs",
        )


def contained_asset(root, prefix, spec, manifest):
    path = (prefix / local_path(spec["path"])).relative_to(root).as_posix()
    require(
        path.startswith("payload/")
        and path in manifest
        and all(manifest[path][k] == spec[k] for k in ("sha256", "bytes")),
        "Render asset not in original worker manifest: " + path,
    )
    result = root / path
    require(
        not result.is_symlink() and signature(result) == {k: spec[k] for k in ("sha256", "bytes")},
        "Changed render artifact",
    )
    return result


def direct_pins(tree):
    """Collect the distinct original scientific runs used by the numerical reader."""
    if isinstance(tree, dict):
        if {"run_id", "manifest_sha256"} <= set(tree):
            yield {"run_id": tree["run_id"], "manifest_sha256": tree["manifest_sha256"]}
        for child in tree.values():
            yield from direct_pins(child)
    elif isinstance(tree, list):
        for child in tree:
            yield from direct_pins(child)


def record_audit(a):
    from make_handoff import publish
    from stage3_reference import perform

    store, project = a.store.resolve(), a.project.resolve()
    require(sha(a.request) == a.request_sha256, "Changed recovery return request")
    req = load(a.request)
    require(req["schema"] == "el-recovery-return-audit-request-1", "Wrong recovery audit request")
    obj, cond = req["object_id"], req["condition"]
    if obj == "set6":
        validate_regression_freeze(store, req.get("freeze"))
    cell = store / "recovery/inputs" / INIT / obj / cond / "selection.json"
    require(sha(cell) == req["input_selection_sha256"], "Original geometry selection changed")
    original = load(cell)
    # Point the numerical checker at the input run and its explicitly published levels of evidence.
    pins = {}
    for pin in direct_pins([original, req]):
        require(pin["run_id"] not in pins or pins[pin["run_id"]] == pin, "Conflicting upstream run seal")
        pins[pin["run_id"]] = pin
    role_paths = {"request": (a.request.resolve(), None)}
    raw_owner = req["render"]
    role_paths["outputs"] = (member(store, raw_owner, "payload/outputs.json"), raw_owner["run_id"])
    gen = req["generation"]
    role_paths["generation"] = (member(store, gen, "payload/reconstruction.json"), gen["run_id"])
    for role in ("job", "recipe", "worker", "registration", "resource_probe", "depth_probe", *(["freeze"] if obj == "set6" else [])):
        upstream = req[role]
        role_paths[role] = (member(store, upstream, upstream["member"]), upstream["run_id"])
    env = load(member(store, raw_owner, "config.resolved.yaml"))
    for ref in env["artifact_origins"].values():
        identity = {"run_id": ref["run_id"], "manifest_sha256": ref["manifest_sha256"]}
        require(
            identity["run_id"] not in pins or pins[identity["run_id"]] == identity,
            "Different original checkpoint parent",
        )
        if ref["execution_status"] == "succeeded":
            pins[identity["run_id"]] = identity
    outroot = store / "recovery/returns" / "audits" / a.request_sha256
    pointer = outroot / "selected.json"
    if pointer.is_file():
        existing = load(pointer)
        require(
            all(sha(member(store, p, p["member"])) == p["sha256"] for p in existing.values()),
            "Different saved verification",
        )
        return existing
    cfg = {
        "experiment_id": "recovery-return-native-audit-v2",
        "trial_id": "recovery-native-" + a.request_sha256[:16],
        "request_sha256": a.request_sha256,
        "scene_family_id": obj,
        "scene_id": obj,
        "condition_id": cond,
        "readback_implementation": "sealed-source-and-CPU-torch",
        "recording": {"expected_output_bytes": 262144},
    }
    scratch = store.parent / "recovery-private-scratch"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        engine = Path(tmp) / "engine"
        for name in (
            "recovery_receive.py",
            "recovery_outputs.py",
            "recovery_inputs.py",
            "stage3_reference.py",
            "make_handoff.py",
            "formal_lineage.py",
        ):
            copy_script(project, engine, name)
        # The recorder launches the separate CPU Torch interpreter against this frozen src/ directory.
        package = engine / "src/evidence_lens"
        package.mkdir(parents=True)
        for name in (
            "__init__.py", "_bundle.py", "_capacity.py", "_dedup.py", "_index.py", "_metrics.py",
            "_snapshot.py", "_verify.py", "archive.py", "stage3_gaussians.py",
        ):
            shutil.copyfile(project / "src/evidence_lens" / name, package / name)
        command = [
            str(a.audit_python.absolute()),
            "{source}/scripts/recovery_receive.py",
            "worker",
            "--request",
            "{input:request}",
            "--config",
            "{config}",
            "--store",
            str(store),
            "--outputs",
            "{input:outputs}",
            "--out",
            "{out}",
        ]
        pin = perform(store, project, engine, cfg, "archive_verify", role_paths, pins, command)
    ps = {}
    for role, product in (
        ("outputs", "payload/outputs.json"),
        ("verification", "payload/return-audit.json"),
    ):
        asset = member(store, pin, product)
        ps[role] = {**pin, "member": product, **signature(asset)}
    publish(pointer, encoded(ps))
    return ps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="action", required=True)
    q = subs.add_parser("archive")
    for name in ("store", "returns", "receipt", "policy"):
        q.add_argument("--" + name, type=Path, required=True)
    q.add_argument("--receipt-sha256", required=True)
    q.add_argument("--container-sha256", required=True)
    q = subs.add_parser("record")
    q.add_argument("--store", type=Path, default=Path(".research"))
    q.add_argument("--project", type=Path, default=Path("."))
    q.add_argument("--request", type=Path, required=True)
    q.add_argument("--request-sha256", required=True)
    q.add_argument("--audit-python", type=Path, default=Path(".venv-vision-cpu/bin/python"))
    q = subs.add_parser("worker")
    for role in ("store", "config", "request", "outputs", "out"):
        q.add_argument("--" + role, type=Path, required=True)
    a = p.parse_args()
    if a.action == "record":
        result = record_audit(a)
    elif a.action == "worker":
        cfg = load(a.config)
        req = load(a.request)
        require(
            sha(a.request) == cfg["request_sha256"]
            and sha(a.outputs) == sha(member(a.store, req["render"], "payload/outputs.json")),
            "Wrong staged audit input",
        )
        from recovery_outputs import normalize

        outputs, result = normalize(a.store, req)
        (a.out / "outputs.json").write_bytes(encoded(outputs))
        (a.out / "return-audit.json").write_bytes(encoded(result))
    else:
        result = read_return(
            a.store.resolve(), a.returns, a.receipt, a.receipt_sha256, a.container_sha256, a.policy
        )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
