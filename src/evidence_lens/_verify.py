"""Verify a bundle from raw bytes; check its declared upstreams separately."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path

from . import _metrics
from ._bundle import (BundleError, HASH, KINDS, LABEL, PROJECT, REQUIRED, RUN_ID, SCHEMA,
                      TERMINAL, asset_path, encoded, files, items, load, local_path, sha)

PARENT_REQUIRED = {"derivation", "render", "dataset_build", "evaluation", "features",
                   "model_fit", "threshold_select", "model_score", "report"}
IDENTITY_FIELDS = ("scene_family_id", "scene_id", "scene_version", "scene_hash", "camera_set_id",
                   "camera_set_hash", "dataset_id", "dataset_hash", "model_id", "model_hash",
                   "condition_id", "split")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def matches(pattern, value: object) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def validate_identity_fields(identity: dict) -> None:
    for key in IDENTITY_FIELDS:
        value = identity.get(key)
        require(value is None or isinstance(value, str), f"{key} must be a string or null")
        if value is not None and key.endswith("_hash"):
            require(matches(HASH, value), f"invalid {key} SHA-256")


def timestamp(value: object) -> datetime:
    require(isinstance(value, str) and value.endswith("Z"), "expected UTC timestamp")
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def file_map(record: dict) -> dict:
    result = {}
    for row in items(record, "files"):
        name = local_path(row.get("path"))
        require(name not in result and matches(HASH, row.get("sha256")) and
                type(row.get("bytes")) is int and row["bytes"] >= 0,
                f"invalid or duplicate file entry: {name}")
        result[name] = row
    return result


def matching_asset(row: dict, listed: dict) -> None:
    name = row.get("path")
    require(name in listed and all(row.get(k) == listed[name][k] for k in ("bytes", "sha256"))
            and row.get("asset_id") == "sha256:" + listed[name]["sha256"],
            f"asset disagrees with manifest: {name}")


def verify_bundle(bundle: Path) -> dict:
    actual = files(bundle)
    require({"manifest.json", "completion.json"} <= actual.keys(), "missing manifest or completion")
    manifest = load(actual["manifest.json"])
    completion = load(actual["completion.json"])
    digest = sha(actual["manifest.json"])
    require(manifest.get("schema_version") == completion.get("schema_version") == SCHEMA
            and completion.get("manifest_sha256") == digest, "invalid manifest hash or version")
    listed = file_map(manifest)
    require(set(listed) == set(actual) - {"manifest.json", "completion.json"} and
            REQUIRED <= listed.keys(), "missing, unexpected or unlisted files in bundle")
    for name, row in listed.items():
        require(name in REQUIRED or name.split("/")[0] in {"source", "inputs", "payload"},
                f"unexpected top-level file: {name}")
        require(actual[name].stat().st_size == row["bytes"] and sha(actual[name]) == row["sha256"],
                f"file size or SHA-256 mismatch: {name}")

    run = load(actual["run.json"])
    rid = run.get("run_id")
    status = run.get("execution_status")
    require(matches(RUN_ID, rid) and rid == manifest.get("run_id") == completion.get("run_id")
            and run.get("schema_version") == SCHEMA and run.get("project_id") == PROJECT
            and run.get("run_kind") in KINDS and matches(LABEL, run.get("experiment_id"))
            and matches(LABEL, run.get("trial_id")), "invalid or inconsistent run identity")
    require(status in TERMINAL and status == completion.get("execution_status")
            and run.get("producer_integrity_snapshot") == "sealed"
            and type(run.get("exit_code")) is int and run["exit_code"] == completion.get("exit_code")
            and ((run["exit_code"] == 0) == (status == "succeeded"))
            and (status == "succeeded" or bool(run.get("failure_reason"))),
            "inconsistent execution or completion state")
    for stamp in (run.get("started_at_utc"), run.get("ended_at_utc"), completion.get("sealed_at_utc")):
        timestamp(stamp)
    require(type(run.get("elapsed_seconds_monotonic")) in {int, float} and
            math.isfinite(run["elapsed_seconds_monotonic"]) and run["elapsed_seconds_monotonic"] >= 0
            and (run.get("failure_reason") is None or isinstance(run["failure_reason"], str)),
            "invalid elapsed time or failure metadata")
    validate_identity_fields(run)
    command = load(actual["command.json"])
    require(command.get("shell") is False and isinstance(command.get("argv_template"), list) and
            all(isinstance(s, str) for s in command["argv_template"]) and isinstance(command.get("cwd"), str)
            and (command.get("argv") is None or (isinstance(command["argv"], list) and
                 all(isinstance(s, str) for s in command["argv"]))), "malformed command")
    require(run.get("config_sha256") == listed["config.resolved.yaml"]["sha256"],
            "resolved configuration hash mismatch")
    snapshot = load(actual["code_snapshot.json"])
    sources = file_map(snapshot)
    require(set(sources) == {name[7:] for name in listed if name.startswith("source/")}
            and all(all(row[k] == listed["source/" + name][k] for k in ("bytes", "sha256"))
                    for name, row in sources.items())
            and snapshot.get("snapshot_sha256") == hashlib.sha256(encoded(list(sources.values()))).hexdigest()
            and snapshot["snapshot_sha256"] == run.get("code_snapshot_sha256"), "code snapshot mismatch")
    artifact_doc = load(actual["artifacts.json"])
    artifacts = file_map(artifact_doc)
    require(set(artifacts) == {name for name in listed if name.startswith("payload/")},
            "artifact catalog differs from payload")
    for row in artifacts.values():
        matching_asset(row, listed)

    dependencies = load(actual["inputs.json"])
    parents = items(dependencies, "parents")
    parent_ids = [row.get("run_id") for row in parents]
    require(parent_ids == run.get("parent_run_ids") and len(set(parent_ids)) == len(parent_ids)
            and rid not in parent_ids and all(matches(RUN_ID, p) for p in parent_ids)
            and all(matches(HASH, row.get("manifest_sha256")) for row in parents)
            and (status != "succeeded" or run["run_kind"] not in PARENT_REQUIRED or bool(parents))
            and manifest.get("parents") == parents, "invalid parent references")
    retry = dependencies.get("retry_of")
    if retry is not None:
        require(isinstance(retry, dict) and matches(RUN_ID, retry.get("run_id")) and
                retry["run_id"] != rid and matches(HASH, retry.get("manifest_sha256")),
                "invalid retry reference")
    require(run.get("retry_of_run_id") == (retry["run_id"] if retry else None)
            and manifest.get("retry_of") == retry, "retry reference mismatch")
    embedded, roles, shared = set(), set(), []
    for inp in items(dependencies, "inputs"):
        role = inp.get("role")
        require(matches(LABEL, role) and role.casefold() not in roles and
                (inp.get("parent_run_id") is None or inp["parent_run_id"] in parent_ids),
                "invalid input role or parent")
        roles.add(role.casefold())
        base = local_path(inp.get("path"))
        name = local_path(inp.get("name"))
        require("/" not in name and inp.get("kind") in {"file", "directory"}, "invalid input name or kind")
        input_files = file_map(inp)
        require(bool(input_files) and (inp["kind"] != "file" or list(input_files) == [base]),
                "inconsistent input cardinality")
        for path, row in input_files.items():
            if inp.get("storage") == "embedded":
                require(base == f"inputs/{role}/{name}" and
                        (path == base if inp["kind"] == "file" else path.startswith(base + "/"))
                        and path not in embedded, "invalid embedded input location")
                matching_asset(row, listed)
                embedded.add(path)
            else:
                require(inp.get("storage") == "shared" and inp["kind"] == "file" and
                        path == asset_path(Path(), row["sha256"]).as_posix() and
                        row.get("asset_id") == "sha256:" + row["sha256"], "invalid shared input")
                shared.append(row)
    require(embedded == {n for n in listed if n.startswith("inputs/")}
            and manifest.get("shared_assets") == shared, "input catalog mismatch")
    _metrics.verify(bundle, run, {v["sha256"] for v in listed.values()} | {r["sha256"] for r in shared})
    reference = None
    if run["run_kind"] == "scene_render" and status == "succeeded" and "payload/reference.json" in artifacts:
        meta = load(actual["payload/reference.json"])
        if meta.get("schema") == "el-reference-1":
            scene = "payload/" + local_path(meta.get("scene_path"))
            require(scene in artifacts and artifacts[scene]["sha256"] == meta.get("blend_sha256")
                    and artifacts.get("payload/cameras.json", {}).get("sha256") == meta.get("camera_sha256"),
                    "reference scene or camera asset digest mismatch")
            reference = meta
    for name in artifacts:
        if status == "succeeded" and Path(name).name == "cameras.json":
            cameras = load(actual[name])
            require(isinstance(cameras.get("views", {}), dict), "invalid camera catalog")
    return {"run": run, "manifest_sha256": digest, "manifest": manifest, "inputs": dependencies,
            "command": command, "reference": reference,
            "artifacts": artifacts, "bytes": sum(r["bytes"] for r in listed.values()),
            "shared": shared}


def verify_shared(result: dict, store: Path) -> None:
    for row in result["shared"]:
        path = asset_path(store, row["sha256"])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == row["bytes"]
                and sha(path) == row["sha256"], f"missing or changed shared input: {row['sha256']}")


def retry_inputs(inputs: list[dict]) -> list[dict]:
    result = []
    for inp in inputs:
        if inp["storage"] == "embedded" and inp["kind"] == "directory":
            # The wrapper may be a new temporary directory. All child names,
            # metadata and content hashes remain part of the comparison.
            prefix = inp["path"] + "/"
            children = [dict(row, path=row["path"].removeprefix(prefix)) for row in inp["files"]]
            result.append(dict(inp, name=None, path=None, files=children))
        else:
            result.append(inp)
    return result


def retry_matches(previous: dict, run: dict, inputs: dict, command: list[str]) -> None:
    old = previous["run"]
    require(old["execution_status"] in {"failed", "cancelled", "interrupted"}
            and old["retry_of_run_id"] is None
            and all(old[k] == run[k] for k in ("trial_id", "experiment_id", "run_kind",
                                                "config_sha256", "code_snapshot_sha256"))
            and previous["inputs"]["parents"] == inputs["parents"]
            and retry_inputs(previous["inputs"]["inputs"]) == retry_inputs(inputs["inputs"])
            and previous["command"]["argv_template"] == command,
            "retry changes frozen identity/configuration/code/inputs/command or exceeds first retry")


def verify_dependencies(result: dict, store: Path, cache: dict | None = None,
                        active: frozenset = frozenset()) -> None:
    cache = {} if cache is None else cache
    rid = result["run"]["run_id"]
    require(rid not in active, "cyclic run provenance")
    verify_shared(result, store)
    deps = result["inputs"]
    parents = {}
    for ref in [*deps["parents"], *([deps["retry_of"]] if deps.get("retry_of") else [])]:
        parent_id = ref["run_id"]
        if parent_id not in cache:
            path = store / "runs" / parent_id
            require(path.is_dir(), f"missing upstream run: {parent_id}")
            upstream = verify_bundle(path)
            require(upstream["run"]["run_id"] == parent_id, "parent path and identity differ")
            verify_dependencies(upstream, store, cache, active | {rid})
            cache[parent_id] = upstream
        upstream = cache[parent_id]
        require(upstream["manifest_sha256"] == ref["manifest_sha256"], f"changed upstream run: {parent_id}")
        if ref in deps["parents"]:
            require(upstream["run"]["execution_status"] == "succeeded", "scientific parent did not succeed")
            hashes = {asset["sha256"] for asset in upstream["artifacts"].values()}
            hashes.update(r["sha256"] for i in upstream["inputs"]["inputs"] for r in i["files"])
            parents[parent_id] = hashes
    for inp in deps["inputs"]:
        if inp["parent_run_id"]:
            require(all(row["sha256"] in parents[inp["parent_run_id"]] for row in inp["files"]),
                    f"input assets missing from declared parent: {inp['role']}")
    if deps.get("retry_of"):
        retry_matches(cache[deps["retry_of"]["run_id"]], result["run"], deps, result["command"]["argv_template"])
