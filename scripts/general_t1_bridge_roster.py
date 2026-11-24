"""Build a label-blind fixed-layout roster using only the original joint seal and deployment pins."""

import argparse
from pathlib import Path

from evidence_lens._bundle import RUN_ID
from evidence_lens_bundle.bridge_history import check_joint_closure, pinned
from evidence_lens_bundle.common import digest, encoded, json_bytes, no_replace_file, read_file, require, sha


def recorder_member(store, pin, filename):
    rid = pin["run_id"]
    require(isinstance(rid, str) and RUN_ID.fullmatch(rid), "invalid recorder run ID")
    folder = store / "runs" / rid
    manifest = read_file(folder, "manifest.json")
    require(sha(manifest) == digest(pin["manifest_sha256"], "recorder manifest"), "recorder manifest differs")
    m = json_bytes(manifest)
    c = json_bytes(read_file(folder, "completion.json"))
    require(
        m["run_id"] == c["run_id"] == rid
        and c["execution_status"] == "succeeded"
        and c["manifest_sha256"] == pin["manifest_sha256"],
        "unsuccessful or unrelated sealed member",
    )
    path = pin["member"]
    require(path.startswith("payload/") and Path(path).name == filename, "unexpected deployment member")
    rows = [f for f in m["files"] if f["path"] == path]
    require(len(rows) == 1 and rows[0]["sha256"] == pin["sha256"], "unbound deployment member")
    data = read_file(folder, path)
    require(sha(data) == pin["sha256"] and len(data) == rows[0]["bytes"], "deployed file differs")
    return {"directory": str((folder / path).parent), "sha256": sha(data)}, json_bytes(data)


def assemble(a):
    matrix_pin = {
        "run_id": a.matrix_run,
        "manifest_sha256": a.matrix_manifest_sha256,
        "member": "payload/sealed-matrix.json",
        "sha256": a.matrix_sha256,
    }
    source, matrix = recorder_member(a.store, matrix_pin, "sealed-matrix.json")
    require(
        matrix["schema"] == "el-t1-closed-matrix-1" and len(matrix["fixed_cells"]) == 12,
        "closed T1 matrix required",
    )
    closure = {
        "directory": source["directory"],
        "run_id": a.matrix_run,
        "manifest_sha256": a.matrix_manifest_sha256,
        "matrix_sha256": a.matrix_sha256,
    }
    config = {"directory": str(a.config.resolve().parent), "sha256": a.config_sha256}
    require(a.config.name == "engineering-config.json", "unexpected frozen config member")
    pinned(config, "engineering-config.json")
    geometry = {
        "M2": {"directory": str(a.geometry_m2.resolve()), "sha256": a.geometry_m2_sha256},
        "M4": {"directory": str(a.geometry_m4.resolve()), "sha256": a.geometry_m4_sha256},
    }
    for g in geometry.values():
        pinned(g, "bundle.json")
    cells = []
    for old in matrix["fixed_cells"]:
        condition = old["condition"]
        cell = {"object_id": old["object_id"], "layout": condition + "-fixed"}
        if old["status"] == "failed":
            cell["terminal_failure"] = {"reason": old["reason"], "receipt_sha256": a.matrix_sha256}
        else:
            require(old["status"] == "succeeded", "nonterminal fixed cell")
            cell["geometry_bundle"] = geometry[condition]
            for key, role, filename in (
                ("case", "case", "case.json"),
                ("readback", "readback", "reconstruction-readback.json"),
                ("historical", "scores", "scores.json"),
            ):
                cell[key], _ = recorder_member(a.store, old[role], filename)
        cells.append(cell)
    check_joint_closure(closure, cells)
    data = encoded(
        {
            "schema": "el-numerical-bridge-input-1",
            "stage": "sealed_T1",
            "configuration": config,
            "joint_closure": closure,
            "cells": cells,
        }
    )
    no_replace_file(a.out, data)
    print(sha(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("store", "config", "geometry-m2", "geometry-m4", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in (
        "matrix-run",
        "matrix-manifest-sha256",
        "matrix-sha256",
        "config-sha256",
        "geometry-m2-sha256",
        "geometry-m4-sha256",
    ):
        parser.add_argument("--" + key, required=True)
    assemble(parser.parse_args())
