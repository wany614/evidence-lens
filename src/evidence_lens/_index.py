"""Single-writer, disposable SQLite projection of sealed bundles and receipts."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pyarrow.parquet as pq

from ._bundle import load
from ._metrics import COUNTS, TEXT

DDL = """
CREATE TABLE IF NOT EXISTS runs (
 run_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, trial_id TEXT NOT NULL, run_kind TEXT NOT NULL,
 execution_status TEXT NOT NULL, integrity_status TEXT NOT NULL, archive_status TEXT NOT NULL,
 manifest_sha256 TEXT NOT NULL, bundle_path TEXT NOT NULL, manifest_bytes INTEGER NOT NULL,
 retry_of_run_id TEXT, config_sha256 TEXT NOT NULL, code_snapshot_sha256 TEXT NOT NULL,
 started_at_utc TEXT NOT NULL, ended_at_utc TEXT NOT NULL, exit_code INTEGER NOT NULL, failure_reason TEXT);
CREATE TABLE IF NOT EXISTS run_parents (
 run_id TEXT, parent_run_id TEXT, manifest_sha256 TEXT, PRIMARY KEY(run_id,parent_run_id));
CREATE TABLE IF NOT EXISTS artifacts (
 asset_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, format TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS run_artifacts (
 run_id TEXT, role TEXT, path TEXT, asset_id TEXT, parent_run_id TEXT,
 PRIMARY KEY(run_id,role,path));
CREATE TABLE IF NOT EXISTS scenes (
 run_id TEXT PRIMARY KEY, scene_family_id TEXT, scene_id TEXT, version TEXT, sha256 TEXT);
CREATE TABLE IF NOT EXISTS cameras (
 run_id TEXT, camera_set_id TEXT, camera_id TEXT, asset_id TEXT, PRIMARY KEY(run_id,camera_id));
CREATE TABLE IF NOT EXISTS datasets (run_id TEXT PRIMARY KEY, dataset_id TEXT, sha256 TEXT);
CREATE TABLE IF NOT EXISTS dataset_members (
 run_id TEXT, role TEXT, path TEXT, asset_id TEXT, PRIMARY KEY(run_id,role,path));
CREATE TABLE IF NOT EXISTS model_releases (run_id TEXT PRIMARY KEY, model_id TEXT, sha256 TEXT);
CREATE TABLE IF NOT EXISTS transfer_receipts (
 receipt_id TEXT PRIMARY KEY, run_id TEXT, manifest_sha256 TEXT, created_at_utc TEXT,
 status TEXT, reason TEXT, receipt_path TEXT, container_sha256 TEXT, bytes INTEGER, elapsed_seconds REAL);
CREATE TABLE IF NOT EXISTS archive_receipts (
 receipt_id TEXT PRIMARY KEY, run_id TEXT, manifest_sha256 TEXT, created_at_utc TEXT,
 status TEXT, reason TEXT, receipt_path TEXT, container_sha256 TEXT, bytes INTEGER, elapsed_seconds REAL);
"""


@contextmanager
def writer_gate(root: Path):
    directory = root / "index"
    directory.mkdir(parents=True, exist_ok=True)
    lock = sqlite3.connect(directory / "writer-lock.sqlite3", timeout=120)
    try:
        lock.execute("BEGIN IMMEDIATE")
        yield
        lock.commit()
    finally:
        lock.close()


def connect(path: Path):
    db = sqlite3.connect(path)
    db.executescript(DDL)
    columns = [f"{name} TEXT" for name in TEXT] + [f"{name} INTEGER" for name in COUNTS]
    db.execute("CREATE TABLE IF NOT EXISTS metrics (row_number INTEGER, " + ",".join(columns) +
               ", value REAL, numerator REAL, source_artifact_ids TEXT, PRIMARY KEY(run_id,row_number))")
    return db


def index_run(db, root: Path, bundle: Path, verified: dict) -> None:
    run = verified["run"]
    rid = run["run_id"]
    db.execute("INSERT OR IGNORE INTO runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (rid, run["experiment_id"], run["trial_id"], run["run_kind"], run["execution_status"],
                "sealed", "not_archived", verified["manifest_sha256"], bundle.relative_to(root).as_posix(),
                verified["bytes"], run["retry_of_run_id"], run["config_sha256"], run["code_snapshot_sha256"],
                run["started_at_utc"], run["ended_at_utc"], run["exit_code"], run["failure_reason"]))
    for parent in verified["inputs"]["parents"]:
        db.execute("INSERT OR IGNORE INTO run_parents VALUES(?,?,?)",
                   (rid, parent["run_id"], parent["manifest_sha256"]))
    catalog = [("output", row, None) for row in verified["artifacts"].values()]
    for inp in verified["inputs"]["inputs"]:
        catalog.extend(("input:" + inp["role"], row, inp["parent_run_id"]) for row in inp["files"])
    for role, row, parent in catalog:
        db.execute("INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?)",
                   (row["asset_id"], row["sha256"], row["bytes"], row["format"]))
        db.execute("INSERT OR IGNORE INTO run_artifacts VALUES(?,?,?,?,?)",
                   (rid, role, row["path"], row["asset_id"], parent))
        if run["run_kind"] == "dataset_build" and role != "output":
            db.execute("INSERT OR IGNORE INTO dataset_members VALUES(?,?,?,?)",
                       (rid, role, row["path"], row["asset_id"]))
        if role == "output" and Path(row["path"]).name == "cameras.json":
            camera_set = load(bundle / row["path"])
            for camera_id in camera_set.get("views", {}):
                db.execute("INSERT OR IGNORE INTO cameras VALUES(?,?,?,?)", (
                    rid, camera_set.get("camera_set_id", run.get("camera_set_id")),
                    camera_id, row["asset_id"]))
    if run.get("scene_id"):
        db.execute("INSERT OR IGNORE INTO scenes VALUES(?,?,?,?,?)", (
            rid, run.get("scene_family_id"), run["scene_id"], run.get("scene_version"), run.get("scene_hash")))
    if run.get("dataset_id"):
        db.execute("INSERT OR IGNORE INTO datasets VALUES(?,?,?)", (rid, run["dataset_id"], run.get("dataset_hash")))
    if run.get("model_id"):
        db.execute("INSERT OR IGNORE INTO model_releases VALUES(?,?,?)", (rid, run["model_id"], run.get("model_hash")))
    rows = pq.read_table(bundle / "metrics.parquet").to_pylist()
    keys = [*TEXT, *COUNTS, "value", "numerator", "source_artifact_ids"]
    for number, row in enumerate(rows):
        refs = row["source_artifact_ids"]
        row["source_artifact_ids"] = json.dumps(refs) if refs is not None else None
        values = [number, *(row[k] for k in keys)]
        db.execute("INSERT OR IGNORE INTO metrics VALUES(" + ",".join("?" for _ in values) + ")", values)


def index_receipt(db, root: Path, path: Path) -> None:
    receipt = load(path)
    kind = receipt["kind"]
    if kind not in {"transfer", "archive"} or path.parent.name != kind or path.stem != receipt["receipt_id"]:
        raise ValueError(f"invalid receipt identity: {path.name}")
    table = kind + "_receipts"
    db.execute(f"INSERT OR IGNORE INTO {table} VALUES(?,?,?,?,?,?,?,?,?,?)", (
        receipt["receipt_id"], receipt["run_id"], receipt["manifest_sha256"], receipt["created_at_utc"],
        receipt["status"], receipt.get("reason"), path.relative_to(root).as_posix(),
        receipt.get("container_sha256"), receipt.get("bytes"), receipt.get("elapsed_seconds")))
    if kind == "transfer" and receipt["status"] in {"accepted", "identical"}:
        db.execute("UPDATE runs SET integrity_status='verified' WHERE run_id=? AND manifest_sha256=?",
                   (receipt["run_id"], receipt["manifest_sha256"]))
    if kind == "archive" and receipt["status"] in {"uploaded_unverified", "verified"}:
        db.execute("""UPDATE runs SET archive_status=? WHERE run_id=? AND manifest_sha256=?
                      AND archive_status!='verified'""",
                   (receipt["status"], receipt["run_id"], receipt["manifest_sha256"]))
