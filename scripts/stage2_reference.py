"""Record one registered B reference and its independent native readback, or verify an existing pair."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from stage2_reference_catalog import (
    CATALOG,
    ENGINE,
    LOCK,
    STUDY,
    read,
    require,
    resolve_selected,
    validation,
)


def staging_source(project, folder):
    """Give the reference subprocess only the frozen native engine. It needs no other family recipe."""
    for relative in ENGINE:
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / relative, target)


def matching_config(run_path, cfg):
    actual = read(run_path / "config.resolved.yaml")
    return all(actual.get(k) == v for k, v in cfg.items() if k != "recording") and all(
        actual["recording"].get(k) == v for k, v in cfg["recording"].items()
    )


def record(
    kind,
    project,
    assets,
    store,
    engine,
    python,
    cfg,
    cfg_path,
    row,
    prior,
    checked,
    reference=None,
):
    history = prior(store, row, cfg["dataset_hash"], kind, reference)
    for item in history:
        path = store / "runs" / item["run_id"]
        if item["execution_status"] == "succeeded":
            require(
                matching_config(path, cfg),
                "Existing stage used different resolved scientific settings",
            )
            return item["run_id"]
    cmd = [
        sys.executable,
        "-m",
        "evidence_lens.recorder",
        "--root",
        str(store),
        "--source-root",
        str(engine),
        "--storage-policy",
        str(project / "configs/storage.json"),
        "--kind",
        kind,
        "--config",
        str(cfg_path),
    ]
    for label, rel in (
        ("roster", CATALOG),
        ("preregistration", STUDY),
        ("reference_freeze", LOCK),
        ("selected_recipe", row["recipe_path"]),
    ):
        cmd += ["--input", label + "=" + str(assets / rel)]
    if reference:
        parent = checked(store, reference, "scene_render", row)
        cmd += [
            "--parent",
            reference + "=" + parent["manifest_sha256"],
            "--input",
            "reference=" + str(store / "runs" / reference / "payload"),
            "--input-parent",
            "reference=" + reference,
        ]
        worker = [
            "{source}/src/evidence_lens/blender_reference/verify_reference.py",
            "--payload",
            "{input:reference}",
        ]
    else:
        worker = [
            "{source}/src/evidence_lens/blender_reference/produce.py",
            "--config",
            "{config}",
        ]
    started = [h for h in history if h.get("worker_exit_code") is not None]
    if started:
        require(
            len(started) == 1 and not started[0].get("retry_of_run_id"),
            f"Recorder retry budget exhausted for {kind}",
        )
        cmd += ["--retry-of", started[0]["run_id"]]
    env = dict(os.environ, PYTHONPATH=str(project / "src"))
    result = subprocess.run(
        [*cmd, "--", str(python), *worker, "--out", "{out}"],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)
    if result.stdout:
        print(result.stdout, file=sys.stderr, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    rid = json.loads(result.stdout.splitlines()[-1])["run_id"]
    require(
        matching_config(store / "runs" / rid, cfg),
        "Recorder resolved different scientific configuration",
    )
    return rid


def run(a):
    project = a.project.absolute()
    assets = a.assets_root.absolute()
    store = a.store.absolute()
    if str(project / "scripts") not in sys.path:
        sys.path.append(str(project / "scripts"))
    catalog, _, lock = validation(assets, project)
    if a.validate_only:
        return {
            "schema": catalog["schema"],
            "catalog_sha256": lock["catalog_sha256"],
            "families": len(catalog["families"]),
            "pilot_families": catalog["pilot_families"],
            "rendered": False,
        }
    rows = {r["scene_family_id"]: r for r in catalog["families"]}
    require(a.family in rows, "Select exactly one registered B family.")
    row = rows[a.family]
    cfg = resolve_selected(project, assets, row, catalog, lock, store)
    from formal_reference import attempts, checked, evidence

    held = store / "formal-references" / lock["catalog_sha256"]
    held.mkdir(parents=True, exist_ok=True)
    dst = held / (a.family + ".json")
    with (held / (a.family + ".lock")).open("a+") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        if dst.exists():
            saved = read(dst)
            ref = saved["reference_run_id"]
            rb = saved["readback_run_id"]
        else:
            require(
                not a.verify_only,
                "No existing reference and readback have been selected",
            )
            python = a.blender_python.absolute()
            require(
                python.is_file(), "Select the pinned Blender 4.5.16 Python environment"
            )
            with tempfile.TemporaryDirectory(
                prefix="stage2-reference-", dir=store
            ) as tmp:
                tmp = Path(tmp)
                src = tmp / "native"
                staging_source(project, src)
                q = tmp / "resolved.json"
                q.write_text(
                    json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                ref = record(
                    "scene_render",
                    project,
                    assets,
                    store,
                    src,
                    python,
                    cfg,
                    q,
                    row,
                    attempts,
                    checked,
                )
                rb = record(
                    "archive_verify",
                    project,
                    assets,
                    store,
                    src,
                    python,
                    cfg,
                    q,
                    row,
                    attempts,
                    checked,
                    ref,
                )
        require(
            all(matching_config(store / "runs" / r, cfg) for r in (ref, rb)),
            "Signed stage settings differ from the enrolled scene",
        )
        receipt = evidence(store, row, ref, rb, lock["catalog_sha256"])
        proof = read(store / "runs" / ref / "payload/reference.json")
        receipt.update(
            recipe_sha256=row["recipe_sha256"],
            geometry_fingerprint=row["geometry_fingerprint"],
            pair_structure_sha256=row["pair_structure_sha256"],
            reference_study_sha256=catalog["preregistration_sha256"],
            primary_view="v01",
            test_group=row["test_group"],
            generation_inputs={
                "input_png": {
                    "run_id": ref,
                    "manifest_sha256": receipt["reference_manifest_sha256"],
                    "path": "payload/input.png",
                    "sha256": proof["input_sha256"],
                },
                "cameras": {
                    "run_id": ref,
                    "manifest_sha256": receipt["reference_manifest_sha256"],
                    "path": "payload/cameras.json",
                    "sha256": proof["camera_sha256"],
                },
            },
        )
        if dst.exists():
            require(read(dst) == receipt, "The family reference selection changed")
        else:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=held, delete=False
            ) as tmp:
                json.dump(receipt, tmp, indent=2, sort_keys=True)
                tmp.write("\n")
            Path(tmp.name).replace(dst)
            dst.chmod(0o444)
        return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True, type=Path)
    p.add_argument("--store", required=True, type=Path)
    p.add_argument(
        "--assets-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--blender-python", required=True, type=Path)
    p.add_argument("--family")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    print(json.dumps(run(a), sort_keys=True), flush=True)
