"""Prepare four R0 jobs exclusively from the sealed main D1 checkpoint families."""

import argparse
import hashlib
import tempfile
from pathlib import Path
import zipfile

from evidence_lens._bundle import encoded, load, sha
from evidence_lens._verify import IDENTITY_FIELDS, require, verify_bundle
from make_handoff import publish, sha_bytes, zip_bytes
from stage3_reference import located
from r0_windows import SETS, validate

BATCH = "r0-dtu-checkpoints-01"


def prepare(store, output, project):
    members = {}
    jobs = []
    old_sources = None
    for obj in ("set1", "set6"):
        for cond, ids in SETS.items():
            sel = store / "dtu/intake" / ("d1-dtu-" + obj + "-256") / cond / "selected.json"
            j = load(sel)
            require(
                j["object_id"] == obj
                and j["condition"] == cond
                and j["trial_id"] == f"d1-dtu-{obj}-{cond}-256",
                "Wrong selected original trial",
            )
            parents = {
                k: j[k] for k in ("generation", "render", "source_conversion", "registration", "budget")
            }
            roots = {k: located(store, pin) for k, pin in parents.items()}
            orig = verify_bundle(roots["generation"])
            r = load(roots["generation"] / "payload/reconstruction.json")
            pred = load(roots["render"] / "payload/prediction.json")
            old = load(roots["generation"] / "config.resolved.yaml")
            require(
                orig["run"]["execution_status"] == "succeeded"
                and orig["manifest_sha256"] == parents["generation"]["manifest_sha256"]
                and r["gaussian_optimization_steps"] == 256
                and pred["checkpoint_sha256"] == r["checkpoint_sha256"] == j["checkpoint_sha256"]
                and r["generator_source_ids"] == ids
                and r["source_rgb_sha256"] == pred["source_rgb_sha256"],
                "Wrong optimization or prediction lineage",
            )
            if old_sources is None:
                old_sources = roots["generation"] / "source"
            prefix = "jobs/" + obj + "-" + cond + "/"
            roles = {}

            def add(role, name, owner, member, expected):
                path = roots[owner] / member
                require(
                    sha(roots[owner] / "manifest.json") == parents[owner]["manifest_sha256"],
                    "Incorrect parent manifest",
                )
                signed = next(x for x in load(roots[owner] / "manifest.json")["files"] if x["path"] == member)
                data = path.read_bytes()
                require(
                    sha_bytes(data) == signed["sha256"] == expected and len(data) == signed["bytes"],
                    "Original role mismatch: " + role,
                )
                roles[role] = {
                    "path": name,
                    "sha256": expected,
                    "bytes": len(data),
                    "parent": parents[owner],
                    "member": member,
                }
                members[prefix + name] = data

            add("initial", "initial.npz", "generation", "payload/initial.npz", r["initial_checkpoint_sha256"])
            add("final", "scene.npz", "generation", "payload/scene.npz", r["checkpoint_sha256"])
            add(
                "cameras",
                "cameras.json",
                "registration",
                f"payload/condition/{cond}/render_cameras.json",
                pred["render_camera_sha256"],
            )
            for v in ids:
                add(
                    v,
                    f"sources/{v}.png",
                    "source_conversion",
                    f"payload/sources/{v}.png",
                    r["source_rgb_sha256"][v],
                )
            ck = {
                "initial": {
                    "file_sha256": r["initial_checkpoint_sha256"],
                    "parameter_sha256": r["initial_parameter_sha256"],
                    "loss_by_source": r["loss_initial_by_source"],
                },
                "final": {
                    "file_sha256": r["checkpoint_sha256"],
                    "parameter_sha256": r["final_parameter_sha256"],
                    "loss_by_source": r["loss_final_by_source"],
                },
            }
            cfg = {
                "object_id": obj,
                "condition_id": cond,
                "source_ids": ids,
                "source_sha256": r["source_rgb_sha256"],
                "camera_sha256": pred["render_camera_sha256"],
                "checkpoints": ck,
                "originals": parents,
                "eps2d": old["optimizer"]["render"]["eps2d"],
                "optimization_steps": 0,
                "selection_sha256": sha(sel),
                "source_loss": "mean absolute full-grid display-sRGB",
                "contribution_measure": "derivative of image-wide direct linear channel sum with respect to each native channel value",
                "thresholds": [0, 1e-12],
            }
            job = {
                "schema": "el-r0-diagnostic-job-1",
                "experiment_id": "r0-original-checkpoint-cuda",
                "trial_id": f"r0-dtu-{obj}-{cond}",
                "config": cfg,
                "roles": roles,
                "identity": {k: orig["run"][k] for k in IDENTITY_FIELDS},
                "reference": parents["generation"],
                "readback": parents["render"],
            }
            raw = encoded(job)
            members[prefix + "job.json"] = raw
            jobs.append({"object_id": obj, "condition": cond, "sha256": sha_bytes(raw)})
    from evidence_lens._bundle import files

    worker = {
        k: v.read_bytes()
        for k, v in files(old_sources).items()
        if k not in {"scripts/dtu_render.py", "scripts/dtu_reconstruct_windows.py"}
    }
    for n in ("scripts/r0_windows.py", "scripts/r0_checkpoint_gpu.py"):
        worker[n] = (project / n).read_bytes()
    worker_hashes = {k: sha_bytes(v) for k, v in sorted(worker.items())}
    members.update({"worker/" + k: v for k, v in worker.items()})
    batch = {
        "schema": "el-r0-diagnostic-batch-1",
        "batch": BATCH,
        "jobs": jobs,
        "worker_files": worker_hashes,
        "worker_sha256": sha_bytes(zip_bytes(worker)),
    }
    raw = encoded(batch)
    members["batch.json"] = raw
    rawpack = zip_bytes(members)
    name = BATCH + ".zip"
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for k, v in members.items():
            p = tmp / k
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(v)
        require(len(list(validate(tmp, sha_bytes(raw)))) == 4, "R0 independent package readback failed")
    publish(output / "batch.json", raw)
    publish(output / name, rawpack)
    ready = {
        "schema": "el-r0-private-delivery-1",
        "batch": BATCH,
        "batch_sha256": sha_bytes(raw),
        "zip": name,
        "zip_sha256": sha_bytes(rawpack),
        "zip_bytes": len(rawpack),
        "worker_sha256": batch["worker_sha256"],
        "jobs": 4,
        "members": len(members),
        "source_photographs": 12,
        "target_photographs": 0,
    }
    publish(output / "READY.json", encoded(ready))
    return ready


if __name__ == "__main__":
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(prepare(a.store.resolve(), a.output.resolve(), Path(__file__).resolve().parents[1])))
