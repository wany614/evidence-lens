"""Export a single source PNG, cameras and an explicitly bounded Windows worker."""

import argparse
import errno
import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

from evidence_lens._bundle import encoded, local_path, sha
from evidence_lens._index import writer_gate
from evidence_lens._verify import require, verify_bundle
from formal_lineage import EXPERIMENT, VerifiedRuns, family_record, reference_identity, run_pin

ROOT = Path(__file__).resolve().parents[1]


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def source_files(formal=False):
    listed = json.loads((ROOT / "configs/windows-worker-files.json").read_text())["files"]
    members = {}
    for name in listed:
        local_path(name)
        if formal and name == "README.md":
            continue
        path = ROOT / name
        require(path.resolve() == path and path.is_file(), f"missing worker source: {name}")
        require(name not in members, f"repeated worker source: {name}")
        members[name] = path.read_bytes()
    return members


def zip_bytes(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100444 << 16
            z.writestr(info, data)
    return buf.getvalue()


def publish(path, data):
    """A concurrent publish cannot replace an existing frozen handoff."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        require(not path.is_symlink() and path.is_file() and sha(path) == sha_bytes(data), f"immutable file conflicts: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".publish-", delete=False) as tmp:
        temp = Path(tmp.name)
        try:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.chmod(0o444)
        try:
            os.link(temp, path)
        except FileExistsError:
            require(not path.is_symlink() and sha(path) == sha_bytes(data), f"immutable file conflicts: {path}")
        except OSError as error:
            if error.errno not in {errno.ENOTSUP, errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS}:
                raise
            with path.open("xb") as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            require(sha(path) == sha_bytes(data), "immutable publication readback failed")
    finally:
        temp.unlink(missing_ok=True)


def build(bundle, out, readback=None, families=None, freeze=None):
    bundle, out = Path(bundle).resolve(), Path(out)
    require(bool(readback) == bool(families), "formal export requires both --families and --readback")
    if families:
        require(bundle.parent.name == "runs" and Path(readback).resolve().parent == bundle.parent,
                "reference and readback must belong to the same canonical Mac store")
        store = bundle.parent.parent
        verified = VerifiedRuns(store)
        checked = verified.get(bundle.name, "scene_render")
        family = family_record(families, freeze or ROOT / "configs/formal-freeze-v1.json", checked["run"]["scene_family_id"])
        ref, checked, rb = reference_identity(verified, bundle.name, Path(readback).resolve().name, family)
        source_dir = store / "formal" / family["dataset_hash"]
        pin_path = source_dir / "worker-source.zip"
        receipt = source_dir / "worker-source.json"
        with writer_gate(store):
            if receipt.is_file():
                frozen = json.loads(receipt.read_text())
                source_bytes = pin_path.read_bytes()
                require(frozen["sha256"] == sha_bytes(source_bytes), "frozen worker-source pack changed")
                with zipfile.ZipFile(io.BytesIO(source_bytes)) as prior:
                    members = {name: prior.read(local_path(name)) for name in prior.namelist()}
                require({n: sha_bytes(d) for n,d in members.items()} == frozen["files"], "frozen worker member hash differs")
            else:
                members = source_files(formal=True)
                source_bytes = zip_bytes(members)
                publish(pin_path, source_bytes)
                publish(receipt, encoded({"schema": "el-formal-worker-source-1", "sha256": sha_bytes(source_bytes),
                                          "files": {name: sha_bytes(data) for name, data in members.items()}}))
        identity = {k: family[k] for k in ("dataset_id", "dataset_hash", "split", "scene_family_id", "scene_id",
                                           "scene_version", "trial_id", "condition_id")}
        identity.update(scene_hash=ref["blend_sha256"], camera_set_hash=ref["camera_sha256"])
        job = {"schema": "el-worker-handoff-2", "experiment_id": EXPERIMENT, "trial_id": family["trial_id"],
               "condition": "B0", "protocol": "1.0", "identity": identity, "reference": run_pin(checked),
               "readback": run_pin(rb), "worker_source_sha256": sha_bytes(source_bytes)}
    else:
        checked = verify_bundle(bundle)
        run = checked["run"]
        require(run["run_kind"] == "scene_render" and run["execution_status"] == "succeeded" and
                run.get("dataset_hash") is None, "formal exports require the frozen catalog and independent readback")
        ref = json.loads((bundle / "payload/reference.json").read_text())
        require(ref["validation_passed"] is True, "reference did not pass its numerical checks")
        members = source_files()
        job = {"schema": "el-worker-handoff-1", "experiment_id": "p2-single-image", "trial_id": run["trial_id"],
               "reference": run_pin(checked), "protocol": "1.0", "condition": "B0"}
    files = {name: (bundle / "payload" / name).read_bytes() for name in ("input.png", "cameras.json")}
    require(sha_bytes(files["input.png"]) == ref["input_sha256"] and sha_bytes(files["cameras.json"]) == ref["camera_sha256"],
            "source image or cameras changed after reference verification")
    files.update({"code/" + key: value for key, value in members.items()})
    job["files"] = {name: sha_bytes(data) for name, data in files.items()}
    files["handoff.json"] = encoded(job)
    data = zip_bytes(files)
    if families:
        # The Mac later checks the returning scientific runs against this exact export.
        binding = {"schema": "el-formal-dispatch-1", "identity": identity, "reference": job["reference"],
                   "readback": job["readback"], "handoff_sha256": sha_bytes(files["handoff.json"]),
                   "container_sha256": sha_bytes(data), "worker_source_sha256": job["worker_source_sha256"]}
        with writer_gate(store):
            publish(source_dir / "dispatch" / (family["trial_id"] + ".json"), encoded(binding))
    publish(out, data)
    result = {"name": out.name, "bytes": len(data), "sha256": sha_bytes(data), "handoff_sha256": sha_bytes(files["handoff.json"]),
              "reference": job["reference"], "trial_id": job["trial_id"], "files": len(files)}
    if families:
        result.update(readback=job["readback"], worker_source_sha256=job["worker_source_sha256"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--readback", type=Path)
    parser.add_argument("--families", type=Path)
    parser.add_argument("--freeze", type=Path, default=ROOT / "configs/formal-freeze-v1.json")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.reference, args.out, args.readback, args.families, args.freeze)))
