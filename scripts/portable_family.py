"""Pack original run bytes once per digest, with explicitly external upstream weights, or restore them."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

from evidence_lens._bundle import HASH, RUN_ID, asset_path, encoded, freeze_tree, load, local_path, sha
from evidence_lens._verify import require, verify_bundle, verify_dependencies
from evidence_lens.archive import rebuild
from formal_archive import inventory

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d"
CHECKPOINT_SIZE = 2809738232
CHECKPOINT_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
MANIFEST = "family.json"


def object_sizes(doc):
    result = {}
    for path, row in doc["files"].items():
        local_path(path)
        digest, size = row["sha256"], row["bytes"]
        require(HASH.fullmatch(digest) and type(size) is int and size >= 0 and result.get(digest, size) == size,
                "invalid family archive object record")
        result[digest] = size
    return result


def entry(name):
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100444 << 16
    return info


def pack_family(store, family, output):
    require(not output.exists(), "choose a new private container destination")
    catalog = ROOT / "protocol/families-v1.json"
    doc, origins = inventory(store, catalog, ROOT / "configs/formal-freeze-v1.json", family)
    addendum = ROOT / "protocol/analysis-addendum-v1.json"
    digest = sha(addendum)
    doc["files"]["protocol/" + addendum.name] = {"sha256": digest, "bytes": addendum.stat().st_size}
    origins[digest] = addendum
    sizes = object_sizes(doc)
    require(sizes[CHECKPOINT] == CHECKPOINT_SIZE, "missing original upstream checkpoint identity")
    doc["external"] = {CHECKPOINT: {"bytes": CHECKPOINT_SIZE, "official_download": CHECKPOINT_URL,
                       "reason": "Research dependency with separate upstream terms. Supply an independently hash-checked copy when restoring."}}
    folder = output.parent
    folder.mkdir(parents=True, exist_ok=True)
    floor = load(ROOT / "configs/storage.json")["local_free_floor_bytes"]
    require(shutil.disk_usage(folder).free - sum(n for k,n in sizes.items() if k != CHECKPOINT) - 2**20 >= floor,
            "family packing would breach the free disk floor")
    fd, name = tempfile.mkstemp(dir=folder, prefix=".family-")
    os.close(fd)
    temp = Path(name)
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_STORED) as z:
            z.writestr(entry(MANIFEST), encoded(doc))
            for digest, original in sorted(origins.items()):
                if digest == CHECKPOINT:
                    continue
                size = 0
                h = hashlib.sha256()
                with original.open("rb") as source, z.open(entry("objects/" + digest), "w", force_zip64=True) as target:
                    while block := source.read(1 << 20):
                        h.update(block)
                        size += len(block)
                        target.write(block)
                require(h.hexdigest() == digest and sizes[digest] == size, "changed family input during packing")
        os.link(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return {"schema": "el-family-transfer-1", "family": family, "name": output.name, "bytes": output.stat().st_size,
            "sha256": sha(output), "family_manifest_sha256": hashlib.sha256(encoded(doc)).hexdigest(),
            "run_manifests": doc["run_manifests"], "external": [CHECKPOINT]}


def verify_or_restore(container, expected, checkpoint, destination=None):
    start = time.monotonic()
    require(HASH.fullmatch(expected) and sha(container) == expected, "incorrect family container SHA-256")
    require(checkpoint.is_file() and checkpoint.stat().st_size == CHECKPOINT_SIZE and sha(checkpoint) == CHECKPOINT,
            "original external checkpoint bytes do not match")
    if destination is not None:
        require(not destination.exists(), "restore needs a new, empty destination")
        destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent if destination else None) as staging, zipfile.ZipFile(container) as z:
        tree = Path(staging) / "restored"
        objects = Path(staging) / "objects"
        doc = json.loads(z.read(MANIFEST))
        require(doc.get("schema") == "el-family-archive-1" and set(doc["external"]) == {CHECKPOINT}
                and doc["external"][CHECKPOINT]["bytes"] == CHECKPOINT_SIZE, "unexpected archive schema or external dependency")
        sizes = object_sizes(doc)
        required = {MANIFEST, *("objects/" + d for d in sizes if d != CHECKPOINT)}
        require(len(z.infolist()) == len(required) and set(z.namelist()) == required, "unexpected family container contents")
        if destination:
            objects.mkdir()
            require(shutil.disk_usage(destination.parent).free - (sum(sizes.values()) - CHECKPOINT_SIZE) >=
                    load(ROOT / "configs/storage.json")["local_free_floor_bytes"], "restore would exceed disk limits")
        for d, size in sizes.items():
            if d == CHECKPOINT:
                continue
            key = "objects/" + d
            require(z.getinfo(key).file_size == size, "archive member size differs")
            with z.open(key) as inp:
                if destination:
                    obj = objects / d
                    with obj.open("xb") as out:
                        shutil.copyfileobj(inp, out, 1 << 20)
                    actual = sha(obj)
                    obj.chmod(0o444)
                else:
                    actual = hashlib.file_digest(inp, "sha256").hexdigest()
            require(actual == d, "archive member hash differs from its original")
        run_prefixes = set()
        for rid, digest in doc["run_manifests"].items():
            require(RUN_ID.fullmatch(rid) and doc["files"]["runs/"+rid+"/manifest.json"]["sha256"] == digest,
                    "archive has invalid enclosing run identity")
            manifest = json.loads(z.read("objects/" + digest))
            prefix = "runs/" + rid + "/"
            run_prefixes.add(prefix)
            completion = json.loads(z.read("objects/" + doc["files"][prefix+"completion.json"]["sha256"]))
            require(completion["run_id"] == rid and completion["manifest_sha256"] == digest
                    and manifest["run_id"] == rid, "enclosing run seal mismatch")
            members = {prefix + local_path(m["path"]): {"sha256": m["sha256"], "bytes": m["bytes"]} for m in manifest["files"]}
            require(len(members) == len(manifest["files"])
                    and all(doc["files"].get(n) == m for n, m in members.items())
                    and {n for n in doc["files"] if n.startswith(prefix)} == set(members) | {prefix+"manifest.json", prefix+"completion.json"},
                    "logical run file set differs from original")
        require(all(not n.startswith("runs/") or any(n.startswith(pre) for pre in run_prefixes) for n in doc["files"]),
                "unsealed run in family container")
        index = None
        if destination:
            for logical, obj in doc["files"].items():
                target = tree / logical
                target.parent.mkdir(parents=True, exist_ok=True)
                d = obj["sha256"]
                os.link(checkpoint if d == CHECKPOINT else objects / d, target)
            for path in sorted((tree / "runs").iterdir()):
                verify_dependencies(verify_bundle(path), tree)
                freeze_tree(path)
            index = rebuild(tree)
            tree.rename(destination)
            index["path"] = "index/index.sqlite3"
        return {"family": doc["scene_family_id"], "container_sha256": expected,
                "family_manifest_sha256": hashlib.sha256(z.read(MANIFEST)).hexdigest(),
                "run_manifests": doc["run_manifests"], "logical_files": len(doc["files"]), "unique_objects": len(sizes),
                "container_bytes": container.stat().st_size, "external_bytes": CHECKPOINT_SIZE,
                "elapsed_seconds": time.monotonic()-start, "index": index}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="op",required=True)
    add = sub.add_parser("pack")
    for key in ("store","output"):
        add.add_argument("--"+key,type=Path,required=True)
    add.add_argument("--family",required=True)
    for stage in ("verify","restore"):
        cmd = sub.add_parser(stage)
        cmd.add_argument("--container",type=Path,required=True)
        cmd.add_argument("--sha256",required=True)
        cmd.add_argument("--checkpoint",type=Path,required=True)
        if stage=="restore":
            cmd.add_argument("--destination",type=Path,required=True)
    a=p.parse_args()
    if a.op=="pack":
        result=pack_family(a.store.resolve(),a.family,a.output)
    else:
        result=verify_or_restore(a.container,a.sha256,a.checkpoint,getattr(a,"destination",None))
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
