"""Transfer only the registered training or development dataset members between research machines."""
import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path


DIGEST = re.compile(r"[0-9a-f]{64}")
RUN = re.compile(r"\d{8}T\d{12}Z-[0-9a-f]{16}")
FAMILY = re.compile(r"scene_\d{3}_[a-z0-9_]+")
VIEWS = {f"v{i:02d}" for i in range(1, 7)}
MEMBERS = {"rgb": "rgb.npy", "samples": "samples.npz", "truth": "truth.npz", "deploy": "deploy.npz"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read(path):
    return json.loads(path.read_bytes())


def inventory(dataset, split):
    require(dataset["schema"] == "el-vision-dataset-1" and dataset["split"] == split
            and split in {"train", "development"} and dataset.get("test_unlock_sha256") is None,
            "learning transport requires a registered training or development split")
    assets, views = {}, {}
    for row in dataset["views"]:
        family, view = row["family"], row["view"]
        require(FAMILY.fullmatch(family) and view in VIEWS and row["condition"] == "B0"
                and view not in views.setdefault(family, set()), "unexpected or repeated research view")
        views[family].add(view)
        require(row["status"] in {"render_failed", "completed"}, "unrecognized view disposition")
        success = row["status"] == "completed"
        require((row["rgb"] is not None) == success and row["samples"] is not None, "incomplete view")
        roles = {"samples": row["samples"], "rgb": row["rgb"]}
        if split == "train":
            require(row["full"] is None, "full reference arrays are outside the training transport")
        else:
            full = row["full"]
            require(set(full) == {"truth", "deploy"} and full["truth"] is not None
                    and (full["deploy"] is not None) == success, "unexpected development roles")
            roles.update(full)
        for role, spec in roles.items():
            if spec is None:
                continue
            name = f"{family}/B0/{view}/{MEMBERS[role]}"
            require(spec["path"] == name and name not in assets and DIGEST.fullmatch(spec["sha256"])
                    and type(spec["bytes"]) is int and spec["bytes"] > 0, "invalid data-member reference")
            assets[name] = spec
    require(views and all(v == VIEWS for v in views.values()), "expected all six planned views per family")
    return assets, sorted(views)


def copy_checked(stream, output, spec):
    measured, size = hashlib.sha256(), 0
    while block := stream.read(1024 * 1024):
        measured.update(block)
        size += len(block)
        require(size <= spec["bytes"], "member exceeds planned length")
        output.write(block)
    require(size == spec["bytes"] and measured.hexdigest() == spec["sha256"], "data member differs")


def pack(args):
    bundle = args.store.resolve() / "runs" / args.run
    require(RUN.fullmatch(args.run) and DIGEST.fullmatch(args.manifest_sha256)
            and digest(bundle / "manifest.json") == args.manifest_sha256, "wrong source run manifest")
    seal = read(bundle / "completion.json")
    manifest = read(bundle / "manifest.json")
    require(seal["execution_status"] == "succeeded" and seal["run_id"] == args.run
            and seal["manifest_sha256"] == args.manifest_sha256 and manifest["run_id"] == args.run
            and read(bundle / "run.json")["run_kind"] == "dataset_build", "source dataset is not sealed")
    files = {item["path"]: item for item in manifest["files"]}
    root = bundle / "payload"
    dataset = root / "dataset.json"
    require(digest(dataset) == args.dataset_sha256 == files["payload/dataset.json"]["sha256"],
            "dataset index differs from its recorder manifest")
    entries, families = inventory(read(dataset), args.split)
    entries["dataset.json"] = {"bytes": dataset.stat().st_size, "sha256": args.dataset_sha256}
    for name, spec in entries.items():
        source = root / name
        record = files["payload/" + name]
        require(source.is_file() and source.resolve() == source and spec["sha256"] == record["sha256"]
                and source.stat().st_size == spec["bytes"] == record["bytes"], "member differs from original seal")
    receipt = {"schema": "el-learning-transport-1", "source_run": args.run, "manifest_sha256": args.manifest_sha256,
               "dataset_sha256": args.dataset_sha256, "split": args.split, "families": families,
               "member_count": len(entries), "uncompressed_data_bytes": sum(v["bytes"] for v in entries.values())}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as work:
        staged = Path(work) / "dataset.zip"
        with zipfile.ZipFile(staged, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            archive.writestr(zipfile.ZipInfo("transport.json"), encode(receipt))
            for name, spec in sorted(entries.items()):
                member = zipfile.ZipInfo(name)
                member.compress_type = zipfile.ZIP_DEFLATED
                with (root / name).open("rb") as stream, archive.open(member, "w") as target:
                    copy_checked(stream, target, spec)
        value = digest(staged)
        if output.exists():
            require(not output.is_symlink() and digest(output) == value, "existing transport differs")
        else:
            os.link(staged, output)
    return {**receipt, "name": output.name, "sha256": value, "bytes": output.stat().st_size}


def unpack(args):
    require(DIGEST.fullmatch(args.sha256) and digest(args.package) == args.sha256, "wrong transport checksum")
    dest = args.output.resolve()
    require(not args.output.is_symlink(), "destination is a symbolic link")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.package) as archive:
        receipt = json.loads(archive.read("transport.json"))
        data = archive.read("dataset.json")
        measured = hashlib.sha256(data).hexdigest()
        require(receipt["schema"] == "el-learning-transport-1" and receipt["split"] == args.split
                and receipt["dataset_sha256"] == measured == args.dataset_sha256, "different learning dataset")
        entries, families = inventory(json.loads(data), args.split)
        entries["dataset.json"] = {"bytes": len(data), "sha256": measured}
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == set(entries) | {"transport.json"}
                and families == receipt["families"] and receipt["member_count"] == len(entries),
                "transport file list differs from declared research inputs")
        if dest.exists():
            actual = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
            require(actual == set(entries) and all((dest / n).resolve() == dest / n
                    and (dest / n).stat().st_size == s["bytes"] and digest(dest / n) == s["sha256"]
                    for n, s in entries.items()), "existing destination differs from the pinned transport")
            return receipt
        with tempfile.TemporaryDirectory(dir=dest.parent) as work:
            staged = Path(work) / "dataset"
            for name, spec in sorted(entries.items()):
                path = staged / name
                path.parent.mkdir(parents=True, exist_ok=True)
                require(archive.getinfo(name).file_size == spec["bytes"], "declared transfer size differs")
                with archive.open(name) as stream, path.open("xb") as target:
                    copy_checked(stream, target, spec)
            staged.rename(dest)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="action", required=True)
    writer = modes.add_parser("pack")
    writer.add_argument("--store", type=Path, required=True)
    writer.add_argument("--run", required=True)
    writer.add_argument("--manifest-sha256", required=True)
    reader = modes.add_parser("unpack")
    reader.add_argument("--package", type=Path, required=True)
    reader.add_argument("--sha256", required=True)
    for sub in (writer, reader):
        sub.add_argument("--dataset-sha256", required=True)
        sub.add_argument("--split", choices=("train", "development"), required=True)
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(pack(args) if args.action == "pack" else unpack(args)))
