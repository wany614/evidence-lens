"""Run the installed Stage A replay on a native CPU and retain its original per-view numerical arrays."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import zipfile


CASES = {
    "square": (256, 256),
    "landscape": (448, 252),
    "portrait": (252, 448),
    "photo": (378, 252),
    "odd": (375, 251),
    "legacy-letterbox": (256, 256),
}
VIEWS = tuple(f"v{i:02d}" for i in range(1, 7))
MODELS = ("B-L1-transfer", "MS-L1", "MS-frozen", "MS-ft")
NEW_WHEEL = "evidence_lens-0.5.0-py3-none-any.whl"
OLD_WHEEL = "evidence_lens-0.4.0-py3-none-any.whl"
ORIGINALS = "evidence-lens-original-replays-v1.zip"
EXAMPLES = "v05-bell-bundles.zip"
PROGRAMS = (
    "v05_install_replay.py",
    "v05_profile_accept.py",
    "v05_platform_replay.py",
    "v05_verify_neural.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    Path(path).write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def compact_arrays(installed, group, out):
    """Store original unopened NPZ byte streams and bind every member back to its scores.json."""
    methods = MODELS if group == "vision" else ("MS-L1",)
    targets = {}
    path = out / "score-arrays.zip"
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as archive:
        for tag, wh in [*CASES.items(), ("historical-original", (256, 256))]:
            for method in methods:
                base = (
                    installed / f"legacy-{group}-0.5.0"
                    if tag == "historical-original"
                    else installed / f"profiles-{group}" / tag
                )
                folder = base / method / "scores"
                spec = load(folder / "scores.json")
                require([v["id"] for v in spec["views"]] == list(VIEWS), "missing or extra target")
                states = spec["model"]
                key = f"{tag}/{method}"
                row = targets[key] = {
                    "analysis_wh": list(wh),
                    "model": states,
                    "scores_sha256": digest(folder / "scores.json"),
                    "bundle_sha256": spec["bundle_sha256"],
                    "views": {},
                }
                for v in spec["views"]:
                    ref = v["arrays"]
                    src = (folder / ref["path"]).resolve()
                    require(
                        src.is_relative_to(folder.resolve())
                        and src.is_file()
                        and digest(src) == ref["sha256"],
                        "score receipt does not identify actual array bytes",
                    )
                    name = f"{key}/{v['id']}.npz"
                    archive.write(src, name)
                    row["views"][v["id"]] = {
                        "path": name,
                        "sha256": ref["sha256"],
                        "bytes": src.stat().st_size,
                    }
    result = {
        "schema": "el-v05-platform-arrays-1",
        "run_id": os.environ["EL_RUN_ID"],
        "methods": list(methods),
        "cells": targets,
        "target_arrays": len(targets) * len(VIEWS),
        "archive_sha256": digest(path),
        "archive_bytes": path.stat().st_size,
    }
    save(out / "array-index.json", result)
    return result


def unpack_ticket(archive, expected, destination):
    require(digest(archive) == expected, "different CPU ticket archive")
    wanted = {NEW_WHEEL, EXAMPLES, "dinov2.tar.gz", "protocol.json", *PROGRAMS}
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        require(
            len(names) == len(set(names)) and set(names) == wanted | {"ticket.json"},
            "different CPU ticket roster",
        )
        ticket = json.loads(z.read("ticket.json"))
        require(
            ticket["schema"] == "el-v05-cpu-ticket-1" and set(ticket["files"]) == wanted,
            "wrong ticket schema or roster",
        )
        for name in wanted:
            path = destination / name
            with z.open(name) as source, path.open("xb") as target:
                shutil.copyfileobj(source, target)
            require(
                digest(path) == ticket["files"][name]["sha256"]
                and path.stat().st_size == ticket["files"][name]["bytes"],
                "changed CPU ticket member: " + name,
            )
    return ticket


def worker(a):
    require(
        os.environ.get("EL_RUN_ID") and platform.system().lower() == a.native, "run under the native recorder"
    )
    require(a.native != "windows" or (a.models and a.models.is_dir()), "missing deployed CPU model directory")
    a.out.mkdir(exist_ok=True, parents=True)
    require(not any(a.out.iterdir()), "need a clean recorded output directory")
    with tempfile.TemporaryDirectory(dir=a.out.parent) as scratch:
        tmp = Path(scratch)
        ticket = unpack_ticket(a.packet, a.packet_sha256, tmp)
        require(
            digest(Path(__file__)) == ticket["files"]["v05_platform_replay.py"]["sha256"],
            "different CPU worker",
        )
        require(
            digest(a.old_wheel) == ticket["old_wheel_sha256"]
            and digest(a.original) == ticket["original_replay_sha256"],
            "original wheel or control archive differs",
        )
        group = "vision" if a.native == "windows" else "default"
        groups = ["default", "vision"] if group == "vision" else ["default"]
        command = [
            sys.executable,
            "-I",
            "-B",
            str(tmp / "v05_install_replay.py"),
            "--wheel",
            str(tmp / NEW_WHEEL),
            "--old-wheel",
            str(a.old_wheel.resolve()),
            "--original",
            str(a.original.resolve()),
            "--wheel-sha256",
            ticket["files"][NEW_WHEEL]["sha256"],
            "--examples",
            str(tmp / EXAMPLES),
            "--examples-sha256",
            ticket["files"][EXAMPLES]["sha256"],
            "--worker",
            str(tmp / "v05_profile_accept.py"),
            "--groups",
            *groups,
            "--keep-scores",
            "--out",
            str(tmp / "installed"),
        ]
        if a.wheelhouse:
            command += ["--wheelhouse", str(a.wheelhouse.resolve())]
        if group == "vision":
            command += [
                "--models",
                str(a.models.resolve()),
                "--neural-source",
                str(tmp / "dinov2.tar.gz"),
                "--neural-worker",
                str(tmp / "v05_verify_neural.py"),
                "--protocol",
                str(tmp / "protocol.json"),
            ]
        subprocess.run(command, check=True)
        installed = tmp / "installed"
        for path in installed.glob("*.json"):
            shutil.copyfile(path, a.out / path.name)
        report = load(a.out / "installation.json")
        require(
            set(report["groups"]) == set(groups) and report["run_id"] == os.environ["EL_RUN_ID"],
            "wrong group set or installation receipt",
        )
        if group == "vision":
            shutil.copyfile(installed / "neural/neural.json", a.out / "neural.json")
            neural = load(a.out / "neural.json")
            require(
                digest(a.out / "neural.json") == report["groups"]["vision"]["official_numerical_sha256"]
                and neural["run_id"] == os.environ["EL_RUN_ID"]
                and neural["passed"],
                "wrong official reference receipt",
            )
        arrays = compact_arrays(installed, group, a.out)
        if a.native == "linux":
            interfaces = sorted(name for _, name in socket.if_nameindex())
            require(interfaces == ["lo"], "Linux acceptance ran with a non-loopback interface")
        else:
            interfaces = None
        save(
            a.out / "platform.json",
            {
                "schema": "el-v05-native-platform-1",
                "run_id": os.environ["EL_RUN_ID"],
                "native_os": platform.system(),
                "architecture": platform.machine(),
                "python": platform.python_version(),
                "packet_run_id": ticket["run_id"],
                "packet_sha256": a.packet_sha256,
                "groups": groups,
                "installation_sha256": digest(a.out / "installation.json"),
                "array_index_sha256": digest(a.out / "array-index.json"),
                "score_arrays_sha256": arrays["archive_sha256"],
                "score_arrays_count": arrays["target_arrays"],
                "official_reference_sha256": digest(a.out / "neural.json") if group == "vision" else None,
                "linux_visible_interfaces": interfaces,
            },
        )


def linux(a):
    if not a.wheelhouse and a.dependency_wheel:
        with tempfile.TemporaryDirectory() as scratch:
            a.wheelhouse = Path(scratch)
            for dep in a.dependency_wheel:
                shutil.copyfile(dep, a.wheelhouse / dep.name)
            return linux(a)
    require(a.wheelhouse and a.wheelhouse.is_dir(), "Linux requires an explicit offline wheelhouse")
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", a.image]))[0]
    require(
        image["Id"] == a.image and image["Architecture"] == "arm64" and image["Os"] == "linux",
        "different Linux image",
    )
    a.out.mkdir(exist_ok=True, parents=True)
    mounts = [
        (a.out, "/out", "rw"),
        (Path(__file__), "/app/worker.py", "ro"),
        (a.old_wheel, "/inputs/" + OLD_WHEEL, "ro"),
        (a.original, "/inputs/" + ORIGINALS, "ro"),
        (a.packet, "/inputs/packet.zip", "ro"),
        (a.wheelhouse, "/wheels", "ro"),
    ]
    cmd = [
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,exec,size=2g",
        "-e",
        "EL_RUN_ID=" + os.environ["EL_RUN_ID"],
    ]
    for host, guest, mode in mounts:
        cmd.extend(["-v", str(host.resolve()) + ":" + guest + ":" + mode])
    cmd.extend(
        [
            a.image,
            "python",
            "-I",
            "-B",
            "/app/worker.py",
            "worker",
            "--native",
            "linux",
            "--packet",
            "/inputs/packet.zip",
            "--packet-sha256",
            a.packet_sha256,
            "--old-wheel",
            "/inputs/" + OLD_WHEEL,
            "--original",
            "/inputs/" + ORIGINALS,
            "--wheelhouse",
            "/wheels",
            "--out",
            "/out/accepted",
        ]
    )
    subprocess.run(cmd, check=True)
    native = load(a.out / "accepted/platform.json")
    require(
        native["native_os"] == "Linux"
        and native["architecture"] in ("aarch64", "arm64")
        and native["groups"] == ["default"]
        and native["score_arrays_count"] == 42,
        "incomplete native Linux result",
    )
    save(
        a.out / "linux.json",
        {
            "schema": "el-v05-offline-linux-1",
            "run_id": os.environ["EL_RUN_ID"],
            "platform_sha256": digest(a.out / "accepted/platform.json"),
            "native": native,
            "image_id": a.image,
            "network": "none",
            "readonly_root": True,
            "fresh_container": True,
            "wheelhouse": {p.name: digest(p) for p in sorted(a.wheelhouse.glob("*.whl"))},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("worker", "linux"))
    for key in ("old-wheel", "original", "packet", "out", "wheelhouse", "models"):
        parser.add_argument("--" + key, type=Path, required=key not in ("wheelhouse", "models"))
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--dependency-wheel", action="append", default=[], type=Path)
    parser.add_argument("--image")
    parser.add_argument("--native", choices=("windows", "linux"))
    args = parser.parse_args()
    (worker if args.action == "worker" else linux)(args)
