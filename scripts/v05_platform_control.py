"""Record the shared CPU ticket, the offline Linux verifier and an immutable Windows handoff."""

import argparse
import json
import os
from pathlib import Path
import zipfile

from dual_records import bind, resolve, selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require
from v05_platform_replay import EXAMPLES, NEW_WHEEL, OLD_WHEEL, ORIGINALS, PROGRAMS

OLD_RUN = "20261124T053239837552Z-8ec25846c9824287"
DISPATCH_RUN = "20261124T104214538720Z-d2fdaadf100546dd"
BUNDLES_RUN = "20261124T104507616076Z-ce303b94f64c4b60"
LINUX_IMAGE = "sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b"
DEPS_RUN = "20261124T021929760866Z-fe12804f88064761"
DEPS = {
    "numpy": "numpy-2.5.5-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
    "pillow": "pillow-12.3.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
}


def ticket(a):
    inputs = {k: Path(v) for k, v in a.input}
    cfg = load(a.config)
    files = {
        NEW_WHEEL: inputs["new_wheel"],
        EXAMPLES: inputs["examples"],
        "dinov2.tar.gz": inputs["official"],
        "protocol.json": inputs["registration"],
        **{n: Path(__file__).with_name(n) for n in PROGRAMS},
    }
    inventory = {n: {"sha256": sha(p), "bytes": p.stat().st_size} for n, p in files.items()}
    doc = {
        "schema": "el-v05-cpu-ticket-1",
        "run_id": os.environ["EL_RUN_ID"],
        "files": inventory,
        "old_wheel_sha256": cfg["sources"]["old_wheel"]["sha256"],
        "original_replay_sha256": cfg["sources"]["original"]["sha256"],
        "registered_resources": cfg["sources"],
        "groups": {"Windows": ["default", "vision"], "Linux": ["default"]},
        "neural_threads": 4,
        "native_output": "platform.json, installation.json, per-group and per-version diagnostics, array-index.json, score-arrays.zip; Windows also returns neural.json.",
    }
    save(a.out / "ticket.json", doc)
    packed = a.out / "v05-cpu-ticket.zip"
    with zipfile.ZipFile(packed, "x", compression=zipfile.ZIP_STORED) as z:
        for n, p in sorted({**files, "ticket.json": a.out / "ticket.json"}.items()):
            z.write(p, n)
    save(
        a.out / "dispatch.json",
        {
            "schema": "el-v05-platform-dispatch-1",
            "run_id": os.environ["EL_RUN_ID"],
            "packet_sha256": sha(packed),
            "packet_bytes": packed.stat().st_size,
            "ticket_sha256": sha(a.out / "ticket.json"),
            "worker_sha256": inventory["v05_platform_replay.py"]["sha256"],
            "new_wheel_sha256": inventory[NEW_WHEEL]["sha256"],
            "old_wheel_sha256": doc["old_wheel_sha256"],
            "original_sha256": doc["original_replay_sha256"],
            "examples_sha256": inventory[EXAMPLES]["sha256"],
        },
    )


def control(a):
    store = a.store.resolve()
    history = store / "history"
    roots = (store, history)
    old = {
        "old_wheel": bind(history, OLD_RUN, "payload/" + OLD_WHEEL),
        "original": bind(history, OLD_RUN, "payload/" + ORIGINALS),
    }
    if a.action == "ticket":
        pins = {
            **old,
            "new_wheel": bind(store, a.wheel_run, a.wheel_member),
            "examples": bind(store, BUNDLES_RUN, "payload/" + EXAMPLES),
            "registration": selected(store, "a-grid-registration"),
            "official": bind(store, DISPATCH_RUN, "inputs/official/pinned-source.tar.gz"),
        }
        require(pins["new_wheel"]["sha256"] == a.wheel_sha256, "candidate wheel differs from issued digest")
        cmd = [
            "{python}",
            "-B",
            "{source}/scripts/v05_platform_control.py",
            "ticket-worker",
            "--config",
            "{config}",
            "--out",
            "{out}",
        ]
        for role in pins:
            cmd += ["--input", role, "{input:" + role + "}"]
        return stage(
            store,
            "a-cpu-platform-ticket",
            "derivation",
            pins,
            {"native_job": "frozen preopening installed CPU replay and official source"},
            cmd,
            "payload/dispatch.json",
            95_000_000,
        )
    dispatch_pin = selected(store, "a-cpu-platform-ticket")
    packed = bind(store, dispatch_pin["run_id"], "payload/v05-cpu-ticket.zip")
    dispatch = load(resolve(roots, dispatch_pin))
    require(packed["sha256"] == dispatch["packet_sha256"], "packet/receipt identity differs")
    if a.action == "linux":
        pins = {
            **old,
            "packet": packed,
            "dispatch": dispatch_pin,
            **{k: bind(history, DEPS_RUN, "inputs/wheels/linux-wheelhouse/" + n) for k, n in DEPS.items()},
        }
        cmd = [
            "{python}",
            "-B",
            "{source}/scripts/v05_platform_replay.py",
            "linux",
            "--image",
            LINUX_IMAGE,
            "--old-wheel",
            "{input:old_wheel}",
            "--original",
            "{input:original}",
            "--packet",
            "{input:packet}",
            "--packet-sha256",
            packed["sha256"],
            "--out",
            "{out}",
        ]
        for k in DEPS:
            cmd += ["--dependency-wheel", "{input:" + k + "}"]
        return stage(
            store,
            "a-linux-installed-default",
            "transfer_verify",
            pins,
            {
                "fresh_linux_container": LINUX_IMAGE,
                "network": "none",
                "native_architecture": "arm64",
                "groups": ["default"],
            },
            cmd,
            "payload/linux.json",
            85_000_000,
        )
    directory = a.exchange
    directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, name, pin in [
        ("packet", "v05-cpu-ticket.zip", packed),
        ("dispatch", "dispatch.json", dispatch_pin),
    ]:
        cmd = [
            "{python}",
            "-B",
            "{source}/scripts/general_exchange.py",
            "--source",
            "{input:asset}",
            "--destination",
            str((directory / name).resolve()),
            "--receipt",
            "{out}/local.json",
        ]
        results[label] = stage(
            store,
            "a-cpu-platform-send-" + label,
            "archive_verify",
            {"asset": pin},
            {"destination": "existing dual-track A Windows handoff", "filename": name},
            cmd,
            "payload/local.json",
            5000,
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("ticket", "ticket-worker", "linux", "send"))
    for key in ("store", "config", "out", "exchange"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--input", action="append", nargs=2, default=[])
    parser.add_argument("--wheel-run")
    parser.add_argument("--wheel-member", default="payload/" + NEW_WHEEL)
    parser.add_argument("--wheel-sha256")
    a = parser.parse_args()
    if a.action == "ticket-worker":
        ticket(a)
    else:
        print(json.dumps(control(a)), flush=True)
