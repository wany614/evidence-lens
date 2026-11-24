"""Record Windows intake, compact verified Mac arrays and compare the two native host pairs."""

import argparse
import json
from pathlib import Path
import zipfile

from dual_records import bind, resolve, selected, stage
from evidence_lens._bundle import load, sha
from evidence_lens._verify import require
from t0_reconstruct_intake import receive_verified_zip
from v05_bell_receive import history_roots

PROJECT = Path(__file__).resolve().parents[1]


def windows_intake(store, receipt):
    r = load(receipt)
    rid, manifest, container = r["run_id"], r["manifest_sha256"], r["sha256"]
    require(
        r.get("status", r.get("execution_status")) == "succeeded" and r["package"] == rid + ".zip",
        "missing successful Windows CPU completion",
    )
    path = receipt.parent / r["package"]
    require(path.stat().st_size == r["bytes"] and sha(path) == container, "different returned CPU container")
    ticket = selected(store, "a-cpu-platform-ticket")
    packet = bind(store, ticket["run_id"], "payload/v05-cpu-ticket.zip")
    with zipfile.ZipFile(path) as z:
        info, inputs, end, native = (
            json.loads(z.read(n))
            for n in ("run.json", "inputs.json", "completion.json", "payload/platform.json")
        )
        require(
            info["run_id"] == native["run_id"] == rid
            and info["run_kind"] == "transfer_verify"
            and info["trial_id"] == "a-windows-installed-cpu"
            and info["experiment_id"] == "dual-track-a"
            and end["manifest_sha256"] == manifest
            and end["execution_status"] == "succeeded",
            "different native CPU job",
        )
        require(
            native["native_os"] == "Windows"
            and native["architecture"].lower() in ("amd64", "x86_64")
            and native["packet_run_id"] == ticket["run_id"]
            and native["packet_sha256"] == packet["sha256"]
            and set(native["groups"]) == {"default", "vision"}
            and native["score_arrays_count"] == 168,
            "wrong Windows workload or receipt",
        )
        require(
            any(
                p["run_id"] == ticket["run_id"] and p["manifest_sha256"] == ticket["manifest_sha256"]
                for p in inputs["parents"]
            ),
            "native CPU archive omits its ticket parent",
        )
        require(
            any(
                item["parent_run_id"] == ticket["run_id"]
                and packet["sha256"] in {f["sha256"] for f in item["files"]}
                for item in inputs["inputs"]
            ),
            "native CPU archive omits the original ticket bytes",
        )
    anchors = history_roots(store, inputs["parents"])
    caught = receive_verified_zip(path, store, container, PROJECT / "configs/storage.json", anchors, ticket)
    require(
        caught["status"] in ("accepted", "identical")
        and caught["run_id"] == rid
        and caught["manifest_sha256"] == manifest,
        "full Windows archive did not verify: " + str(caught),
    )
    pins = {
        role: bind(store, rid, "payload/" + name)
        for role, name in {
            "native": "platform.json",
            "installation": "installation.json",
            "array_index": "array-index.json",
            "neural": "neural.json",
        }.items()
    }
    for field, role in (
        ("installation_sha256", "installation"),
        ("array_index_sha256", "array_index"),
        ("official_reference_sha256", "neural"),
    ):
        require(native[field] == pins[role]["sha256"], "changed native payload: " + role)
    require(
        native["score_arrays_sha256"] == bind(store, rid, "payload/score-arrays.zip")["sha256"],
        "changed native score archive",
    )
    pins["ticket"] = ticket
    cmd = [
        "{python}",
        "-B",
        "{source}/scripts/v05_platform_records.py",
        "intake-worker",
        "--config",
        "{config}",
        "--out",
        "{out}",
    ]
    return stage(
        store,
        "a-windows-installed-intake",
        "transfer_verify",
        pins,
        {
            "native_payload": rid,
            "native_manifest_sha256": manifest,
            "container_sha256": container,
            "container_bytes": path.stat().st_size,
            "return_receipt_sha256": sha(receipt),
            "native_windows_inputs_verified": True,
        },
        cmd,
        "payload/readback.json",
        15000,
        {"return_receipt": receipt},
    )


def local(store, action):
    roots = (store, store / "history")
    if action == "mac-pack":
        p = selected(store, "a-mac-install")
        path = resolve(roots, p)
        require(path.name == "installation.json", "expected the original Mac install receipt")
        cmd = [
            "{python}",
            "-B",
            "{source}/scripts/v05_platform_compare.py",
            "mac-pack",
            "--installed",
            str(path.parent),
            "--parent-manifest",
            str(store / "runs" / p["run_id"] / "manifest.json"),
            "--manifest-sha256",
            p["manifest_sha256"],
            "--out",
            "{out}",
        ]
        return stage(
            store,
            "a-mac-array-projection",
            "derivation",
            {"native": p},
            {"source": "verified original Mac per-view arrays"},
            cmd,
            "payload/mac-arrays.json",
            85_000_000,
        )
    is_linux = action == "compare-linux"
    mac = selected(store, "a-mac-array-projection")
    host = selected(store, "a-linux-installed-default" if is_linux else "a-windows-installed-intake")
    native_root = host["run_id"] if is_linux else load(resolve(roots, host))["native_run_id"]
    prefix = "payload/accepted/" if is_linux else "payload/"
    pins = {
        "mac_receipt": mac,
        "host_receipt": host,
        "mac_index": bind(store, mac["run_id"], "payload/array-index.json"),
        "mac_arrays": bind(store, mac["run_id"], "payload/score-arrays.zip"),
        "host_index": bind(store, native_root, prefix + "array-index.json"),
        "host_arrays": bind(store, native_root, prefix + "score-arrays.zip"),
    }
    platform = "Linux" if is_linux else "Windows"
    cmd = [
        "{python}",
        "-B",
        "{source}/scripts/v05_platform_compare.py",
        "compare",
        "--host",
        platform,
        "--out",
        "{out}",
    ]
    for role in ("mac_index", "mac_arrays", "host_index", "host_arrays"):
        cmd.extend(["--" + role.replace("_", "-"), "{input:" + role + "}"])
    return stage(
        store,
        "a-installed-differences-" + platform.lower(),
        "evaluation",
        pins,
        {
            "signed_difference": "other native CPU minus Mac, float64 on common finite",
            "quotas_percent": [10, 20],
            "tie": "row-major native index",
            "target_reference_reads": 0,
        },
        cmd,
        "payload/differences.json",
        1_500_000,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("receive", "intake-worker", "mac-pack", "compare-linux", "compare-windows")
    )
    for k in ("store", "receipt", "config", "out"):
        parser.add_argument("--" + k, type=Path)
    a = parser.parse_args()
    if a.action == "intake-worker":
        import os
        from evidence_lens._bundle import save

        cfg = load(a.config)
        save(
            a.out / "readback.json",
            {
                "schema": "el-v05-native-windows-cpu-readback-1",
                "run_id": os.environ["EL_RUN_ID"],
                "native_run_id": cfg["native_payload"],
                "native_manifest_sha256": cfg["native_manifest_sha256"],
                "sources": cfg["sources"],
                "container_sha256": cfg["container_sha256"],
                "container_bytes": cfg["container_bytes"],
            },
        )
    else:
        print(
            json.dumps(
                windows_intake(a.store.resolve(), a.receipt)
                if a.action == "receive"
                else local(a.store.resolve(), a.action)
            ),
            flush=True,
        )
