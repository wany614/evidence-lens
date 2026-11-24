"""Package the four authored workloads and their actual verified CLI scores and HTML reports."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

from evidence_lens_bundle.common import encoded, parts, read_file, require, sha, write
from evidence_lens_bundle.numerics import METHODS


def pack(workloads, workloads_sha, bridge_dir, bridge_sha, verifier, readme, legal, output):
    with workloads.open("rb") as f:
        require(
            hashlib.file_digest(f, "sha256").hexdigest() == workloads_sha, "original prepared controls differ"
        )
    raw = read_file(bridge_dir, "numerical-bridge.json")
    require(sha(raw) == bridge_sha, "recorded numerical bridge differs")
    bridge = json.loads(raw)
    require(
        bridge["stage"] == "original_development"
        and len(bridge["cells"]) == 4
        and bridge["summary"]["compared_method_views"] == 64
        and all(
            bridge["summary"][key] == 64
            for key in ("old_to_off_bit_exact", "old_to_on_bit_exact", "off_to_on_bit_exact")
        ),
        "the signed four-method original replay is incomplete",
    )
    index = {
        "schema": "el-original-replay-gold-1",
        "bridge_run_id": bridge["run_id"],
        "bridge_sha256": bridge_sha,
        "workloads_archive_sha256": workloads_sha,
        "sequences": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "evidence-lens-original-replays-v1.zip"
    seen = set()
    with ZipFile(path, "x") as dest, ZipFile(workloads) as src:

        def put(name, data):
            parts(name)
            require(name not in seen, "duplicate output name")
            seen.add(name)
            z = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            z.external_attr = 0o100644 << 16
            z.compress_type = ZIP_STORED if name.endswith((".png", ".npz")) else ZIP_DEFLATED
            dest.writestr(z, data)

        controls = json.loads(src.read("workloads.json"))
        permitted = {s["sequence_id"]: s for s in controls["sequences"]}
        require(
            set(permitted) == {"bell-M2", "bell-M4", "beads-M2", "beads-M4"}, "unexpected workload roster"
        )
        for n in src.namelist():
            ps = parts(n)
            if n == "workloads.json":
                put("workloads.json", src.read(n))
                continue
            require(
                ps[0] in permitted
                and (
                    len(ps) == 2
                    and ps[1] in ("bundle.json", "cameras.json", "producer.json")
                    or len(ps) == 3
                    and ps[1] in ("renders", "sources")
                ),
                "unregistered workload role",
            )
            put("inputs/" + n, src.read(n))
        for i, cell in enumerate(bridge["cells"]):
            seq = (
                {"offset-bell-cage": "bell", "helical-beads": "beads"}[cell["object_id"]]
                + "-"
                + cell["layout"][:2]
            )
            spec = permitted[seq]
            require(
                cell["bundle_sha256"] == spec["bundle_sha256"] == sha(src.read(seq + "/bundle.json")),
                "bridge and prepared inputs differ",
            )
            entry = {
                "object_id": cell["object_id"],
                "condition": cell["layout"][:2],
                "bundle_sha256": cell["bundle_sha256"],
                "target_order": spec["actual_target_ids"],
                "methods": {},
            }
            for m in METHODS:
                old = bridge_dir / f"cell-{i:02d}" / m
                report = read_file(old, "report.html")
                sj = read_file(old, "scores/scores.json")
                r = json.loads(sj)
                gold = cell["files"][m]
                require(
                    sha(report) == gold["html_sha256"]
                    and sha(sj) == gold["scores_sha256"]
                    and r["target_order"] == entry["target_order"]
                    and r["bundle_sha256"] == entry["bundle_sha256"],
                    "original report/receipt fails its content check",
                )
                prefix = f"expected/{seq}/{m}/"
                put(prefix + "report.html", report)
                put(prefix + "scores/scores.json", sj)
                roles = [s["display"] for s in r["sources"]]
                for view in r["views"]:
                    roles.append(view["arrays"])
                    if view["prediction"] is not None:
                        roles.append(view["prediction"])
                for f in roles:
                    data = read_file(old / "scores", f["path"])
                    require(
                        len(data) == f["bytes"] and sha(data) == f["sha256"],
                        "original numerical member differs",
                    )
                    put(prefix + "scores/" + f["path"], data)
                entry["methods"][m] = {
                    "scores_sha256": sha(sj),
                    "html_sha256": sha(report),
                    "model": r["model"],
                }
            index["sequences"][seq] = entry
        put("numerical-bridge.json", raw)
        gold_json = encoded(index)
        put("expected.json", gold_json)
        put("accept.py", verifier.read_bytes())
        put("README.md", readme.read_bytes())
        allowed = {
            "LICENSE",
            "CC-BY-4.0.txt",
            "RESEARCH-RECORDS.txt",
            "ORIGINAL-EXAMPLE-NOTICE.txt",
            "multisource-assets.json",
        }
        require({p.name for p in legal.iterdir()} == allowed, "incomplete original rights notices")
        for p in sorted(legal.iterdir()):
            put("licenses/" + p.name, read_file(legal, p.name))
    with path.open("rb") as f:
        checksum = hashlib.file_digest(f, "sha256").hexdigest()
    result = {
        "schema": "el-original-replay-archive-1",
        "run_id": os.getenv("EL_RUN_ID"),
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": checksum,
        "expected_sha256": sha(gold_json),
        "numerical_bridge_sha256": sha(raw),
        "bridge_run_id": bridge["run_id"],
        "sequences": 4,
        "reports": 16,
        "method_views": 64,
        "rights": "Authored development material; CC BY 4.0 and Apache-2.0 software.",
    }
    write(output / "original-replays.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("workloads", "bridge-dir", "verifier", "readme", "legal", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("workloads-sha256", "bridge-sha256"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    print(
        json.dumps(
            pack(
                a.workloads,
                a.workloads_sha256,
                a.bridge_dir,
                a.bridge_sha256,
                a.verifier,
                a.readme,
                a.legal,
                a.out,
            )
        )
    )
