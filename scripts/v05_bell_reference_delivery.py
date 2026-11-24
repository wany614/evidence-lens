"""Package the independent authored reference observations after the comparison opening."""

import argparse
import json
import os
from pathlib import Path
import zipfile

from dual_records import bind, resolve, selected, stage
from evidence_lens._bundle import encoded, load, save, sha
from evidence_lens_bundle.common import require
from v05_bell_common import TARGETS


def control(s):
    refs = selected(s, "a-bell-references")
    opening = selected(s, "a-bell-reference-opening")
    report = load(resolve((s, s/"history"), refs))
    require(report["role"] == "references" and report["opening"] == opening and report["target_reference_renders"] == 6,
            "the separate post-opening reference run is required")
    pins = {"references": refs, "opening": opening}
    for view in TARGETS:
        record = report["products"]["landscape"][view]
        for kind in ("rgb8", "display_fp32"):
            pins[view + "-" + kind] = bind(s, refs["run_id"], "payload/" + record[kind]["path"])
    cmd = ["{python}", "-B", "{source}/scripts/v05_bell_reference_delivery.py", "worker", "--config", "{config}", "--out", "{out}"]
    for key in pins:
        cmd += ["--input", key, "{input:" + key + "}"]
    return stage(s, "a-bell-reference-delivery", "dataset_build", pins,
                 {"role": "independent evaluation only", "views": TARGETS, "size": [896, 504], "license": "CC-BY-4.0"},
                 cmd, "payload/references.json", 40_000_000)


def worker(cfg, paths, out):
    original = load(paths["references"])
    gate = load(paths["opening"])
    require(gate["registered_scores_sealed"] and gate["terminal_score_count"] == 48
            and original["opening"] == cfg["sources"]["opening"] and original["source_renders"] == 0
            and set(original["products"]["landscape"]) == set(TARGETS), "invalid reference lineage or roles")
    archive = out / "v05-bell-independent-references.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as z:
        z.writestr("photos.json", paths["references"].read_bytes())
        z.writestr("opening.json", paths["opening"].read_bytes())
        z.writestr("NOTICE.txt", ("Six independently generated Blender Cycles observations of the original Offset Bell Cage scene.\n"
                  "Creator: wany614, Evidence Lens. CC BY 4.0, https://creativecommons.org/licenses/by/4.0/\n"
                  "Reference photographs are reserved for independent evaluation of sealed predictions.\n").encode())
        for view in TARGETS:
            row = original["products"]["landscape"][view]
            require([row["camera"]["width"], row["camera"]["height"]] == cfg["size"], "incorrect original camera")
            for kind in ("rgb8", "display_fp32"):
                member = row[kind]
                file = paths[view + "-" + kind]
                require(sha(file) == member["sha256"] and file.stat().st_size == member["bytes"], "changed independent observation")
                z.writestr(member["path"], file.read_bytes())
    save(out / "references.json", {"schema": "el-v05-bell-independent-reference-archive-1", "run_id": os.environ["EL_RUN_ID"],
         "opening": cfg["sources"]["opening"], "references": cfg["sources"]["references"], "target_order": TARGETS,
         "photographs": 6, "size": cfg["size"], "formats": ["display-sRGB RGB8 PNG", "display-sRGB FP32 NPZ"],
         "license": cfg["license"], "archive": {"path": archive.name, "bytes": archive.stat().st_size, "sha256": sha(archive)}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("publish", "worker"))
    for key in ("store", "config", "out"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--input", action="append", nargs=2, default=[])
    args = parser.parse_args()
    if args.action == "worker":
        worker(load(args.config), {k: Path(v) for k, v in args.input}, args.out)
    else:
        print(json.dumps(control(args.store.resolve())))
