"""Independently recover C exhibit colours and counts from the public release-format archive."""

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from dual_records import selected, stage
from dtu_eval import read_pinned
from evidence_lens._bundle import load, save
from evidence_lens._verify import require


def register(store):
    labels = {"case": "c-foundation-case-data", "bell": "a-bell-landscape-comparison-packaged",
              "gallery": "b-eval-b05-perforated-gallery-r128", "gallery_ranks": "b-matrix-b05-perforated-gallery-r128"}
    parents = {k: selected(store, v) for k, v in labels.items()}
    return stage(store, "c-foundation-case-readback", "archive_verify", parents,
                 {"scope": "All public exact archives, display PNGs and original independent denominators for the two authored cases"},
                 ["{python}", "-B", "{source}/scripts/foundation_case_readback.py", "verify",
                  "--store", str(store), "--config", "{config}", "--out", "{out}"], "payload/readback.json", 30000)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def image(raw):
    require(raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[24:26] == b"\x08\x02", "PNG encoding differs")
    with Image.open(io.BytesIO(raw)) as obj:
        require(obj.mode == "RGB" and obj.format == "PNG" and getattr(obj, "n_frames", 1) == 1, "invalid display member")
        return np.asarray(obj)


def decode(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as arr:
        return dict(arr)


def compare_image(a, b):
    require(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b), "public pixels differ from the array")


def rgb(col):
    return bytes.fromhex(col.lstrip("#"))


def categorical(codes, entries):
    lookup = np.zeros((256, 3), np.uint8)
    for e in entries:
        lookup[e["code"]] = np.frombuffer(rgb(e["color"]), np.uint8)
    return lookup[codes]


def continuous(risk, legend):
    stops = [np.frombuffer(rgb(entry["color"]), np.uint8) for entry in legend]
    fractions = np.nan_to_num(risk.astype(np.float64), nan=0.) * 4
    left = np.minimum(fractions.astype(np.int32), 3)
    w = (fractions - left)[..., None]
    ramp = np.array(stops[:5], dtype=np.float64)
    result = np.rint(ramp[left] * (1. - w) + ramp[left + 1] * w).astype(np.uint8)
    result[~np.isfinite(risk)] = stops[-1]
    return result


def verify(store, config, out):
    pins = load(config)["sources"]
    inputs = {k: read_pinned((store,), v) for k, v in pins.items()}
    native, idx = inputs["case"]
    index = load(idx)
    archive = index["source_photo_archive"]
    files = load(native.open("payload/case-files.json"))["files"]
    external = load(native.open("payload/source-photos.json"))["files"]
    pngs = {}
    for row in files:
        raw = native.open("payload/" + row["path"]).read_bytes()
        pixels = image(raw)
        require(sha(raw) == row["sha256"] and sha(pixels.tobytes()) == row["pixel_sha256"] and len(raw) == row["bytes"],
                "offline positive inventory mismatch")
        pngs[row["path"]] = pixels
    a_run, a_manifest = inputs["bell"]
    a_proof = {v["view"]: v for v in load(a_manifest)["views"]}
    with a_run.open("payload/per-view-populations.csv").open() as f:
        pop = {(v["view"], v["method"]): v for v in csv.DictReader(f)
               if v["route"] == "landscape" and v["plane"] == "own_content"}
    b_proof = load(inputs["gallery"][1])["views"]["h00"]
    b_ranks = load(inputs["gallery_ranks"][1])["views"]["h00"]
    stats = Counter()
    rows = []
    zip_path = native.open("payload/" + archive["file"])
    with zip_path.open("rb") as fp:
        require(hashlib.file_digest(fp, "sha256").hexdigest() == archive["sha256"], "photo archive digest differs")
    with ZipFile(zip_path) as z:
        inside = z.read("contents.json")
        archive_names = [r["member"] for r in external] + ["contents.json"]
        require(len(set(z.namelist())) == len(archive_names) == len(z.infolist()) and set(archive_names) == set(z.namelist())
                and external == json.loads(inside)["files"] and sha(inside) == archive["index_sha256"], "ZIP contains extra members")
        for row in external:
            raw = z.read(row["member"])
            require(len(raw) == row["bytes"] and sha(raw) == row["sha256"], "changed archive member")
        for case in index["cases"]:
            is_bell = case["id"] == "offset-bell-cage"
            for src in case["sources"]:
                hi = Image.fromarray(image(z.read(src["original"]["member"])))
                actual = image(z.read(src["analysis"]["member"]))
                compare_image(np.asarray(hi.resize(actual.shape[1::-1], Image.Resampling.LANCZOS)), actual)
                thumb = pngs[src["image"]["path"]]
                compare_image(np.asarray(Image.fromarray(actual).resize(thumb.shape[1::-1], Image.Resampling.LANCZOS)), thumb)
                stats["independently_resampled_sources"] += 1
            for v in case["views"]:
                view = v["id"]
                raw = z.read(v["native_prediction"]["member"])
                pred = decode(raw)["rgb"]
                require(sha(raw) == v["target_prediction_sha256"] and sha(pred.tobytes()) == v["target_rgb_sha256"],
                        "prediction or RGB hash differs")
                p8 = np.rint(np.clip(pred, 0, 1) * np.float32(255)).astype(np.uint8)
                compare_image(p8, pngs[v["target"]["path"]])
                ev = v["evaluation"]
                if is_bell:
                    measured = decode(z.read(ev["native_measurement"]["member"]))["rgb"].astype(np.float64)
                    ref = ((measured[::2, ::2] + measured[1::2, ::2] + measured[::2, 1::2] + measured[1::2, 1::2]) * .25).astype(np.float32)
                    compare_image(np.rint(ref * 255).astype(np.uint8), pngs[ev["reference"]["path"]])
                else:
                    ref = pngs[ev["reference"]["path"]].astype(np.float32) / np.float32(255.)
                    full_record = next(c for c in v["conditions"] if c["k"] == case["available_count"])
                    full = decode(z.read(full_record["native_arrays"]["member"]))
                mae = np.mean(np.abs(pred - ref), axis=-1, dtype=np.float32)
                finite = np.isfinite(mae)
                high = finite & (mae > np.float32(.05))
                codes = np.where(finite, high, 2).astype(np.uint8)
                compare_image(categorical(codes, ev["legend"]), pngs[ev["error"]["path"]])
                require((int(high.sum()), int(finite.sum())) == (ev["counts"]["errors"], ev["counts"]["labelled"]),
                        "reference label counts disagree")
                checks = []
                for c in v["conditions"]:
                    native_id = c["native_arrays"]["sha256"]
                    a = decode(z.read(c["native_arrays"]["member"]))
                    reason = a["unknown_reason"] if is_bell else a["reason"]
                    if is_bell:
                        grade = a_proof[view]["deployment"]["landscape"]["methods"][c["method"]]
                        require(grade["arrays"]["sha256"] == native_id and pop[(view, c["method"])]["high_error_pixels"] == str(int(high.sum())),
                                "published A results differ from native endpoint rows")
                        rank = a["risk"]
                        state = np.full(rank.shape, 255, np.uint8)
                        state[a["valid"]] = (rank[a["valid"]].astype(np.float64) >= c["threshold"]).astype(np.uint8)
                        require(np.array_equal(state, a["state"]), "threshold state differs")
                    else:
                        r = b_ranks["cells"][c["id"]]
                        comp = b_proof["mechanism"][c["id"]]
                        require(r["sha256"] == native_id and r["queries"] == c["query_ids"] and b_proof["errors"] == int(high.sum()),
                                "native B budget or independent reference changed")
                        rank = a["rank_B-CPU-LR"]
                        comp_full = full["comparable"].astype(bool)
                        state = np.full(rank.shape, 2, np.uint8)
                        state[comp_full] = 1
                        state[a["comparable"] > 0] = 5
                        state[a["incomplete"] | full["incomplete"]] = 3
                        state[a["invalid_geometry"] | full["invalid_geometry"]] = 4
                        require(int((state == 1).sum()) == comp["selected_gap"] == c["counts"]["selected_gap"]
                                and int((state == 2).sum()) == comp["library_no_comparable"], "B scope differs")
                    require(c["target_prediction_sha256"] == sha(raw) and int(np.isfinite(rank).sum()) == c["counts"]["finite"],
                            "condition has a different fixed prediction or rank population")
                    compare_image(continuous(rank, c["legends"]["rank"]), pngs[c["rank"]["path"]])
                    compare_image(categorical(state, c["legends"]["state"]), pngs[c["state"]["path"]])
                    compare_image(categorical(reason, c["legends"]["reason"]), pngs[c["reason"]["path"]])
                    checks.append(c["id"])
                    stats["rank_state_reason_planes_bitwise_exact"] += 3
                rows.append(dict(case=case["id"], target=view, checks=checks, reference=int(finite.sum()),
                                 errors=int(high.sum()), prediction_sha256=v["target_prediction_sha256"]))
    save(out / "readback.json", dict(schema="el-foundation-case-independent-readback-1", parents=pins,
         passed=True, case_index_sha256=pins["case"]["sha256"], application="public positive whitelist and exact native numerical arrays",
         archive_sha256=archive["sha256"], archive_data_members=len(external), local_pngs=len(files), **dict(stats),
         rows=rows, native_view_conditions=sum(len(r["checks"]) for r in rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "verify"))
    for n in ("store", "config", "out"):
        parser.add_argument("--" + n, type=Path, required=n == "store")
    args = parser.parse_args()
    if args.mode == "record":
        print(json.dumps(register(args.store.resolve()), indent=2))
    else:
        verify(args.store, args.config, args.out)
