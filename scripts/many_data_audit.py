"""Independently verify source-package pixels, nested libraries, camera conventions and held-out roles."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import stat
import sys
import time
import zipfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dual_records import selected, stage, resolve
from dtu_eval import read_pinned
from evidence_lens._bundle import load, save
from evidence_lens._verify import require

SIZE = (378, 252)


def stage_names(spec):
    names = {
        "registration": "b-data-registration",
        "study": "b-study-registration",
        "devopening": "b-train-dev-h-opening",
    }
    for i, row in enumerate(spec["originals"] + spec["external"]["roster"]):
        slug = row["id"].replace("_", "-")
        names[f"source{i}"] = "b-source-" + slug
        if "url" in row:
            names[f"download{i}"] = "b-eth-download-" + slug
            names[f"roster{i}"] = "b-roles-" + slug
        if row["split"] != "confirmation":
            names[f"truth{i}"] = "b-truth-" + slug
    return names


def audit_stage(store):
    store = store.resolve()
    spec = load(read_pinned((store,), selected(store, "b-data-registration"))[1])
    parents = {key: selected(store, name) for key, name in stage_names(spec).items()}
    return stage(
        store,
        "b-data-preopening-audit",
        "archive_verify",
        parents,
        {
            "phase": "R-only cross-check with prior training/development H receipts",
            "analysis_size": list(SIZE),
            "pixel_recheck": "lexically first R in every scene",
        },
        [
            "{python}",
            "-B",
            "{source}/scripts/many_data_audit.py",
            "worker",
            "--store",
            str(store),
            "--config",
            "{config}",
            "--out",
            "{out}",
        ],
        "payload/data.json",
        300000,
    )


def verify_zip_roster(z, expected):
    names = z.namelist()
    require(
        len(names) == len(set(names)) and set(names) == set(expected),
        "ZIP members differ from the explicit role roster",
    )
    require(
        all(not n.startswith(("/", "\\")) and ".." not in Path(n).parts for n in names),
        "unexpected archive member name",
    )


def zip_bytes(z, ref):
    raw = z.read(ref["path"])
    require(
        len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"],
        "wrong image member size or digest",
    )
    return raw


def package_path(owner, ref):
    rel = "payload/" + ref["path"]
    entry = owner.files[rel]
    require(
        entry["bytes"] == ref["bytes"] and entry["sha256"] == ref["sha256"],
        "different recorder member identity",
    )
    return owner.open(rel)


def camera_geometry(cam, grid):
    width, height = cam["original_size"]
    crop_w = min(width, math.floor(height * SIZE[0] / SIZE[1] + 0.5))
    crop_h = min(height, math.floor(width * SIZE[1] / SIZE[0] + 0.5))
    left, top = (width - crop_w) // 2, (height - crop_h) // 2
    require(grid["crop_xywh"] == [left, top, crop_w, crop_h], "different integer principal crop")
    expected = np.array(cam["original_K"], dtype=np.float64)
    for k, start, length in ((0, left, crop_w), (1, top, crop_h)):
        expected[k, k] *= SIZE[k] / length
        expected[k, 2] = (expected[k, 2] - start + 0.5) * SIZE[k] / length - 0.5
    require(
        np.allclose(expected, cam["K"], rtol=0.0, atol=1e-10),
        "different integer-centre projected calibration",
    )
    trans = np.array(cam["c2w"])
    require(
        trans.shape == (4, 4)
        and np.isfinite(trans).all()
        and np.array_equal(trans[3], [0, 0, 0, 1])
        and np.allclose(trans[:3, :3].T @ trans[:3, :3], np.eye(3), rtol=0, atol=1e-9)
        and abs(np.linalg.det(trans[:3, :3]) - 1) < 1e-9,
        "invalid right-handed rigid camera",
    )
    return [left, top, crop_w, crop_h]


def compare_pixels(raw, analysis, cam, grid):
    left, top, w, h = camera_geometry(cam, grid)
    with Image.open(io.BytesIO(raw)) as src, Image.open(io.BytesIO(analysis)) as result:
        require(
            src.mode == "RGB"
            and list(src.size) == cam["original_size"]
            and result.mode == "RGB"
            and result.size == SIZE,
            "unexpected physical image dimensions or modes",
        )
        expected = src.crop((left, top, left + w, top + h)).resize(SIZE, Image.Resampling.LANCZOS)
        require(
            expected.tobytes() == result.tobytes(), "original R to saved analysis RGB is not bitwise exact"
        )


def check_scene(row, src, owner):
    orig = "seed" in row
    count = 128 if orig else row["R"]
    R = [f"r{i:03d}" for i in range(count)]
    H = [f"h{i:02d}" for i in range(6 if orig else row["H"])]
    require(
        src["R"] == R
        and src["H"] == H
        and src["first_access_role"] == "R"
        and src["image_decode_calls"] == len(R)
        and src["H_photo_decodes"]
        == src["source_truth_depth_members_sent"]
        == src["H_data_members_sent"]
        == 0,
        "source role does not match registration",
    )
    originals = src["originals"]
    require(
        set(originals) == set(R) and len({ref["path"] for ref in originals.values()}) == len(R),
        "wrong originals manifest",
    )
    with zipfile.ZipFile(package_path(owner, src["original_supplement"])) as raw_zip:
        verify_zip_roster(raw_zip, [originals[v]["path"] for v in R])
        for v in R:
            raw = zip_bytes(raw_zip, originals[v])
            if v == R[0]:
                first = raw
    subsets = {f"R{count}": R}
    if orig:
        subsets["R16"] = R[::8]
    require(set(src["packages"]) == set(subsets), "different source configurations")
    documents = {}
    summary = {}
    for name, subset in subsets.items():
        ref = src["packages"][name]
        with zipfile.ZipFile(package_path(owner, ref)) as deployment:
            doc = json.loads(deployment.read("scene.json"))
            documents[name] = doc
            ids = subset + H
            require(
                doc["schema"] == "el-many-source-data-1"
                and doc["scene_id"] == row["id"]
                and doc["split"] == row["split"]
                and doc["R"] == doc["A"] == subset
                and doc["H"] == H
                and set(doc["views"]) == set(ids)
                and doc["target_pixels_in_package"]
                == doc["geometry_prior_in_package"]
                == doc["point_tracks_in_package"]
                == 0,
                "different source/target/geometry identity",
            )
            require(
                set(doc["source_rgb"]) == set(subset)
                and doc["original_rgb"] == {v: originals[v] for v in subset}
                and doc["permission"]["R"] == doc["permission"]["A"] == subset
                and doc["permission"]["H"] == H,
                "different permission sets or original sources",
            )
            require(
                doc["grids"]["analysis_size"] == list(SIZE)
                and doc["unit"] == ("m" if orig else "scene_units")
                and doc["frame_receipt"]["world_to_deployment"] == np.eye(4).tolist(),
                "different analysis grid or frame",
            )
            verify_zip_roster(deployment, ["scene.json"] + [f"analysis/{v}.png" for v in subset])
            for v in ids:
                cam = doc["views"][v]
                g = doc["grids"]["views"][v]
                require(
                    cam["K"] == g["analysis_K"]
                    and cam["original_K"] == g["original_K"]
                    and [cam["width"], cam["height"]] == list(SIZE),
                    "different declared analysis matrix",
                )
                camera_geometry(cam, g)
                if not orig:
                    native = cam["official_corner_principal_point"]
                    require(
                        [native[0] - 0.5, native[1] - 0.5]
                        == [cam["original_K"][0][2], cam["original_K"][1][2]],
                        "different ETH3D corner to centre transform",
                    )
            for v in subset:
                ref_img = doc["source_rgb"][v]
                require(ref_img["path"] == f"analysis/{v}.png", "different analysis image identifier")
                pixels = zip_bytes(deployment, ref_img)
                if v == R[0] and len(subset) == count:
                    compare_pixels(first, pixels, doc["views"][v], doc["grids"]["views"][v])
        summary[name] = {**ref, "source_members": len(subset), "camera_only_targets": len(H)}
    large = documents[f"R{count}"]
    if orig:
        small = documents["R16"]
        require(
            small["frame_receipt"] == large["frame_receipt"] and small["H"] == large["H"],
            "nested frame or H mismatch",
        )
        for v in small["R"] + H:
            require(
                small["views"][v] == large["views"][v]
                and small["grids"]["views"][v] == large["grids"]["views"][v],
                "different shared R16/R128 camera",
            )
        for v in small["R"]:
            require(small["source_rgb"][v] == large["source_rgb"][v], "different shared R16/R128 RGB bytes")
    return {
        "scene_id": row["id"],
        "split": row["split"],
        "unit": large["unit"],
        "R_unique": len(R),
        "H_cameras": len(H),
        "source_members_across_configurations": sum(map(len, subsets.values())),
        "configuration_archives": summary,
        "raw_R_supplement": src["original_supplement"],
        "R_photographs_with_original_sha256_verified": len(R),
        "analysis_members_with_sha256_verified": sum(map(len, subsets.values())),
        "original_to_analysis_bitwise_probe": R[0],
        "original_probe_size": large["views"][R[0]]["original_size"],
        "source_original_bytes_decoded": src["decoded_original_bytes"],
        "source_original_decode_seconds": src["image_decode_seconds"],
        "source_original_first_read_timestamp_granularity": "native run bounds; per-image cumulative calls and time",
        "geometry_prior_members": 0,
        "target_photo_members": 0,
        "nested_16_128_shared_pixels_and_cameras_exact": True if orig else None,
    }


def check_eth(roster, row):
    by_name = sorted(roster["views"].items(), key=lambda t: t[1]["original_name"].encode("utf-8"))
    n = len(by_name)
    held = [int((j + 0.5) * n / 10) for j in range(10)]
    require(
        n == row["official_total"]
        and sorted(roster["H"]) == [by_name[i][0] for i in held]
        and roster["R"] == [v for i, (v, _) in enumerate(by_name) if i not in held]
        and roster["points3D_members_opened"]
        == roster["keypoint_or_track_rows_parsed"]
        == roster["H_photo_members_opened"]
        == 0,
        "official photo rank or camera-only input role differs from registration",
    )
    return {
        "official_count": n,
        "heldout_zero_based_lexical_ranks": held,
        "heldout_official_names": [by_name[i][1]["original_name"] for i in held],
        "source_and_heldout_calibrations_stay_in_official_world": True,
        "official_3d_points_opened": 0,
        "official_observation_rows_parsed": 0,
    }


def resource_row(label, owner, receipt, inodes):
    paths = list(owner.files) + ["manifest.json", "completion.json"]
    apparent = 0
    for rel in paths:
        s = (owner.root / rel).lstat()
        require(stat.S_ISREG(s.st_mode), "unexpected non-regular member in sealed run archive")
        inodes[(s.st_dev, s.st_ino)] = (s.st_size, s.st_blocks * 512)
        apparent += s.st_size
    with owner.open("telemetry.csv").open(newline="") as f:
        samples = list(csv.DictReader(f))
    sampled = 0.0
    covered = 0.0
    for prev, now in zip(samples, samples[1:]):
        if now["tree_cpu_percent"]:
            dt = float(now["monotonic_seconds"]) - float(prev["monotonic_seconds"])
            sampled += dt * float(now["tree_cpu_percent"]) / 100
            covered += dt
    worker = receipt.get("resource", {})
    process = worker.get("cpu_seconds", receipt.get("cpu_s"))
    child = worker.get("child_cpu_seconds", 0)
    return {
        "label": label,
        "run_id": owner.run["run_id"],
        "start_utc": owner.run["started_at_utc"],
        "end_utc": owner.run["ended_at_utc"],
        "native_wall_seconds": owner.run["elapsed_seconds_monotonic"],
        "sampled_tree_peak_rss_bytes": owner.run["sampled_peak_tree_rss_bytes"],
        "telemetry_interval_tree_cpu_seconds": sampled if covered else None,
        "telemetry_covered_interval_seconds": covered,
        "worker_process_plus_child_cpu_seconds": process + child if process is not None else None,
        "worker_wall_seconds": worker.get("wall_seconds", receipt.get("wall_s")),
        "manifest_file_bytes_plus_seal": apparent,
        "payload_bytes": sum(v["bytes"] for k, v in owner.files.items() if k.startswith("payload/")),
    }


def worker(a):
    require(os.environ.get("EL_RUN_ID"), "use native recorder")
    started, cpu = time.monotonic(), time.process_time()
    pins = load(a.config)["sources"]
    native = {}
    records = {}
    for key, pin in pins.items():
        native[key], path = read_pinned((a.store,), pin)
        records[key] = load(path)
    spec = records["registration"]
    names = stage_names(spec)
    require(set(records) == set(names), "missing native source-data inputs")
    scenes = []
    open_targets = records["devopening"]["allowed_targets"]
    scope = {}
    for i, row in enumerate(spec["originals"] + spec["external"]["roster"]):
        idx = f"source{i}"
        rec = records[idx]
        info = check_scene(row, rec, native[idx])
        info["source_receipt"] = pins[idx]
        if "seed" in row:
            info["seed"] = row["seed"]
        else:
            d = records[f"download{i}"]
            require(
                d["scene_id"] == row["id"]
                and d["url"] == row["url"]
                and d["individual_H_photo_opens"]
                == d["photograph_decodes"]
                == d["geometric_truth_reads"]
                == 0,
                "invalid opaque download",
            )
            ent = native[f"download{i}"].files["payload/" + d["file"]]
            require(
                ent["sha256"] == d["sha256"] and ent["bytes"] == d["bytes"],
                "wrong native opaque transport identity",
            )
            info.update(
                external=check_eth(records[f"roster{i}"], row),
                official_transport={
                    "url": d["url"],
                    "bytes": d["bytes"],
                    "sha256": d["sha256"],
                    "pin": pins[f"download{i}"],
                },
            )
        if row["split"] != "confirmation":
            truth = records[f"truth{i}"]
            scope[row["id"]] = rec["H"]
            require(
                truth["opening_sha256"] == pins["devopening"]["sha256"]
                and truth["H"] == rec["H"]
                and set(truth["photo_decode_events"]) == set(rec["H"])
                and truth["H_photo_decodes"] == len(rec["H"]),
                "invalid prior training/development target first reads",
            )
            checks = truth["independent_world_ray_checks"]
            require(
                set(checks) == (set(rec["R"] + rec["H"]) if "seed" in row else set()),
                "wrong authored geometry coverage",
            )
            info["previous_authorized_evaluation"] = {
                "truth_receipt": pins[f"truth{i}"],
                "H_first_decodes": len(truth["H"]),
                "mean_decode_seconds": truth["image_decode_seconds"] / len(truth["H"]),
                "H_first_observed_utc": min(
                    r["decode_start_utc"] for r in truth["photo_decode_events"].values()
                ),
                "H_last_decode_start_utc": max(
                    r["decode_start_utc"] for r in truth["photo_decode_events"].values()
                ),
                "geometry_images": len(checks),
                "independent_world_rays": sum(c["pixel_rays"] for c in checks.values()),
                "largest_observed_depth_probe_error_metres": max(
                    (c["max_absolute_z_error_m"] for c in checks.values()), default=None
                ),
            }
        else:
            info["previous_authorized_evaluation"] = None
        scenes.append(info)
    require(scope == open_targets, "training/development permission denominator differs")
    inodes = {}
    resources = [resource_row(names[k], native[k], records[k], inodes) for k in records if k != "study"]
    downloads = sum(r["official_transport"]["bytes"] for r in scenes if "external" in r)
    require(downloads <= spec["external"]["download_cap_bytes"], "download cap exceeded")
    evidence = [r["previous_authorized_evaluation"] for r in scenes if r["previous_authorized_evaluation"]]
    budget = {
        "download_response_bytes": downloads,
        "download_cap_bytes": spec["external"]["download_cap_bytes"],
        "production_native_runs_excluding_shared_study": len(resources),
        "zero_gpu_seconds": 0,
        "exact_worker_cpu_seconds_where_instrumented": sum(
            r["worker_process_plus_child_cpu_seconds"] or 0 for r in resources
        ),
        "native_runs_with_exact_worker_cpu": sum(
            r["worker_process_plus_child_cpu_seconds"] is not None for r in resources
        ),
        "sampled_tree_cpu_seconds_where_interval_available": sum(
            r["telemetry_interval_tree_cpu_seconds"] or 0 for r in resources
        ),
        "native_runs_with_sampled_interval": sum(
            r["telemetry_interval_tree_cpu_seconds"] is not None for r in resources
        ),
        "payload_bytes": sum(r["payload_bytes"] for r in resources),
        "same_inode_deduplicated_apparent_bytes": sum(x[0] for x in inodes.values()),
        "allocated_bytes_deduplicated_by_inode": sum(x[1] for x in inodes.values()),
        "peak_sampled_tree_rss_bytes": max(r["sampled_tree_peak_rss_bytes"] or 0 for r in resources),
        "archive_scope": "Sealed native files plus seals from the listed production stages. Same-inode hardlinks counted once. Filesystem clone/external sharing, directory metadata, the audit itself and other study roles are outside this measurement.",
        "cpu_scope": "Exact worker and sampled process-tree values are distinct overlapping measures. Sampled sums exclude unsampled edges and short-lived processes; stage clocks may overlap and are not summed as elapsed wall time.",
    }
    out = {
        "schema": "el-many-source-data-audit-1",
        "run_id": os.environ["EL_RUN_ID"],
        "A_BASELINE_ID": spec["A_BASELINE_ID"],
        "registered_public_exhibit": spec["public_display"],
        "source_registration": pins["registration"],
        "study_registration": pins["study"],
        "training_development_opening": pins["devopening"],
        "analysis_size": list(SIZE),
        "scenes": scenes,
        "production_resources": resources,
        "data_role_resources": budget,
        "totals": {
            "unique_R_original_production_decodes": sum(r["R_unique"] for r in scenes),
            "R_deployment_memberships": sum(r["source_members_across_configurations"] for r in scenes),
            "deployment_libraries": sum(len(r["configuration_archives"]) for r in scenes),
            "authorized_training_development_H_first_decodes": sum(r["H_first_decodes"] for r in evidence),
            "authored_optical_depth_images": sum(r["geometry_images"] for r in evidence),
            "world_ray_checks": sum(r["independent_world_rays"] for r in evidence),
            "confirmation_H_image_reads": 0,
            "QA_decoded_R_originals": len(scenes),
            "QA_decoded_R_analysis_PNGs": len(scenes),
            "QA_decoded_H_images": 0,
        },
        "audit_runtime": {
            "worker_cpu_seconds": time.process_time() - cpu,
            "worker_wall_seconds": time.monotonic() - started,
            "peak_process_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    save(a.out / "data.json", out)


def geometry_link(s):
    pins = {
        k: selected(s, label)
        for k, label in {
            "bank": "b-bank-b04-layered-louvers-r16",
            "truth": "b-truth-b04-layered-louvers",
            "physical": "b-exact-visibility-b04-layered-louvers",
            "opening": "b-train-dev-h-opening",
            "runtime": "b-score-runtime",
        }.items()
    }
    runtime = load(resolve((s,), pins["runtime"]))
    cmd = [
        str(s / runtime["relative_executable"]),
        "-I",
        "-B",
        "{source}/scripts/many_data_audit.py",
        "geometry-link-worker",
        "--store",
        str(s),
        "--config",
        "{config}",
        "--out",
        "{out}",
    ]
    return stage(
        s,
        "b-development-visibility-link",
        "archive_verify",
        pins,
        {
            "scope": "first authored development R16, h00, real Q4",
            "purpose": "check native acquisition/frame and distinct exact and sampled-depth evaluator inputs",
        },
        cmd,
        "payload/link.json",
        24000,
    )


def geometry_link_worker(a):
    from evidence_lens_many import Bank, select_sources, stream_features
    from many_data_access import EvaluatorTruth
    from many_data_visibility import VisibilityTruth
    from many_evaluate import exact_visibility, visibility

    require(os.environ.get("EL_RUN_ID"), "submit the development integration to the recorder")
    pins = load(a.config)["sources"]
    paths = {k: resolve((a.store,), p) for k, p in pins.items()}
    gate = load(paths["opening"])
    sid = "b04-layered-louvers"
    v = "h00"
    require(
        gate["phase"] == "train-development" and v in gate["allowed_targets"][sid],
        "development authorization missing",
    )
    bank = Bank(paths["bank"].parent, pins["bank"]["sha256"])
    truth = EvaluatorTruth(paths["truth"], pins["truth"]["sha256"], bank=bank)
    frame = truth.verify_bank(bank)
    require(
        truth.meta["scene_id"] == sid and truth.meta["opening_sha256"] == pins["opening"]["sha256"],
        "different development data",
    )
    probe = VisibilityTruth(paths["physical"], pins["physical"]["sha256"])
    probe.verify(truth)
    plan = select_sources(bank, v, strategy="coverage")
    (step,) = stream_features(bank, v, plan, budgets=(4,))
    cells = {"coverage-4": {"queries": list(step.queried_source_ids)}}
    geometry = {"coverage-4": {"comparable": step.comparable_count}}
    sampled = visibility(bank, truth, {}, truth.meta, v, cells, geometry)["coverage-4"]
    physical = exact_visibility(probe, v, cells, geometry)["coverage-4"]
    require(
        physical["probe_pixels"] == 1944
        and sampled["independent_target_surface"] > physical["independent_target_surface"],
        "sparse mesh and dense sampled depth counts refer to the wrong support",
    )
    for entry in (sampled, physical):
        require(
            sum(entry[k] for k in ("both", "only_true", "only_model", "neither"))
            == entry["independent_target_surface"],
            "comparison lost a foreground position",
        )
    save(
        a.out / "link.json",
        {
            "schema": "el-many-dev-native-geometry-link-1",
            "run_id": os.environ["EL_RUN_ID"],
            "sources": pins,
            "scene": sid,
            "R": 16,
            "H_view": v,
            "Q": cells["coverage-4"]["queries"],
            "verified_frame": frame,
            "true_surface_mesh_probe": physical,
            "dense_nearest_depth_approximation": sampled,
            "RGB_reference_decodes": 0,
            "calibration_or_model_fits": 0,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("record", "worker", "geometry-link", "geometry-link-worker"))
    p.add_argument("--store", required=True, type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.action == "record":
        print(json.dumps(audit_stage(args.store)), flush=True)
    elif args.action == "geometry-link":
        print(json.dumps(geometry_link(args.store.resolve())), flush=True)
    elif args.action == "geometry-link-worker":
        geometry_link_worker(args)
    else:
        worker(args)
