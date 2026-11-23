"""Disposable sealed examples of the eight-candidate and paired development-cost contracts."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from evidence_lens.formal_inputs import signature
from evidence_lens.scoring_io import file_hash, write_json
from evidence_lens.stage2_authority import (
    METHODS,
    NEURAL,
    SEEDS,
    RATES,
    TARGETS,
    WORKER_SOURCE_SHA,
    external_cost_fields,
    device_difference,
)
from test_vision_data import seal_fixture


def gate_fixture(tmp, names=None, cost_fault=None):
    store = tmp / "store"
    shelf = tmp / "cold-shelf"
    tmp.mkdir(exist_ok=True, parents=True)
    serial = iter(range(1000, 4000))
    parents = []

    def rid():
        return "20000101T000000000000Z-" + f"{next(serial):016x}"

    def file(stem, body):
        p = tmp / (rid() + "-" + stem)
        write_json(p, body)
        return p

    def seal(kind, files, meta=None, ident=None):
        pin = seal_fixture(shelf, ident or rid(), kind, "succeeded", meta or {}, files)
        parents.append(pin)
        return pin

    def item(pin, n):
        return shelf / "runs" / pin["run_id"] / n

    def spec(pin, filenames):
        return [{"path": n, **signature(item(pin, n))} for n in filenames]

    names = names or [f"syntheticnew{i}" for i in range(8)]
    devnames = [f"fixture-development-{i}" for i in range(8)]
    train_names = [f"synthetic-train-{n:02}" for n in range(28)]
    catalog = file(
        "catalog.json",
        {
            "schema": "el-vision-family-roster-1",
            "families": [
                {
                    "scene_family_id": f,
                    "scene_id": f + "/fixture-v1",
                    "test_group": "synthetic-familiar" if i < 4 else "synthetic-novel",
                    "split": "test",
                    "trials": {c: f + "-" + c for c in ("B0", "R1", "Q1")},
                }
                for i, f in enumerate(names)
            ]
            + [
                {"scene_family_id": f, "split": "development", "trials": {"B0": f + "-B0"}}
                for f in devnames[:4]
            ]
            + [{"scene_family_id": f, "split": "train"} for f in train_names[:12]],
        },
    )
    original_obj = {
        "families": [
            {"scene_family_id": f, "split": role}
            for role, ids in (
                ("train", train_names[12:]),
                ("development", devnames[4:]),
                ("test", ["syntheticpast" + str(n) for n in range(4)]),
            )
            for f in ids
        ]
    }
    original_file = file("original-catalog.json", original_obj)
    catsha = file_hash(catalog)
    study = file("study.json", {"catalog_sha256": catsha})
    from evidence_lens import stage2_parity

    parity_obj = {
        "schema": "el-stage2-cpu-parity-protocol-1",
        "training_protocol_sha256": file_hash(study),
        "finite_support_must_match": True,
        "minimum_common_finite_per_view": 1,
        "fp32_max_absolute_score_difference": 0.0001,
        "test_scoring_device": "macOS CPU",
    }
    parity_file = file("parity.json", parity_obj)
    fam = devnames[0]
    identity = {"scene_family_id": fam, "split": "development", "condition_id": "B0"}
    ref = seal(
        "scene_render",
        {
            "payload/input.png": {"synthetic": "input"},
            "payload/cameras.json": {"views": {}, "synthetic": "cameras"},
        },
        identity,
    )
    img = file_hash(item(ref, "payload/input.png"))
    cam = file_hash(item(ref, "payload/cameras.json"))
    rr = rid()
    pred = {
        "producer_run_id": rr,
        "input_sha256": img,
        "camera_sha256": cam,
        "views": {v: {"sha256": hashlib.sha256(v.encode()).hexdigest()} for v in TARGETS},
    }
    render = seal("render", {"payload/prediction.json": pred}, identity, rr)
    lookup_ref = file_hash(item(render, "payload/prediction.json"))
    dev_selection = store / "trials" / (fam + "-B0") / "trial_selection.json"
    dev_selection.parent.mkdir(parents=True)
    write_json(
        dev_selection,
        {
            "scene_family_id": fam,
            "trial_id": fam + "-B0",
            "condition_id": "B0",
            "selection_status": "selected",
            "selected": {"reference": ref, "render": render},
        },
    )
    views = [
        {
            "rgb": {"sha256": img},
            "full": {"deploy": {"sha256": img}},
            "family": f,
            "trial_id": f + "-B0",
            "status": "scored",
            "condition": "B0",
            "view": v,
            "height": 512,
            "width": 512,
            "source_input_sha256": img if f == fam else "a" * 64,
            "selection_sha256": file_hash(dev_selection) if f == fam else "0" * 64,
        }
        for f in devnames
        for v in TARGETS
    ]
    dr = seal(
        "dataset_build",
        {"payload/dataset.json": {"schema": "el-vision-dataset-1", "split": "development", "views": views}},
    )
    dh = file_hash(item(dr, "payload/dataset.json"))
    train_obj = {
        "schema": "el-vision-dataset-1",
        "split": "train",
        "views": [
            {
                "family": f,
                "condition": "B0",
                "view": v,
                "source_input_sha256": hashlib.sha256(f.encode()).hexdigest(),
            }
            for f in train_names
            for v in TARGETS
        ],
    }
    tr = seal("dataset_build", {"payload/dataset.json": train_obj})
    tr_sha = file_hash(item(tr, "payload/dataset.json"))
    split_audit = stage2_parity.split_families(
        train_obj, {"split": "development", "views": views}, json.loads(catalog.read_bytes()), original_obj
    )
    vals = {m: 0.5 for m in METHODS}
    thresholds = {m: {"tau": 0.5, "domain": "development-B0", "coverage": 0.75} for m in METHODS}
    raw = tmp / (rid() + "-synthetic-native-scores.npz")
    arr = np.ones((512, 512), np.float32)
    valid = np.ones(arr.shape, bool)
    state = np.ones(arr.shape, np.uint8)
    np.savez_compressed(raw, risk=arr, valid=valid, state=state)
    score_sig = signature(raw)
    train = []
    allrows = []
    models = {}
    chosen = {}
    selected_scores = {}
    pre = file("pre.json", {"synthetic": "preprocessing"})
    base_pin = file("base.json", {"synthetic": "base"})
    heads = {s: file("bounded-head.safetensors", {"synthetic_head": s}) for s in SEEDS}
    origin = {
        "training_dataset_sha256": tr_sha,
        "development_dataset_sha256": dh,
        "preprocessing_sha256": file_hash(pre),
        "base_pin_sha256": file_hash(base_pin),
        "source_sha256": {"worker-source.json": WORKER_SOURCE_SHA},
    }
    for m in NEURAL:
        rates = sorted(RATES[m])
        for j, (seed, rate) in enumerate(
            [(1729, rates[0]), (1729, rates[1]), (2718, rates[0]), (3141, rates[0])]
        ):
            id_ = rid()
            package = {}
            for n in ("weights.safetensors", "architecture.json", "base.json", "preprocessing.json"):
                package["payload/package/" + n] = file(
                    n, {"synthetic": True, "branch": m, "seed": seed, "rate": rate, "name": n}
                )
            package["payload/package/preprocessing.json"] = pre
            package["payload/package/base.json"] = base_pin
            index = {
                **origin,
                "schema": "el-vision-model-1",
                "model_id": id_,
                "phase": "train",
                "branch": m,
                "seed": seed,
                "threshold": 0.5,
                "development_dataset_sha256": dh,
                "training_dataset_sha256": tr_sha,
                "files": {Path(n).name: signature(p) for n, p in package.items()},
            }
            ip = file("model.json", index)
            package["payload/package/model.json"] = ip
            dev = file(
                "development.json",
                {
                    "family_equal_ap": 0.66,
                    "threshold": {"tau": 0.5, "domain": "development-B0", "coverage": 0.75},
                },
            )
            devrows = [
                {
                    "family": v["family"],
                    "condition": "B0",
                    "view": v["view"],
                    "rgb_sha256": img,
                    "deploy_sha256": img,
                    "status": "scored",
                    "reason": None,
                    "score": {"path": v["family"] + "/B0/" + v["view"] + ".npz", **score_sig},
                }
                for v in views
            ]
            scores = file(
                "producer-scores.json",
                {
                    "schema": "el-vision-scores-1",
                    "method": m,
                    "dataset_sha256": dh,
                    "model_sha256": file_hash(ip),
                    "selected_epoch": 1,
                    "device": "cuda",
                    "threshold": 0.5,
                    "views": devrows,
                },
            )
            grids = {"payload/development-scores/" + r["score"]["path"]: raw for r in devrows}
            curve = file(
                "curves.json", [{"epoch": i, "development": {"family_equal_ap": 0.66}} for i in (1, 2)]
            )
            h, t = stage2_parity.registered_groups(m)
            g = {"projection_head": h, **({"pretrained_suffix": t} if t else {})}
            change_obj = {
                "frozen_before": {"backbone.cls_token": "8" * 64},
                "frozen_after": {"backbone.cls_token": "8" * 64},
                "trainable": dict.fromkeys(h | t),
            }
            setup_obj = {
                "initial_tensors": dict.fromkeys(h | t | {"backbone.cls_token"}),
                "optimizer": {"groups": [{"role": k, "names": sorted(v)} for k, v in g.items()]},
            }
            change_file = file("parameter-changes.json", change_obj)
            setup_file = file("setup.json", setup_obj)
            names_manifest = stage2_parity.trainable_parameters(m, change_obj, setup_obj)
            training = {
                **origin,
                "schema": "el-vision-training-1",
                "phase": "train",
                "status": "completed",
                "seed": seed,
                "branch": m,
                "run_id": id_,
                "best_epoch": 1,
                "epochs": 2,
                "optimizer_steps": 2,
                "initial_head_sha256": file_hash(heads[seed]),
                "paired_design_sha256": "7" * 64,
                "source_sha256": {"worker-source.json": WORKER_SOURCE_SHA},
                "threshold": 0.5,
                "training_dataset_sha256": tr_sha,
                "development_dataset_sha256": dh,
                "head_lr": rate[1],
                "backbone_lr": rate[0],
                "best_development_family_equal_ap": 0.66,
                "wall_seconds": 2.0,
                "peak_cuda_allocated": 0,
                "peak_cuda_reserved": 0,
                "files": {
                    "curves.json": signature(curve),
                    "setup.json": signature(setup_file),
                    "parameter-changes.json": signature(change_file),
                    "development.json": signature(dev),
                    "development-scores/scores.json": signature(scores),
                },
                "package": {"path": "package/model.json", **signature(ip)},
            }
            run = seal(
                "model_fit",
                {
                    **package,
                    **grids,
                    "payload/curves.json": curve,
                    "payload/setup.json": setup_file,
                    "payload/parameter-changes.json": change_file,
                    "payload/initial-head.safetensors": heads[seed],
                    "payload/development-scores/scores.json": scores,
                    "payload/development.json": dev,
                    "payload/training.json": training,
                },
                {"experiment_id": "stage2-vision-train"},
                id_,
            )
            fields = {
                **run,
                **{
                    k: training[k]
                    for k in (
                        "seed",
                        "head_lr",
                        "backbone_lr",
                        "best_epoch",
                        "best_development_family_equal_ap",
                        "threshold",
                    )
                },
                "model_sha256": file_hash(ip),
            }
            allrows.append({**fields, "branch": m})
            compact = {
                k: v for k, v in fields.items() if k not in ("best_development_family_equal_ap", "threshold")
            }
            train.append(
                {
                    **compact,
                    "method": m,
                    "trainable_parameters": names_manifest,
                    "weights_sha256": index["files"]["weights.safetensors"]["sha256"],
                    "training_json_sha256": file_hash(item(run, "payload/training.json")),
                    "development_json_sha256": file_hash(dev),
                }
            )
            if j == 0:
                models[m] = {**run, "model_sha256": file_hash(ip), "files": spec(run, package)}
                chosen[m] = fields
                selected_scores[m] = file_hash(scores)
    lid = rid()
    loc = {
        n: file(n, {"synthetic": True, "name": n})
        for n in ("L1-full.json", "L1-geometry.json", "feature_schema.json")
    }
    loc["thresholds.json"] = file(
        "lr-thresholds.json",
        {
            "threshold_run_id": lid,
            "thresholds": {m: 0.5 for m in ("L1-full", "L1-geometry", "constant", "1-alpha")},
        },
    )
    lidx = {
        "status": "frozen",
        "model_id": lid,
        "threshold_run_id": lid,
        "files": {n: file_hash(p) for n, p in loc.items()},
    }
    loc["model.json"] = file("lr-index.json", lidx)
    lr = seal("threshold_select", {"payload/model/" + n: pt for n, pt in loc.items()}, ident=lid)
    for m in METHODS[3:5]:
        models[m] = {
            **lr,
            "model_sha256": file_hash(loc["model.json"]),
            "files": spec(
                lr,
                [
                    "payload/model/" + n
                    for n in (m + ".json", "model.json", "thresholds.json", "feature_schema.json")
                ],
            ),
        }
    th = seal(
        "threshold_select",
        {
            "payload/thresholds.json": {
                "schema": "el-vision-thresholds-1",
                "primary_seed": 1729,
                "development_dataset_sha256": dh,
                "thresholds": vals,
                "choices": thresholds,
                "method_sources": {
                    tag: {
                        "model_sha256": models[m]["model_sha256"],
                        "sha256": selected_scores.get(m, "5" * 64),
                    }
                    for tag, m in (
                        ("original-five", "L1-full"),
                        (NEURAL[0], NEURAL[0]),
                        (NEURAL[1], NEURAL[1]),
                    )
                },
            }
        },
    )
    costs = {}
    for m in NEURAL:
        w = models[m]
        runpin = {k: w[k] for k in ("run_id", "manifest_sha256")}
        data = {}
        native = []
        for k in range(4):
            for v in TARGETS:
                data["payload/" + ("cold" if k == 0 else "warm-" + str(k)) + "/" + v + ".npz"] = raw
                native.append(
                    {
                        "pass": k,
                        "view": v,
                        "height": 512,
                        "width": 512,
                        "status": "scored",
                        "supported_pixels": 512**2,
                        "wall_through_npz_seconds": 1.0,
                        "inputs": {
                            key: {"sha256": h}
                            for key, h in (
                                ("input_png", img),
                                ("cameras", cam),
                                ("prediction_metadata", lookup_ref),
                                ("prediction/" + v, pred["views"][v]["sha256"]),
                            )
                        },
                        "producer_run_id": render["run_id"],
                        "output_sha256": file_hash(raw),
                    }
                )
        native[0]["fresh_process_to_first_output_seconds"] = 2.0
        cr = {
            "schema": "el-vision-native-cost-1",
            "device": "cpu",
            "method": m,
            "platform": "macOS-fixture",
            "model_id": w["run_id"],
            "model_sha256": w["model_sha256"],
            "all_measured_views_512": True,
            "all_measured_views_scored": True,
            "rss_interval_seconds": 0.01,
            "warm_views": 18,
            "views": native,
            "hot_p50_seconds": 1.0,
            "hot_p95_seconds": 1.0,
            "cold_process_to_first_output_seconds": 2.0,
            "process_tree_peak_rss_bytes": 100,
        }
        costd = file("native-cost.json", cr)
        roles = []
        process = {
            "schema": "el-vision-external-process-cost-1",
            "model_id": w["run_id"],
            "model_sha256": w["model_sha256"],
            "runtime_report_sha256": file_hash(costd),
            "first_output_sha256": score_sig["sha256"],
            "launch_to_first_complete_npz_seconds": 2.5,
            "process_tree_peak_rss_bytes": 115,
            "sample_interval_seconds": 0.01,
        }
        normal = copy.deepcopy(process)
        if m == NEURAL[0] and cost_fault == "swapped_external":
            process["runtime_report_sha256"] = "f" * 64
        if m == NEURAL[0] and cost_fault == "wrong_raster":
            bad = tmp / "false-native.npz"
            np.savez_compressed(bad, risk=arr[:2, :2], valid=valid[:2, :2], state=state[:2, :2])
            data["payload/warm-2/v04.npz"] = bad
            native[15]["output_sha256"] = file_hash(bad)
            costd = file("native-cost.json", cr)
            process["runtime_report_sha256"] = normal["runtime_report_sha256"] = file_hash(costd)
        external = file("process-cost.json", process)
        boundary = external_cost_fields(
            cr, file_hash(costd), normal, file_hash(external), score_sig["sha256"]
        )
        for k, pin, src in (
            ("model", runpin, "payload/package/model.json"),
            ("input_image", ref, "payload/input.png"),
            ("cameras", ref, "payload/cameras.json"),
            ("prediction", render, "payload/prediction.json"),
        ):
            member = "inputs/" + k + "/" + Path(src).name
            data[member] = item(pin, src)
            roles.append(
                {
                    "role": k,
                    "parent_run_id": pin["run_id"],
                    "storage": "embedded",
                    "kind": "file",
                    "path": member,
                    "files": [{"path": member, **signature(data[member])}],
                }
            )
        cp = seal(
            "model_score",
            {
                **data,
                "payload/native-cost.json": costd,
                "payload/process-cost.json": external,
                "inputs.json": {"parents": [runpin, ref, render], "inputs": roles},
            },
        )
        costs[m] = {
            "pin": cp,
            "native_cost_sha256": file_hash(costd),
            "model_sha256": w["model_sha256"],
            "family": fam,
            **boundary,
            "hot_p50_seconds": 1.0,
            "hot_p95_seconds": 1.0,
            "canonical_win_to_mac": {v: device_difference(arr, arr, 0.5) for v in TARGETS},
        }
    replay = file(
        "replay.json",
        {
            "paired": {str(s): {m: {"value": 0.66} for m in ("L1-full", *NEURAL)} for s in SEEDS},
            "primary_domains": {m: {"own": {"macro": {"value": 0.66}}} for m in METHODS},
        },
    )
    replay_data = json.loads(replay.read_bytes())
    regions = [
        {
            "family": v["family"],
            "view": v["view"],
            "S7": 512**2,
            "S7Y": 0,
            "mask_sha256": "6" * 64,
            "candidate_support_sha256": "7" * 64,
        }
        for v in views
    ]
    candidates = {
        r["run_id"]: {
            "branch": r["branch"],
            "seed": r["seed"],
            "ap": {"value": 0.66},
            "head_lr": r["head_lr"],
            "backbone_lr": r["backbone_lr"],
        }
        for r in allrows
    }
    independent = stage2_parity.independent_choice(candidates, {"chosen": chosen})
    replay = file(
        "replay-complete.json",
        {
            **replay_data,
            "paired_domain": {str(s): regions for s in SEEDS},
            "candidates": candidates,
            "independent_winners": independent,
            "matched_48_three_seed_domains": 48,
        },
    )
    selection = file(
        "selection.json",
        {
            **origin,
            "schema": "el-vision-primary-selection-1",
            "seed": 1729,
            "chosen": chosen,
            "all_candidates": allrows,
        },
    )
    roster = {
        "schema": "el-vision-postfit-roster-1",
        "inference_parity_protocol_sha256": file_hash(parity_file),
        "original_catalog_sha256": file_hash(original_file),
        "candidate_runs": [{k: r[k] for k in ("run_id", "manifest_sha256")} for r in train],
        "catalog_sha256": catsha,
        "study_sha256": file_hash(study),
        "stages": {"lr_export": lr, "thresholds": th, "dev_data": dr, "train_data": tr},
        "cost_family": fam,
        "cpu_costs": {m: c["pin"] for m, c in costs.items()},
    }
    method = {
        "schema": "el-vision-method-freeze-1",
        "inference_parity_protocol_sha256": file_hash(parity_file),
        "original_catalog_sha256": file_hash(original_file),
        "native_inference_parity": stage2_parity.parity(costs, parity_obj),
        "learning_split_audit": split_audit,
        "catalog_sha256": catsha,
        "study_sha256": file_hash(study),
        "formal_worker_index_sha256": WORKER_SOURCE_SHA,
        "test_family_ids": names,
        "methods": list(METHODS),
        "models": models,
        "training_receipts": train,
        "display_rule": "v01",
        "seeds": list(SEEDS),
        "primary_seed": 1729,
        "thresholds": thresholds,
        "train_dataset_sha256": tr_sha,
        "development_dataset_sha256": dh,
        "selection_sha256": file_hash(selection),
        "thresholds_sha256": file_hash(item(th, "payload/thresholds.json")),
        "development_replay_sha256": file_hash(replay),
        "cpu_costs": costs,
        "baseline_development_qa": {"synthetic": True},
        "deployment_rule": {
            "default": "L1-full",
            "primary_seed": 1729,
            "paired_ft_minus_frozen": [0.0] * 3,
            "mean_difference": 0.0,
            "sample_sd_difference": 0.0,
            "checks": {
                "three_paired_seeds": True,
                "ft_exceeds_lr_every_seed": False,
                "paired_mean_exceeds_sample_sd": False,
                "hot_p95_at_most_5s": True,
                "rss_at_most_2gib": True,
            },
        },
    }

    def freeze(mutate=None):
        d = copy.deepcopy(method)
        ro = copy.deepcopy(roster)
        if mutate:
            mutate(d, ro)
        rp = file("roster.json", ro)
        d["input_roster_sha256"] = file_hash(rp)
        template = file("method.json", d)
        cfg = file(
            "config.json",
            {
                "experiment_id": "stage2-method-freeze",
                "trial_id": "disposable-synthetic-method",
                "dataset_hash": catsha,
            },
        )
        source = tmp / "fixture-source"
        (source / "scripts").mkdir(exist_ok=True, parents=True)
        (source / "configs").mkdir(exist_ok=True)
        write_json(
            source / "configs/storage.json",
            {"local_free_floor_bytes": 0, "local_active_batch_target_bytes": 10**9},
        )
        worker = source / "scripts/freeze.py"
        worker.write_text(
            "import json,os,shutil,sys\nfrom pathlib import Path\nout=Path(sys.argv[1]); d=json.loads(Path(sys.argv[2]).read_bytes()); d['run_id']=os.environ['EL_RUN_ID']; (out/'method-freeze.json').write_text(json.dumps(d)); shutil.copyfile(sys.argv[3],out/'selection.json'); shutil.copyfile(sys.argv[4],out/'development-replay.json')\n"
        )
        cmd = [
            sys.executable,
            "-m",
            "evidence_lens.recorder",
            "--root",
            str(store),
            "--source-root",
            str(source),
            "--kind",
            "report",
            "--config",
            str(cfg),
        ]
        for k, fp in (
            ("method_roster", rp),
            ("catalog", catalog),
            ("study", study),
            ("parity", parity_file),
            ("original_catalog", original_file),
            ("fixture_template", template),
            ("fixture_selection", selection),
            ("fixture_replay", replay),
        ):
            cmd += ["--input", k + "=" + str(fp)]
        for p in parents:
            cmd += ["--parent", p["run_id"] + "=" + p["manifest_sha256"]]
        cmd += [
            "--",
            "{python}",
            "{source}/scripts/freeze.py",
            "{out}",
            "{input:fixture_template}",
            "{input:fixture_selection}",
            "{input:fixture_replay}",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, res.stderr + res.stdout
        actual = json.loads(res.stdout.strip().splitlines()[-1])
        pin = {k: actual[k] for k in ("run_id", "manifest_sha256")}
        folder = store / "runs" / pin["run_id"]
        return (
            pin,
            json.loads((folder / "payload/method-freeze.json").read_bytes()),
            file_hash(folder / "payload/method-freeze.json"),
        )

    return {
        "store": store,
        "shelf": shelf,
        "catalog": catalog,
        "catalog_sha": catsha,
        "names": names,
        "freeze": freeze,
        "models": models,
        "dev_data": item(dr, "payload/dataset.json"),
        "values": vals,
    }
