"""Record and resume the four-family learning pilot or the complete train/development Stage B baseline."""

import argparse
import json
from pathlib import Path

from evidence_lens._bundle import load
from evidence_lens.formal_inputs import compact_pin, require
from evidence_lens.scoring_io import file_hash
from vision_record import Recorder, parse_pin
from vision_sources import signed


# Immutable pointers for the already replayed original 16 training / 4 development family transports.
ORIGINAL = {
    "train": "20261123T121614857023Z-12dd70baa12449a4=2eaa2247469f4d65c0e60daf1b53fed0d1f4440ef5cf5ca70a02fb080ab70a1c",
    "development": "20261123T121624794620Z-632e2f91e0384cb4=a5225a622f02995b0d5044e91916380f4ddc1cab1639643a30a893bd6a2009b3",
}
DEFAULT_STUDY_SHA = "1467a9ffaff46b35d879dfaae9ac9784fbee9f8621bbaee782e250810bc32159"
ORIGINAL_CATALOG_SHA = "fb7fabef06d006c992bc38cb03c283a41d90238e0b174664a1357f274bb36baa"
SOURCES = ["vision_sources.py"]
DATA = SOURCES + ["vision_data.py"]
LR = DATA + ["vision_baselines.py"]
EVAL = LR + ["vision_evaluate.py"]


def candidate_trials(store, catalog, split, names):
    roster = {r["scene_family_id"]: r for r in catalog["families"] if r["split"] == split}
    require(set(names) <= set(roster), "acquisition list differs from preregistered learning families")
    trials = {n: roster[n]["trials"]["B0"] for n in sorted(names)}
    paths = {n: store / "trials" / trial / "trial_selection.json" for n, trial in trials.items()}
    missing = [trials[n] for n, path in paths.items() if not path.is_file()]
    return paths, missing


def new_transport(rec, catalog, path, split, names, phase, paths):
    parents = {}
    hashes = {}
    inputs = {"catalog": (path, None)}
    for i, (family, selection_path) in enumerate(sorted(paths.items())):
        hashes[family] = file_hash(selection_path)
        q = signed(selection_path, hashes[family])
        require(q["scene_family_id"] == family and q["condition_id"] == "B0", "incorrect selection identity")
        pin = compact_pin(q["reference"] if q["selection_status"] == "failed" else q["selected"]["evaluation"])
        parents[pin["run_id"]] = pin
        inputs[f"selection_{i:02d}"] = (selection_path, None)
    if phase == "pilot":
        require(set(names) == set(catalog["pilot_families"]) and len(names) == 4, "wrong pilot population")
    prefix = phase + "-" + split
    choose = [item for name in sorted(names) for item in ("--family", name)]
    outer = ["--store", str(rec.store), "--shelf", str(rec.shelf)]
    reg = rec.stage(
        prefix + "-registry",
        "dataset_build",
        list(parents.values()),
        inputs,
        SOURCES + ["vision_registry.py"],
        [
            "vision_registry.py",
            *outer,
            "--catalog",
            "{input:catalog}",
            rec.catalog_sha,
            "--split",
            split,
            *choose,
            "--output",
            "{out}",
        ],
        "selections.json",
        100000,
        {"family_selections": hashes},
    )
    rp, rs = rec.asset(reg, "selections.json")
    approved = load(rp)
    require(
        {
            f: approved["trials"][q["trials"]["B0"]]["selection_sha256"]
            for q in catalog["families"]
            if (f := q["scene_family_id"]) in names
        }
        == hashes,
        "trial selection changed during registry publication",
    )
    material = rec.stage(
        prefix + "-materials",
        "dataset_build",
        [reg],
        {"catalog": (path, None), "selections": (rp, reg["run_id"])},
        SOURCES,
        [
            "vision_sources.py",
            *outer,
            "--catalog",
            "{input:catalog}",
            "--catalog-sha256",
            rec.catalog_sha,
            "--selections",
            "{input:selections}",
            "--selection-sha256",
            rs,
            "--split",
            split,
            *choose,
            "--output",
            "{out}",
        ],
        "materials.json",
        1000000,
        {"families": sorted(names)},
    )
    mp, ms = rec.asset(material, "materials.json")
    pilot_opts = [v for f in sorted(names) for v in ("--pilot-family", f)] if phase == "pilot" else []
    packed = rec.stage(
        prefix + "-transport",
        "dataset_build",
        [material],
        {"materials": (mp, material["run_id"])},
        DATA,
        [
            "vision_data.py",
            "prepare",
            *outer,
            "--index",
            "{input:materials}",
            ms,
            "--split",
            split,
            *pilot_opts,
            "--output",
            "{out}",
        ],
        "dataset.json",
        len(names) * (23000000 if split == "train" else 120000000),
        {"families": sorted(names)},
    )
    return {"registry": reg, "materials": material, "transport": packed}


def source_input(rec, pin, name):
    a, sha = rec.asset(pin, name)
    return a, sha, (a, pin["run_id"])


def learn(rec, phase, catalog, new_path, original_path, original, budget):
    if phase == "pilot":
        splits = {"train": catalog["pilot_families"]}
    else:
        splits = {
            s: [r["scene_family_id"] for r in catalog["families"] if r["split"] == s]
            for s in ("train", "development")
        }
    selected = {s: candidate_trials(rec.store, catalog, s, names) for s, names in splits.items()}
    waiting = {s: m for s, (_, m) in selected.items() if m}
    if waiting:
        return {"status": "awaiting_selections", "phase": phase, "missing_trials": waiting}
    budget_pin = None
    if phase == "full":
        pilot = rec.state / "pilot-train-transport/selected.json"
        if not budget or not pilot.is_file():
            return {"status": "awaiting_pilot_budget", "phase": phase}
        pilot_pin = load(pilot)
        budget_pin = parse_pin(budget)
        bp, _ = rec.asset(budget_pin, "budget.json")
        b = load(bp)
        require(
            b["schema"] == "el-stage2-pilot-budget-1"
            and b["catalog_sha256"] == rec.catalog_sha
            and b["status"] == "approved"
            and b["decision"] == "proceed"
            and sorted(b["pilot_families"]) == sorted(catalog["pilot_families"])
            and len(set(b["pilot_families"])) == 4
            and b["pilot_dataset_sha256"] == rec.asset(pilot_pin, "dataset.json")[1]
            and isinstance(b["measurements"], dict)
            and b["measurements"],
            "complete measured pilot budget required",
        )
    stages = {}
    combined = {}
    new_cat = (new_path, None)
    orig_cat = (original_path, None)
    catalog_opts = ["--catalog", "{input:catalog}", rec.catalog_sha]
    if phase == "full":
        catalog_opts += ["--catalog", "{input:original_catalog}", ORIGINAL_CATALOG_SHA]
    for split, names in splits.items():
        stages[split] = new_transport(rec, catalog, new_path, split, names, phase, selected[split][0])
        packed = stages[split]["transport"]
        if phase == "full":
            original_pin = parse_pin(original[split])
            op, os, oi = source_input(rec, original_pin, "dataset.json")
            np, ns, ni = source_input(rec, packed, "dataset.json")
            budget_path, _, bi = source_input(rec, budget_pin, "budget.json")
            # Join validated member bytes by hard link locally. The final dataset.json pins every transport.
            packed = rec.stage(
                "full-" + split + "-combined",
                "dataset_build",
                [original_pin, packed, budget_pin],
                {"original_data": oi, "new_data": ni, "budget": bi},
                DATA,
                [
                    "vision_data.py",
                    "combine",
                    "--split",
                    split,
                    "--dataset",
                    str(op.parent),
                    os,
                    "--dataset",
                    str(np.parent),
                    ns,
                    "--output",
                    "{out}",
                ],
                "dataset.json",
                100000000,
                {"old": original[split], "new": packed["run_id"]},
            )
            stages[split]["combined"] = packed
        combined[split] = packed
    tp, ts, ti = source_input(rec, combined["train"], "dataset.json")
    inputs = {"dataset": ti, "catalog": new_cat}
    if phase == "full":
        inputs["original_catalog"] = orig_cat
    pre = rec.stage(
        phase + "-preprocessing",
        "features",
        [combined["train"]],
        inputs,
        LR,
        [
            "vision_data.py",
            "preprocess",
            "--dataset",
            str(tp.parent),
            "--sha256",
            ts,
            *catalog_opts,
            *(["--pilot"] if phase == "pilot" else []),
            "--output",
            "{out}",
        ],
        "preprocessing.json",
        100000,
        {"train_dataset_sha256": ts, "scope": phase},
    )
    stages["preprocessing"] = pre
    if phase == "full":
        _, _, pi = source_input(rec, pre, "preprocessing.json")
        fit = rec.stage(
            "full-lr-fit",
            "model_fit",
            [combined["train"], pre],
            {**inputs, "preprocessing": pi},
            LR,
            [
                "vision_baselines.py",
                "fit",
                "--dataset",
                str(tp.parent),
                "--sha256",
                ts,
                *catalog_opts,
                "--run-id",
                "{run_id}",
                "--output",
                "{out}",
            ],
            "fit.json",
            130000000,
        )
        stages["baseline_fit"] = fit
        mp, ms, mi = source_input(rec, fit, "model/model.json")
        dp, ds, di = source_input(rec, combined["development"], "dataset.json")
        sc = rec.stage(
            "full-dev-baseline-scores",
            "model_score",
            [combined["development"], fit],
            {"model": mi, "dataset": di},
            EVAL,
            [
                "vision_baselines.py",
                "score",
                "--dataset",
                str(dp.parent),
                "--sha256",
                ds,
                "--model",
                str(mp.parent),
                "--model-sha256",
                ms,
                "--split",
                "development",
                "--output",
                "{out}",
            ],
            "scores.json",
            220000000,
        )
        stages["development_baseline_scores"] = sc
        sp, ss, si = source_input(rec, sc, "scores.json")
        fr = rec.stage(
            "full-dev-baseline-thresholds",
            "threshold_select",
            [sc, combined["development"]],
            {"scores": si, "dataset": di, "catalog": new_cat, "original_catalog": orig_cat},
            EVAL,
            [
                "vision_evaluate.py",
                "freeze-dev",
                "--baseline-only",
                "--dataset",
                str(dp.parent),
                "--sha256",
                ds,
                "--scores",
                str(sp),
                ss,
                *catalog_opts,
                "--run-id",
                "{run_id}",
                "--output",
                "{out}",
            ],
            "thresholds.json",
            200000,
        )
        stages["development_baseline_thresholds"] = fr
    return {"status": "completed", "phase": phase, "catalog_sha256": rec.catalog_sha, "stages": stages}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("pilot", "full"))
    p.add_argument("--project", type=Path, default=Path.cwd())
    p.add_argument("--store", type=Path, default=Path(".research"))
    p.add_argument("--shelf", type=Path)
    p.add_argument("--study-sha256", default=DEFAULT_STUDY_SHA)
    p.add_argument("--budget", metavar="RUN_ID=MANIFEST_SHA256")
    p.add_argument("--original-train", default=ORIGINAL["train"])
    p.add_argument("--original-development", default=ORIGINAL["development"])
    p.add_argument("--retry", action="append", default=[], metavar="STAGE=FAILED_RUN_ID")
    a = p.parse_args()
    project = a.project.resolve()
    store = a.store.resolve()
    study = signed(project / "protocol/stage2-training-v1.json", a.study_sha256)
    path = project / "protocol/stage2-families-v1.json"
    catalog = signed(path, study["catalog_sha256"])
    orig = project / "protocol/families-v1.json"
    signed(orig, ORIGINAL_CATALOG_SHA)
    root = store / "stage2" / study["catalog_sha256"] / "learning"
    rec = Recorder(
        project,
        store,
        a.shelf or store.parent / "cold-shelf",
        root,
        study["catalog_sha256"],
        a.study_sha256,
        dict(value.split("=", 1) for value in a.retry),
    )
    result = learn(
        rec,
        a.phase,
        catalog,
        path,
        orig,
        {"train": a.original_train, "development": a.original_development},
        a.budget,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
