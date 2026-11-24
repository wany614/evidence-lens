"""Audit one original batch-two Windows execution and emit its native, truth-free scoring case."""

import argparse
from pathlib import Path
from evidence_lens._bundle import encoded, load, sha
from evidence_lens._verify import require
from t0_reconstruct_readback import finish


def audit(store, config, out):
    ready = load(store / "runs" / config["originals"]["dispatch"]["run_id"] / "payload/READY.json")
    srcroot = store / "runs" / ready["source"]["run_id"]
    src = load(srcroot / ready["source"]["member"])
    study = load(store / "runs" / ready["freeze"]["run_id"] / "payload/protocol.json")
    require(
        ready["schema"] == "el-b2-worker-ticket-1"
        and not ready["forced_fallback_fixture"]
        and ready["target_reference_members"] == 0
        and sha(srcroot / ready["source"]["member"]) == ready["source"]["sha256"]
        and ready["official_by_slot"] == {s: rec["official_view_id"] for s, rec in src["sources"].items()}
        and ready["object_id"] == src["object_id"]
        and ready["condition"] == src["condition"]
        and ready["optimizer_seed"]
        == load(store / "runs" / ready["freeze"]["run_id"] / "payload/recipe.json")["seed"],
        "Wrong batch-two scientific identity, source authority or optimizer seed",
    )
    if ready["cohort"]:
        cohort = ready["cohort"]
        f = store / "runs" / cohort["run_id"] / cohort["member"]
        c = load(f)
        require(
            src.get("cohort") == cohort
            and sha(f) == cohort["sha256"]
            and ready["object_id"] in c["splits"]["C6"]
            and ready["object_id"] not in c["splits"]["T1"]
            and ready["layout"] == ready["condition"] + "-fixed"
            and ready["optimizer_seed"] == 43171
            and ready["selection"] is None,
            "Unregistered fixed C6 execution",
        )
    elif ready["selection"]:
        selection = ready["selection"]
        f = store / "runs" / selection["run_id"] / selection["member"]
        choice = load(f)
        require(
            src["selection"] == selection
            and sha(f) == selection["sha256"]
            and choice["object_id"] == ready["object_id"]
            and choice["strategy_id"] in study["development_selection"]["strategies"]
            and sorted(choice["selected_official_ids"])
            == [ready["official_by_slot"][s] for s in ("s01", "s02")]
            and ready["optimizer_seed"] == 43171
            and ready["layout"] == "M4-selected",
            "Unregistered adaptive D execution",
        )
    else:
        require(
            ready["object_id"] in study["development_objects"]
            and ready["layout"] == "M4-fixed"
            and ready["optimizer_seed"] in (43172, 43173),
            "Unregistered sensitivity",
        )
    finish(store, config, out)
    target = out / "reconstruction-readback.json"
    info = load(target)
    info.update(
        schema="el-b2-reconstruction-readback-1",
        **{k: ready[k] for k in ("cohort", "selection", "layout", "official_by_slot", "optimizer_seed")},
    )
    target.write_bytes(encoded(info))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("store", "config", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    audit(a.store, load(a.config), a.out)
