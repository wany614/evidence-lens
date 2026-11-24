"""Bind the humanities source reading and the already designated authored cases to A and B."""

import argparse
import os
from pathlib import Path

from dual_records import selected, stage
from evidence_lens._bundle import load, save
from evidence_lens._verify import require


def register(store):
    labels = {
        "A": "a-baseline",
        "B": "b-research-release",
        "study": "b-study-registration",
        "data": "b-data-registration",
        "findings": "b-scientific-closeout",
    }
    parents = {role: selected(store, label) for role, label in labels.items()}
    return stage(store, "c-foundation-registration", "archive_verify", parents,
                 {"version": "humanities-foundation-v1", "additional_training": 0},
                 ["{python}", "scripts/foundation_register.py", "--config", "{config}",
                  "--release", "{input:B}", "--status", "{input:findings}", "--out", "{out}"],
                 "payload/registration.json", 12000)


def write(args):
    config = load(args.config)
    release = load(args.release)
    status = load(args.status)
    require(release["status"] == status["status"] and status["status"] == "completed_negative",
            "foundation must use the sealed two-scene comparison")
    require("sha256:" + config["sources"]["A"]["sha256"] == release["A_BASELINE_ID"], "A identity changed")
    save(args.out / "registration.json", dict(
        schema="el-humanities-foundation-registration-1", version=config["version"],
        run_id=os.environ["EL_RUN_ID"], parents=config["sources"], B_code_commit=release["code_commit"],
        B_status=status["status"], A_BASELINE_ID=release["A_BASELINE_ID"],
        cases=["offset-bell-cage, four real sources, all six released target views",
               "b05-perforated-gallery, R128, h00, all registered K on the identical prediction"],
        organization=["prediction", "model assessment", "sources and procedure", "separate evaluation"],
        reading=["Bolter and Grusin 1996, Remediation, pp. 313–314",
                 "London Charter 2.1, documentation 4.4–4.7 and glossary",
                 "Seville Principles, virtual archaeology, authenticity and scientific transparency"],
        stop="The evidence, sources and apparatus remain open for Wany to choose a later interpretation."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("store", "out", "config", "release", "status"):
        parser.add_argument("--" + field, type=Path)
    args = parser.parse_args()
    if args.store:
        print(register(args.store))
    else:
        write(args)
