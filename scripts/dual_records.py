"""Use the existing recorder for the separately sealed A, B and C research stages."""

from pathlib import Path

from evidence_lens._bundle import load
from t0_records import bind, record, resolve


def selected(store, name):
    return load(Path(store) / "work" / "controller" / name / "selected.json")


def stage(store, name, kind, parents, details, command, receipt, estimate, extra_files=None, retry=None):
    store = Path(store)
    phase = name.split("-", 1)[0]
    if phase not in {"a", "b", "c"}:
        raise ValueError("identify the protocol phase in the stage name")
    config = dict(details, experiment_id="dual-track-" + phase, trial_id=name)
    return record(store, store / "history", name, kind, parents, config, command,
                  receipt, estimate, extra_files=extra_files, retry=retry)
