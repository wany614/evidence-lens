"""Use the project recorder for batch B, with a separate experiment and trial namespace."""

from pathlib import Path
from t0_records import bind, record, resolve
from evidence_lens._bundle import load


def stage(
    store,
    history,
    name,
    kind,
    pins,
    details,
    command,
    member,
    size,
    extra=None,
    retry=None,
):
    settings = {
        **details,
        "experiment_id": "batch-two-dtu-v1",
        "trial_id": "b2-" + name,
    }
    return record(
        store, history, name, kind, pins, settings, command, member, size, extra, retry
    )


def selected(store, name):
    return load(Path(store) / "work/controller" / name / "selected.json")
