"""Build an example pinned job.json from an existing source-only cell and registration pins."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
for key in ("cell", "recipe"):
    p.add_argument("--" + key, required=True, type=Path)
for key in ("object-id", "experiment-id", "trial-id"):
    p.add_argument("--" + key, required=True)
p.add_argument("--condition", choices=("M2", "M4"), required=True)
p.add_argument("--stop-step", type=int, default=5000)
p.add_argument("--parent", action="append", required=True, metavar="RUN=MANIFEST_SHA256")
a = p.parse_args()
def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
ids = ("s00", "s03") if a.condition == "M2" else ("s00", "s01", "s02", "s03")
paths = ("init.json", "init.npz", "source_cameras.json", "render_cameras.json", *("sources/" + s + ".png" for s in ids))
parents = []
for raw in a.parent:
    rid, manifest = raw.split("=", 1)
    parents.append({"run_id": rid, "manifest_sha256": manifest})
doc = {"schema": "el-recovery-cell-1", "purpose": "formal", "object_id": a.object_id, "condition": a.condition,
       "experiment_id": a.experiment_id, "trial_id": a.trial_id, "stop_step": a.stop_step, "parents": parents,
       "recipe_sha256": digest(a.recipe),
       "files": {n: {"bytes": (a.cell / n).stat().st_size, "sha256": digest(a.cell / n)} for n in paths}}
dest = a.cell / "job.json"
with dest.open("x", encoding="utf8", newline="\n") as f:
    json.dump(doc, f, indent=2, sort_keys=True)
    f.write("\n")
print(digest(dest))
