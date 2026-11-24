"""Record role-limited H authorization and read only declared evaluator members with verified source identity."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from dual_records import resolve, selected, stage
from evidence_lens._bundle import load, save, sha
from evidence_lens._verify import require


def grant(a):
    doc = load(a.registration)
    scope = {
        r["id"]: [f"h{i:02d}" for i in range(6)]
        for r in doc["originals"]
        if r["split"] in ("train", "development")
    }
    scope.update(
        {
            r["id"]: [f"h{i:02d}" for i in range(r["H"])]
            for r in doc["external"]["roster"]
            if r["split"] == "development"
        }
    )
    save(
        a.out / "opening.json",
        {
            "schema": "el-many-h-opening-1",
            "phase": "train-development",
            "run_id": os.environ["EL_RUN_ID"],
            "allowed_targets": scope,
            "individual_H_photo_reads": 0,
            "confirmation_scenes_permitted": [],
            "purpose": "original training labels; registered original and real development selection and upstream diagnostic",
            "roles_registration_sha256": sha(a.registration),
        },
    )


class EvaluatorTruth:
    """An opened evaluator archive; score and reconstruction workers receive only the separate R package."""

    def __init__(self, index, expected_index_sha256, *, bank=None):
        self.index = Path(index).resolve()
        require(sha(self.index) == expected_index_sha256, "different native truth receipt")
        self.meta = load(self.index)
        require(self.meta["schema"] == "el-many-evaluator-truth-1", "invalid H-only truth receipt")
        if bank is not None:
            self.verify_bank(bank)
        ref = self.meta["archive"]
        self.path = self.index.parent / ref["path"]
        require(
            self.path.resolve().parent == self.index.parent.resolve()
            and self.path.stat().st_size == ref["bytes"]
            and sha(self.path) == ref["sha256"],
            "different authorized truth archive bytes",
        )

    def verify_bank(self, bank, *, store=None):
        """Bind every actual R/H camera and the physical R archive to this truth receipt's parent."""
        b = bank.meta
        provenance = b.get("provenance", {})
        source = self.meta["inputs"]["source"]
        data = provenance.get("native_evidence", {}).get("data", {})
        n = len(bank.reconstruction_ids)
        require(
            b["scene_id"] == self.meta["scene_id"]
            and provenance.get("source_acquisition") == source["run_id"] == data.get("run_id")
            and data.get("manifest_sha256") == source["manifest_sha256"]
            and data.get("member") == f"payload/R{n}.zip",
            "different native source acquisition, scene or reconstruction archive",
        )
        root = Path(store) if store is not None else self.index.parents[3]
        roots = (root, root / "history")
        delivery = load(resolve(roots, source))
        physical = resolve(roots, data)
        spec = delivery["packages"][f"R{n}"]
        require(
            physical.name == spec["path"]
            and physical.stat().st_size == spec["bytes"]
            and data["sha256"] == spec["sha256"],
            "source package differs from truth acquisition",
        )
        with zipfile.ZipFile(physical) as z:
            scene = json.loads(z.read("scene.json"))
        expected = {
            v: {k: scene["views"][v][k] for k in ("width", "height", "K", "c2w")}
            for v in scene["R"] + scene["H"]
        }
        frame = dict(scene["frame_receipt"], matrix=scene["frame_receipt"]["world_to_deployment"])
        require(
            scene["run_id"] == source["run_id"]
            and scene["scene_id"] == self.meta["scene_id"] == delivery["scene_id"]
            and scene["split"] == self.meta["split"]
            and scene["R"] == list(bank.reconstruction_ids)
            and scene["A"] == list(bank.accessible_ids)
            and scene["H"] == list(bank.target_ids) == self.meta["H"]
            and set(scene["R"]) <= set(self.meta["R"])
            and expected == b["cameras"]
            and scene["grids"] == b["grids"]
            and b["frame_receipt"] == frame
            and b["metric_anchor_ids"] == frame["metric_anchor_ids"]
            and scene["unit"] == frame["unit"] == provenance["unit"]
            and all(
                scene["original_rgb"][v]["sha256"] == b["sources"][v]["original_sha256"]
                and scene["source_rgb"][v]["sha256"] == b["sources"][v]["photo"]["sha256"]
                for v in scene["R"]
            ),
            "truth and prediction differ in cameras, grid, frame, units or actual source pictures",
        )
        return {
            "source_acquisition": source,
            "configuration_archive": data,
            "camera_count": len(expected),
            "camera_calibration_sha256": hashlib.sha256(
                json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest(),
            "frame_unit": frame["unit"],
            "R": n,
            "H": len(scene["H"]),
        }

    def _read(self, ref):
        with zipfile.ZipFile(self.path) as z:
            raw = z.read(ref["path"])
        require(
            len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"],
            "different role-tagged truth bytes",
        )
        return raw

    def target_rgb(self, target):
        require(target in self.meta["H"], "registered target H required")
        ref = self.meta["targets"][target]["analysis_rgb"]
        with Image.open(io.BytesIO(self._read(ref))) as image:
            require(image.mode == "RGB" and image.size == (378, 252), "incorrect H analysis grid")
            return np.asarray(image, dtype=np.float32) / np.float32(255)

    def optical_z(self, view):
        require(view in self.meta["z"], "registered authored optical depth unavailable for requested view")
        with np.load(io.BytesIO(self._read(self.meta["z"][view])), allow_pickle=False) as record:
            require(record.files == ["z"], "unexpected authored depth channels")
            z = record["z"]
        require(
            z.shape == (252, 378)
            and z.dtype == np.float32
            and not np.isinf(z).any()
            and np.all(z[np.isfinite(z)] > 0),
            "invalid authored optical-axis grid",
        )
        return z


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("record-development", "grant-worker"))
    for n in ("store", "registration", "out"):
        p.add_argument("--" + n, type=Path)
    a = p.parse_args()
    if a.action == "grant-worker":
        grant(a)
    else:
        s = a.store.resolve()
        pins = {"roles": selected(s, "b-data-registration"), "study": selected(s, "b-study-registration")}
        pin = stage(
            s,
            "b-train-dev-h-opening",
            "archive_verify",
            pins,
            {"phase": "train-development", "confirmation_disallowed": True},
            [
                "{python}",
                "-B",
                "{source}/scripts/many_data_access.py",
                "grant-worker",
                "--registration",
                "{input:roles}",
                "--out",
                "{out}",
            ],
            "payload/opening.json",
            12000,
        )
        print(json.dumps(pin), flush=True)
