# Comparing engineering and frozen scores

On one macOS arm64 host, the file CLI, uncached resident API and cached resident API produced identical four-array outputs for all 64 method–view combinations across four original workloads. The two bead layouts had substantially less finite support than the corresponding bell layouts. The comparison includes NaN bit patterns, uint8 reasons, raw FP32 risk, the finite mask and the original threshold states. No threshold classification or attainable matched-quota membership changed. [Original development receipt](../general-closeout/numerical-bridge.json) · [Four report examples](../../examples/general-four-models/README.md)

| Original evidence | Targets | MS-L1 finite / full canvas |
| --- | ---: | ---: |
| Helical beads M2 | 3 | 33,366 / 196,608 |
| Helical beads M4 | 1 | 11,188 / 65,536 |
| Offset bell cage M2 | 6 | 387,167 / 393,216 |
| Offset bell cage M4 | 6 | 388,358 / 393,216 |

The original-replays archive contains all four inputs, 16 reference HTML reports, the original numerical outputs and an installed-wheel grader. Its 44,233,110 bytes have SHA-256 `0ed367c24019004d5126c823d899db3941f0fe9dd24473394c3cd6b581c7cfa0`. The expected.json SHA-256 is `7aed41037e7c4ec7f0a8583d33bdbe5291d6b02bf0ef8df325a13d337490f9cc`. [Grading commands](original-archive.md) · [Archive receipt](../general-closeout/original-replays.json)

A second control used the previously sealed C6 set48 M4 case. All 24 archived method–view scores matched both resident configurations on this host. Its original v0.3.0-rc.1 score archives stored risk, valid and state. An isolated interpreter reopened the deployment case with the authenticated original CPU runtime to recover the feature reason, using the original primary-source validity priority for B-L1-transfer and the archived finite mask for visual support. The per-pixel reason comparison explicitly records that provenance. Target photographs, true labels, evaluation masks and outcome rankings were outside the reader. [Historical C6 numerical receipt](../general-closeout/archived-c6-numerical-bridge.json)

## Reproduce the development bridge

With the archive unpacked in the current directory, a frozen configuration can be produced and the original four-workload roster reconstructed locally. SHA-256 identities in the config cover the installed production and comparison sources. Load all four original models from the verified CPU release.

```sh
python -m evidence_lens_bundle.bridge freeze-config --threads 4 --out cfg/engineering-config.json
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path.cwd()
config = root / "cfg" / "engineering-config.json"
gold = json.loads((root / "expected.json").read_bytes())
pin = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = [{"object_id": record["object_id"], "layout": record["condition"] + "-fixed",
         "bundle": {"directory": str(root / "inputs" / seq), "sha256": record["bundle_sha256"]}}
        for seq, record in gold["sequences"].items()]
roster = {"schema": "el-numerical-bridge-input-1", "stage": "original_development",
          "configuration": {"directory": str(config.parent), "sha256": pin(config)},
          "joint_closure": None, "cells": rows}
path = root / "development-roster.json"
with path.open("x") as out:
    json.dump(roster, out, sort_keys=True, indent=2)
print(pin(path))
PY
python -m evidence_lens_bundle.bridge run --manifest development-roster.json \
  --manifest-sha256 ROSTER_SHA256 --models /path/to/cpu --out bridge-result
```

The receipt gives signed and absolute score differences, old/new digests, changing mask/reason/state counts, cutoff crossings and distances to the original threshold. It compares accepted pixel IDs at the registered 1, 2, 5, 10, 20, 40, 60 and 80 percent quotas and both frozen MS-L1 acceptance counts for each target. Exact-k cutoff ties follow the frozen per-object, condition and view PCG64 ordering. The engineering comparison uses each implementation’s four-method finite intersection on the full deployment canvas. Independent scientific AP and error on a reference-evaluable domain require the separate evaluator.

## Sealed fixed-layout comparison

The sealed_T1 manifest contains exactly twelve fixed-layout dispositions from the six registered physical subjects. The reader authenticates the full recorder manifest, the sealed-matrix.json member, its 42 logical dispositions and its 288 registered fixed method–view outcomes. Successful rows must identify the exact case, readback and archived scores from the joint matrix. Each failed row cites the reason and hash of that matrix and remains visible for all six targets and four methods. The program emits scheduled, unavailable and actually compared denominators.

Successful direct-case inputs use a separately authenticated C6 fixed-layout Bundle as the original metric/crop/rectification geometry exemplar. Only the exemplar's JSON is read. Each target and source camera must match that layout exactly. Only authorized source RGB and predicted RGB/Z/opacity enter scoring. The model source index, checkpoint, producer revisions, source scope and old prediction digest are checked against the closed score records. A change in finite support or reasons, a threshold crossing, or a change in matched acceptance remains visible per method and view. Cross-platform evidence must name the host and dependency versions that actually produced its arrays.
