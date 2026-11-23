# Reproducing the reconstruction recovery experiment

The original Stage C case 057 and its source cameras are downloadable under the example terms. An independent CPU replay on macOS arm64 reconstructed the same 66 starting points as the input sent to the Windows optimizer. The point positions, linear colours, metric scales, rotations and opacities matched byte for byte. The run also verified the example ZIP before extracting its four source photographs and source calibration. [Public example and terms](https://github.com/wany614/evidence-lens/releases/tag/v0.3.0-rc.1) · [Source initialization rules](../protocol/recovery-initialization-v1.json)

## Replay the starting points

Use Python 3.12 in the source checkout. The commands create separate recorder and feature extraction environments. The recorder uses the project's existing 30 GiB free-space floor. The example ZIP is 99,402,309 bytes.

```sh
python3.12 -m venv .venv-r-record
.venv-r-record/bin/python -m pip install -e '.[record]'
python3.12 -m venv .venv-r-geometry
.venv-r-geometry/bin/python -m pip install -r requirements/mac-recovery-init.txt
curl -fL -o ../evidence-lens-example-v0.3.0-rc.1.zip \
  https://github.com/wany614/evidence-lens/releases/download/v0.3.0-rc.1/evidence-lens-example-v0.3.0-rc.1.zip
.venv-r-record/bin/python scripts/recovery_public_control.py \
  --store ../evidence-lens-recovery-store \
  --example-zip ../evidence-lens-example-v0.3.0-rc.1.zip \
  --geometry-python .venv-r-geometry/bin/python \
  --export-worker-cell ../recovery-057-M4
```

The tested Python 3.12.16 run `20261123T212809916051Z-d398fa5d52924e3c` produced parameter digest `13c0649f58d4f6ee50775f7db4a553957f93f18698c616c43aae1b6018dd4b99` and an `init.npz` with SHA-256 `1c58d61d5726b320a6d025fbdbb262b0e9b3a02825f14f102215286f088d15ff`. The run's `payload` includes the admitted sparse tracks, per-source coverage grid, per-pair matching statistics and initialization metadata. A fresh run has its own recorder identity. Other hardware or feature extraction builds can change the detected keypoints. [Pinned feature environment](../requirements/mac-recovery-init.txt) · [Installed distribution notices](../licenses/recovery-initialization-dependencies.json)

## Reconstruct the authored positive control

The public cell export writes the four original source photographs, source and render calibrations, initial arrays, provenance and a fresh job ticket. Keep its printed `job_sha256`. The six target photographs have no role in this cell. The independent Mac replay matched the scientific handoff on both calibrations, all four source photographs and the complete initial array file. The initializer metadata identifies the replay's own inputs and execution.

Materialize the registered worker from a Git checkout that includes the D1 scientific commit. This step checks every file against the Windows source-tree manifest. The existing CPU scoring installation is sufficient for scoring the published example. The following CUDA stage uses the separate research environment.

```sh
python3 scripts/recovery_build_worker.py --out ../recovery-windows-worker
```

Copy the exported cell and materialized worker to the Windows research drive. Use the worker's `configs/candidate-recipe.json` for `$Recipe`, the printed job hash for `$JobSHA`, and new recorder and return directories. From the materialized worker directory run the optimize and render commands in its README. `--steps all` renders all five issued checkpoints. The original research receipts identify the measured CUDA environment. Independent GPU implementations may produce numerical differences. [Windows commands and original worker contract](../reproduce/recovery-worker-v1/README.md) · [CUDA research environment](windows-worker.md)

## Access to the real photographs

Obtain DTU photographs from the authors. The reconstruction study used the D1 registration for lighting 3, centre-crop intrinsics, source identities and six target camera poses. M2 used the two endpoint photographs. M4 used all four source photographs. The archived Windows inputs supply these views, the initializer and the source and render camera files. Training reads only source calibration. Target photographs enter the separate evaluator after the checkpoint renders have been saved. The research reports publish file hashes and measurements. The photo-containing boards are available in the owner's research archive. [Official DTU data](https://roboimagedata.compute.dtu.dk/?page_id=36) · [Original download and calibration replay](d1/reproduction.md) · [Recovery registration](../protocol/reconstruction-recovery-v1.json)
