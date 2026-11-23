# Evidence Lens

Evidence Lens studies where a single photograph supports a nearby synthesized view. Its CPU scorer uses the input PNG, calibrated cameras and the generator's own predicted RGB, expected camera depth and accumulated opacity. A separate evaluator compares the prediction with a fixed Blender world.

L0 is a deterministic rule baseline. Two L2-regularized logistic regressions, L1-full and L1-geometry, were fitted on 393,216 uniformly sampled target pixels from 16 authored scene families. Both used the same rows. The frozen training manifest identifies all 96 target views, their sampled pixel coordinates and their family folds.

## Tested configurations

The first reference contains a single opaque procedural tabletop with a foreground occluder, two thin stems and a partly hidden rear object. Blender 4.5.16 LTS on macOS 27.2 arm64 rendered all seven 512×512 views with Cycles on CPU. An independent readback reopened the frozen scene and matched all 1,216 sampled visibility classifications. A fresh CPython 3.12 base install loaded the CPU scorer with NumPy 2.5.5 and Pillow 12.3.0. On a Windows RTX 3060 Laptop GPU with PyTorch 2.8.0+cu128, a single SHARP forward pass took 14.81 seconds and produced 1,179,648 Gaussians. The Mac verified the complete generation archive against its producer's SHA-256.

Windows 11 build 26200 rendered the source and six target views with gsplat 1.5.3 using expected optical-axis depth in metres. The Mac base-only CPU installation scored the six targets in a recorded stage of 3.20 seconds with a sampled process-tree peak of 195 MiB. The independent evaluator found 416,365 RGB errors above 0.05 among 1,572,864 comparable target pixels. L0 flagged 27.82% of those errors and left a 21.54% error rate among pixels it marked low risk. One fixed view was checked again from its recorded arrays, including all 262,144 pixel labels and nine metric ratios.

## CPU scoring

Use Python 3.11 or later. Install from a checkout in a fresh virtual environment.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m evidence_lens.risk_model --help
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`. The base installation uses NumPy and Pillow. With a compatible prediction bundle, run

```sh
python -m evidence_lens.risk_model --input example/input.png --cameras example/cameras.json \
  --predictions example/predictions --output results/l0
```

Here `example/` denotes the input, camera and predictions supplied by a producer. The authored tabletop-plane example has a 128×128 input and seven cameras. Its complete prediction arrays accompany the model release. The six target views have separate numeric scores and high, low and unknown decisions. Their scope is single-image, nearby perspective views of static opaque scenes.

## Research reproduction

The recorder, Blender and Windows GPU worker use separate environments. Install recording and evaluation extras with `python -m pip install '.[record]'`. For the pinned Blender 4.5 LTS Python module use **Python 3.11** in another environment. See the environment locks in `requirements/`. Render the reference through the recorder. Pass the resulting source image and cameras to the Windows worker. Return its sealed generation and render records to the Mac, then run CPU scoring and independent evaluation. Each attempt has its own ID and checksum manifest.

The original protocol is Evidence Lens master plan v1.1, numerical protocol 1.0. The engineering family belongs to training. It cannot establish cross-family performance. The preregistered split assigns 16 families to training, 4 to development and 4 to held-out testing. The CPU scorer reads the input PNG, cameras and predicted arrays. Research-only dependencies are separate.

- [Rebuild the study in a fresh checkout](docs/rebuild-study.md) with the authored scenes, registered splits and historical Windows worker.
- Module inputs, outputs and storage responsibilities are documented in [the data contract](docs/data-contract.md).
- The [Blender reference guide](docs/blender-reference.md) describes the frozen scene, numeric channels and independent readback.
- [Recording and transfer](docs/recording.md) documents how to run and receive an attempt.
- The [CPU input and output dictionary](docs/cpu-scoring.md) specifies rule states, missing values, metric denominators and examples.
- The [Windows worker guide](docs/windows-worker.md) covers the input handoff, dependency versions and return packages.
- License decisions and upstream source pins are in [the license inventory](docs/licenses.md).

Raw arrays, Gaussians, experiments, failure logs and the live SQLite database belong in a private research store. The public source file list is explicit. Model weights, data licensing, held-out evaluation, clean example reproduction and an identified public GitHub destination are separate release requirements.
