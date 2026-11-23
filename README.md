# Evidence Lens

Evidence Lens studies where a single photograph supports a nearby synthesized view. Its CPU scorer uses the input PNG, calibrated cameras and the generator's own predicted RGB, expected camera depth and accumulated opacity. A separate evaluator compares the prediction with a fixed Blender world.

This is a **research implementation in development**. L0 is a deterministic rule baseline. The trained L1 classifier, frozen operating threshold and downloadable model release are pending the family-split study.

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
  --predictions example/prediction --output results/l0
```

Here `example/` denotes files supplied by a producer. A redistributable example with a trained model is scheduled for the model release. The six target views have separate numeric scores and high, low and unknown decisions. Their scope is single-image, nearby perspective views of static opaque scenes.

## Research reproduction

The recorder, Blender and Windows GPU worker use separate environments. Install recording and evaluation extras with `python -m pip install '.[record]'`. For the pinned Blender 4.5 LTS Python module use **Python 3.11** in another environment. See the environment locks in `requirements/`. Render the reference through the recorder. Pass the resulting source image and cameras to the Windows worker. Return its sealed generation and render records to the Mac, then run CPU scoring and independent evaluation. Each attempt has its own ID and checksum manifest.

The original protocol is Evidence Lens master plan v1.1, numerical protocol 1.0. The engineering family belongs to training. It cannot establish cross-family performance. The formal split is planned as 16 train, 4 development and 4 held-out families. The CPU scorer does not load testing labels, Blender, CUDA, SHARP, PyArrow or trained parameters for L0.

- Module inputs, outputs and storage responsibilities are documented in [the data contract](docs/data-contract.md).
- The [Windows worker guide](docs/windows-worker.md) covers the input handoff, dependency versions and return packages.
- License decisions and upstream source pins are in [the license inventory](docs/licenses.md).

Raw arrays, Gaussians, experiments, failure logs and the live SQLite database belong in a private research store. The public source file list is explicit. Model weights, data licensing, held-out evaluation, clean example reproduction and an identified public GitHub destination are separate release requirements.
