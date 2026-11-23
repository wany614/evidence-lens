# Run Evidence Lens v0.1.0 on a CPU

**Private release candidate.** The project owner is reviewing the distribution terms and GitHub destination.

This release scores nearby views synthesized from a single image. Its L1-full model assigns a per-pixel risk score and a low, high or unknown decision. The study used static opaque Blender scenes, Apple SHARP at commit `aed6527499ef91cba3b54c18d49a870f25947190` and gsplat 1.5.3. It tested the original 512-pixel input, a half-resolution input and a colour-only FP16 round trip. Scores target mean absolute display-sRGB error above 0.05. Expected depth uses optical-axis metres. The bundled demonstration uses an independently authored plane projection and a 128-pixel input.

## Install and grade the example

Download the release ZIP and its SHA-256 sidecar together. From the download directory, verify the container with `shasum -a 256 -c <downloaded-sidecar>` on macOS or `sha256sum -c <downloaded-sidecar>` on Linux. In PowerShell, compare `Get-FileHash -Algorithm SHA256 <downloaded-zip>` with the sidecar. Extract the ZIP into a new folder and open a shell inside it.

Use Python 3.11 or later. The environment tested on macOS used CPython 3.12.16, NumPy 2.5.5 and Pillow 12.3.0. The wheel contains the Evidence Lens code. Pip obtains the two numeric and image dependencies from your package index or wheelhouse.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install ./distribution/evidence_lens-0.1.0-py3-none-any.whl
python -I -m evidence_lens.l1_demo grade --bundle . --manifest-sha256 bc6c3cc8580451deab569b6778042c92d0b93180d181d20d2ba5f896b2e497cc --output grade
python -I -m evidence_lens.score_maps --scores grade/scores --output maps
```

In PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1` in place of the second line. The installed modules and their command-line arguments are the same. Use fresh output directories for subsequent runs.

Open `maps/v01-risk.png` and `maps/v01-state.png`. Dark blue to orange shows increasing continuous risk on a fixed 0–1 scale. In the decision map, teal means low risk, orange means high risk and grey means unknown. The six targets each have their own PNGs, numeric NPZ and state counts. The NPZ codes are 0 for low, 1 for high and 255 for unknown.

`grade/grade.json` checks the first target against the release's fixed numerical expectation. It compares finite masks, scores and all five methods' decisions with absolute score tolerance `1e-6`. On the original six-view example the frozen L1-full decision maps contain 70,874 low, 22,761 high and 4,669 unknown pixels. The files `example/input.png` and `example/v01-prediction.png` show the input and one projected view.

## Score compatible predictions

Supply the input PNG, the calibrated source and target cameras, and a prediction bundle with the declared RGB encoding, accumulated alpha and validated expected camera-Z. The bundle needs its own source-view prediction and camera intrinsics for the actual input grid. Each target carries its own pose and dimensions. When the interface cannot establish the required depth semantics, L1 records an unsupported result. The risk scorer takes its inputs from these files. A separate research entry point consumes target references for evaluation.

```sh
python -I -m evidence_lens.l1 --input example/input.png --cameras example/cameras.json --predictions example/predictions --model model --model-sha256 b3c65130bf803611e3904ac69a270f6944b48a8d78ce049bdc1424c15ba6cb54 --output my-scores
python -I -m evidence_lens.score_maps --scores my-scores --method L1-full --view v01 --output my-maps
```

## Identify this build

The frozen model ID is `20261123T042819219876Z-27eaa01eba29489d`. The `release.json` file binds the release code commit, original method commit, model, training manifest, thresholds, study protocol, wheel and example. `SHA256SUMS` covers the release payload. The model card reports the training population, held-out results, false negatives, resource measurements and tested scope. Separate license files describe the original code, fitted parameters and illustrative data.
