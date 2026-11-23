# Evidence Lens v0.2.0-rc.1

The CPU bundle evaluates pixel-level risk in a calibrated predicted view. It contains the two logistic models, a visual model with a frozen image encoder, a visual model with two adapted DINOv2 blocks, and three reference methods. Open `study/selected-method.json` for the default chosen on development data. Model hashes, operating thresholds and the public source commit are in `release.json`.

## Reproduce the authored example

Download the CPU ZIP and its SHA-256 list from the same release. Verify the archive, then extract into a clean directory. From that directory, run the following command on macOS or Linux with Python 3.11 or 3.12.

```sh
python3 -I -B smoke.py --output ../vision-plane-result
```

On Windows with the Python launcher, run the equivalent command in PowerShell.

```powershell
py -3.12 -I -B smoke.py --output ../vision-plane-result
```

The script creates a temporary environment and installs CPU dependencies. A prepopulated wheel directory for the target Python and platform can be supplied with `--wheelhouse PATH`. Add `--archive PATH --archive-sha256 HEX` to compare every extracted file with an independently pinned ZIP. Choose a new output directory for each execution.

The example is wany614’s textured two-metre plane with known camera geometry and an authored prediction. Six target views produce seven score maps per view in `scores/`. `maps/` contains 36 risk and state PNGs for L1-full and both visual models. `smoke.json` compares each numerical output with a separate recorded CPU run, including finite support, threshold states, L0 reason codes and maximum absolute score difference. White risk pixels denote unknown scores. State previews use teal for low risk, orange for high risk and light grey for unknown. The state-array values are 0 for low risk, 1 for high risk and 255 for unknown.

## Score the independent 512-pixel plane control

The `--case` option accepts the separately distributed deployment control for `scene_037_bottle_carrier`. The authored two-metre plane adapter uses the registered development source image and camera calibration to produce six full-resolution target views. The control has the condition ID `public-plane`. The study's 40 SHARP conditions and the development model-selection calculations have their own records.

Extract the control attachment, obtain its published SHA-256 for `case.json` and point the command to the directory containing that file.

```sh
python3 -I -B smoke.py --case /path/to/CASE --case-sha256 "$CASE_SHA" --output ../public-plane-512
```

Windows PowerShell accepts `py -3.12 -I -B smoke.py --case $Case --case-sha256 $CaseSha --output ../public-plane-512`. The command verifies and opens ten deployment inputs, writes six seven-method NPZ files and 36 PNGs, and records input and output hashes, model identities and native support counts in `control-scores.json`. Blender references and independent error measurements accompany the control separately. The bundle's built-in 128-pixel example uses its own expected arrays and smoke receipt.

## Run the default L1 model with NumPy and Pillow

Install the tagged Evidence Lens source into a new Python environment with `python -m pip install .`. Extract the CPU archive into `vision-bundle` and the public 512-pixel control into `vision-control`. Run these commands from their parent directory with that environment's Python.

```sh
python -I -m evidence_lens.l1 --input vision-control/CASE/input.png --cameras vision-control/CASE/cameras.json --predictions vision-control/CASE/predictions --model vision-bundle/models/L1 --model-sha256 cd8573eaf809000be8510a7aa28f6f992215b75dbe2da44048ed1d0e34a18ca2 --output lightweight-scores
python -I -m evidence_lens.score_maps --scores lightweight-scores --output lightweight-maps
```

In a clean arm64 Mac environment, the wheel installed as `evidence-lens 0.2.0rc1` with NumPy 2.5.5 and Pillow 12.3.0. It produced all six native score files and 12 L1-full preview PNGs. Across five methods and 7,864,320 pixel scores on the authored control, finite support and threshold states matched the full pipeline exactly. The largest absolute score difference was zero. This check used release source revision `3d74f0734121` and run `20261123T150628833335Z-dfb6380cb0664f14`.

## Scientific measurements

The eight new confirmation families and four historical regression families are reported separately in `study/study-summary.json` and `study/family-metrics.csv`. Each family has B0, R1 and Q1 conditions. Read the support and error counts alongside each result. The formal public assets contain quantitative results. The CPU archive's images and predictions belong to the authored deployment example.

See [the model card](MODEL_CARD.md), [the source and rebuild guide](REBUILD.md), and the archive’s `licenses/ASSET-SCOPE.json`. [Source asset licenses](https://github.com/wany614/evidence-lens/blob/v0.2.0-rc.1/licenses/vision-assets.json)
