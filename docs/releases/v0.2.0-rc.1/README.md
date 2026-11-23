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

## Scientific measurements

The eight new confirmation families and four historical regression families are reported separately in `study/study-summary.json` and `study/family-metrics.csv`. Each family has B0, R1 and Q1 conditions. Read the support and error counts alongside each result. The formal public assets contain quantitative results. The downloadable images and predictions belong to the separate authored deployment example.

See [the model card](MODEL_CARD.md), [the source and rebuild guide](REBUILD.md), and [the asset license inventory](licenses/ASSET-SCOPE.json).
