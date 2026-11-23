# Evidence Lens v0.1.1-rc.1

**Private release candidate.** Evidence Lens estimates the risk of a large RGB error at each pixel in a synthesized nearby view. Read the selected diagnostic finding, limits and status of the formal results before interpreting a score. The exact model ID, method digest and diagnostic run pins are supplied in RELEASE_SCOPE.json. [Scientific scope](SCIENTIFIC_SCOPE.md) · [Model card](MODEL_CARD.md) · [Known issues](KNOWN_ISSUES.md)

## Run the CPU example

Use Python 3.11 or newer. Obtain the ZIP checksum independently from the release owner. Compare it with the ZIP before extracting. On macOS use `shasum -a 256 evidence-lens-v0.1.1-rc.1-cpu.zip`, on Linux use `sha256sum`, and in PowerShell use `Get-FileHash -Algorithm SHA256`. Extract into an empty directory and open a terminal there.

```sh
python -I smoke.py --output ../evidence-lens-smoke
```

On Windows use `py -3 -I smoke.py --output ../evidence-lens-smoke`. The script creates a fresh virtual environment, installs the shipped wheel with NumPy and Pillow, checks every file against the manifest and grades all six targets. For an offline run, add `--wheelhouse PATH` pointing to compatible NumPy and Pillow wheels. If an institution uses a populated uv package cache, add `--offline-uv`. Use a new output directory for every run.

To bind the original downloaded ZIP to the same result, add `--archive PATH --archive-sha256 DIGEST` using its independently supplied checksum. The smoke checks all members of that ZIP against the extracted files and records the verified container digest.

The result is `smoke.json`, with the installed versions, platform, model, absolute score difference and mask/state checks. Open `maps/v01-risk.png` and `maps/v01-state.png` inside the output directory. The other five target views have the same file naming. Each view has a fixed 0–1 risk scale. The state map uses teal for low risk, orange for high risk and grey for unknown. The shipped demonstration uses an authored image projected onto a plane at two metres and 128 × 128 pixels. Its purpose is to check deployment. [Example source](example/provenance.json) · [Numerical grading manifest](demo.json)

## Score your own compatible prediction

Install the local wheel in a Python 3.11+ virtual environment. For v0.1.1-rc.1 its Python package version is `0.1.1rc1`. The model digest below is read from the already verified release manifest. From the extracted directory on macOS or Linux:

```sh
python -m venv ../evidence-lens-cpu
. ../evidence-lens-cpu/bin/activate
python -m pip install ./distribution/evidence_lens-0.1.1rc1-py3-none-any.whl
MODEL_SHA=$(python -I -c 'import json; print(json.load(open("release.json"))["model_index_sha256"])')
python -I -m evidence_lens.l1 --input example/input.png --cameras example/cameras.json --predictions example/predictions --model model --model-sha256 "$MODEL_SHA" --output ../my-scores
python -I -m evidence_lens.score_maps --scores ../my-scores --output ../my-maps
```

Substitute your calibrated input PNG, cameras, source self-render and six predicted views. Each prediction carries display-sRGB colour, accumulated alpha and expected camera-Z in metres. The scorer reads the five model files and produces continuous scores for five methods plus their low, high and unknown states. [Feature schema](model/feature_schema.json) · [Frozen operating thresholds](model/thresholds.json)

The readable source in `source/evidence_lens/` matches the installed wheel. The formal numerical report covers registered SHARP/gsplat predictions and independent Blender references. The model card explains its design and the selected scope identifies any corrections or withdrawals. [Inference source](source/evidence_lens/l1.py) · [Formal report](proofs/report-public.json) · [Release notes](RELEASE_NOTES.md) · [Use terms](docs/licenses.md)
