# Score a prepared reconstruction on CPU

[Open the original bell-cage report](../../examples/offset-bell-cage/bundle-report.html) to inspect six predicted views, continuous risk, threshold states and evidence availability. The page contains its images and works offline. Download the HTML and open it locally when browsing a source viewer that displays HTML as text.

The default **MS-L1** uses the released linear model and its original cutoff of **0.06428983435034752**. The wheel includes its coefficients and preprocessing. Python 3.11 or later, NumPy and Pillow are sufficient for the default path. A prepared bundle supplies the images, calibrated cameras and renderer outputs. The scoring command performs no reconstruction or rendering.

## Run the original example

From a source checkout, install with `python -m pip install .` in a fresh virtual environment. Alternatively, the portable example includes the universal wheel. Install that wheel from the unpacked directory, then run:

```sh
python -m pip install ./evidence_lens-0.4.0a3-py3-none-any.whl
evidence-lens validate --bundle bundle
evidence-lens score --bundle bundle --model MS-L1 --out scores
evidence-lens report --scores scores --out report.html
python grade.py --scores scores --expected expected --out grade.json
```

The output directories and report file must be new. A report for the four-source example contains four source images, six target predictions and a JSON download. Low, high and unknown counts sum to the complete target canvas. You can select individual target cameras, show all six, or use the per-view table to locate missing evidence. Scores rank potential reconstruction error. A score of 0.02 has no established interpretation as a 2% error probability on an external scene.

## Use an existing Case

Extract the original example ZIP from the Stage C release. Its four-source Case has the following file-byte identity:

```sh
evidence-lens adapt-legacy --case CASE/M4 \
  --case-sha256 8ab6d51ec5c348fc61121478631a7ead7a9299659337cc8a8a5811922ec1d598 \
  --input-domain "Original Blender offset bell cage, development control" --out prepared
evidence-lens validate --bundle prepared
evidence-lens score --bundle prepared --out scores-legacy
evidence-lens report --scores scores-legacy --out report-legacy.html
```

The adapter runs the original Case identity checks and copies the authenticated source PNGs and numerical source/target renders. It records their upstream origins. Retain the `bundle_sha256` emitted by adaptation and pass it to `validate` or `score` with `--bundle-sha256` when transferring between systems. Supply `--scores-sha256` to `report` to pin `scores.json`.

Some historical exporters omitted a renderer run identifier, generation receipt or prepared camera definition. The optional proof path reads sealed producer, deployment and calibration metadata. For a recovered Case, it additionally checks every selected numerical frame, checkpoint and camera-Z probe. The public bell-cage example has explicit camera and producer evidence and can be converted directly.

```sh
python scripts/bundle_recovery_proof.py --store ARCHIVE_ROOT \
  --deployment RUN_ID=MANIFEST_SHA256 --out proof
evidence-lens adapt-legacy --case CASE --case-sha256 CASE_SHA256 \
  --proof proof --proof-sha256 PROOF_JSON_SHA256 --out prepared-recovery
```

The proof collector resolves named parents recorded by the signed run. Its input is a local recorder archive containing those originals. Obtain the proof digest from the trusted collection run. For the checked DTU cases, the original rectification and metric-frame evidence supply the camera conventions and the conversion from millimetres, already applied upstream. The collector needs no target reference images.

## Run the released visual models

Install the optional visual dependencies and extract the original **CPU release ZIP** to a directory containing `release.json` and `models/`. The same prepared input supports the genuinely fine-tuned **MS-ft**, the frozen-encoder **MS-frozen**, and the single-source **B-L1-transfer**. The latter uses the bundle's declared primary photo. MS-L1, MS-frozen and MS-ft consume the declared source order. All four retain their published weights, preprocessing and thresholds.

```sh
python -m pip install '.[vision]'
evidence-lens score --bundle prepared --model MS-ft --models CPU_RELEASE --device cpu --out ft-scores
evidence-lens report --scores ft-scores --out ft-report.html
```

The visual model files are separate downloads. The Stage C CPU archive is 164,854,827 bytes including its scientific evidence. The weights in that archive are individually verified by model-index SHA-256. Its original `cpu.py run` and `replay` entry points remain available. [Release and model card](../releases/v0.3.0-rc.1/README.md)

## Raw outputs and scope

Each target has one `scores.npz` with float32 `risk`, boolean `valid`, uint8 `state`, and uint8 `unknown_reason`. The threshold convention is strict low when the finite score is less than the recorded cutoff, otherwise high. Unknown scores are NaN with state 255. The report verifies the raw arrays and the per-view counts before plotting them.

This interface accepts prepared, undistorted, native 256 × 256 data with two or four authorized source views. Additional target IDs are representable. Published numerical regression covers the original six-target design. Version 0.4.0b1 also accepts the named COLMAP and Gaussian asset profiles, with separate CPU inspection, CUDA rendering and optional empirical operating points. [Bundle v1 schema](bundle-v1.md) · [Measured compatibility](compatibility.md) · [Gaussian asset import and calibration](assets/README.md)
