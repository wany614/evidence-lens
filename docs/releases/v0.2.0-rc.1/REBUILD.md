# Reproduce scoring and verify the release

The standalone bundle includes complete trained CPU models. The supplied plane can be scored after extracting the ZIP, using the command in the README. To score another calibrated prediction, use either visual model’s `score-native.py` with its independently supplied SHA-256 for `model.json`. The source image, cameras and seven predicted views are ordinary local files. Fresh output directories prevent inadvertent replacement of a previous result.

```sh
python3 -I -B models/L2-vision-ft/score-native.py --model-sha256 "$MODEL_SHA" --input input.png --cameras cameras.json --predictions predictions --view all --output ../native-risk
```

For L2-frozen select its directory and model index hash. Run within an environment containing the pinned dependencies in `requirements.txt`. `score-native.py` performs camera projection, numerical feature extraction, full-backbone inference, thresholding and serialization. The compact `infer.py` entry accepts explicitly supplied native RGB and deployment-feature arrays. Both work without evaluation labels.

## Original-study reconstruction

`release.json` identifies the public source commit, frozen scientific method, completed quantitative report, independently recorded example and each selected model. The training protocols, model artifacts and public numerical reports are distributed in the bundle. The original sealed private training/evaluation records and provider-dependent rendered outputs remain in the owner research archive. An independent project can use the public acquisition recipe to record new inputs and receive its own run identities. Exact reconstruction of this archive uses the recorded inputs from its three supplied parents.

From the pinned public checkout use a Python environment with the project recording dependencies, PyTorch and safetensors. The original author uses the sealed research store and public A example ZIP to run the build. The `--method`, `--report` and `--example-grade` arguments take `RUN_ID=MANIFEST_SHA256`. The builder checks the real eight-run design, both complete FP32 exports, the current cohort opening, every numerical table row and the authored example. It reads its own source files from the specified commit.

```sh
python3 scripts/vision_release.py build --version v0.2.0-rc.1 --repository . --commit "$PUBLIC_COMMIT" --store "$STORE" --shelf "$SHELF" --method "$METHOD_PIN" --report "$REPORT_PIN" --example-grade "$EXAMPLE_PIN" --rc3-archive "$PUBLIC_A_ZIP" --output output-one
python3 scripts/vision_release.py build --version v0.2.0-rc.1 --repository . --commit "$PUBLIC_COMMIT" --store "$STORE" --shelf "$SHELF" --method "$METHOD_PIN" --report "$REPORT_PIN" --example-grade "$EXAMPLE_PIN" --rc3-archive "$PUBLIC_A_ZIP" --output output-two --rebuild-of output-one
```

The release directory contains the public ZIP, checksum list and build receipt. `private-sources.json` is an owner-side inventory. An independently verified original DINOv2 checkpoint may be supplied with `--base-asset PATH` to emit a separate, exact upstream file. Rebuilding requires the same option and input each time. The full public models already contain their backbones.
