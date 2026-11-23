# Reproduce scoring and verify the release

The standalone bundle includes complete trained CPU models. The supplied plane can be scored after extracting the ZIP, using the command in the README. To score another calibrated prediction, use either visual model’s `score-native.py` with its independently supplied SHA-256 for `model.json`. The source image, cameras and seven predicted views are ordinary local files. Fresh output directories prevent inadvertent replacement of a previous result.

`evidence/runtime-source.json` maps the ten deployed Python modules to both selected training exports, the exact formal worker ZIP and regular Git blobs in the shipping commit. The two short launchers are reproduced from the frozen exporter with the recorded text newline convention. Anonymous verification compares the runtime bytes with the trained model indexes, the frozen worker and GitHub's actual tagged tree.

```sh
python3 -I -B models/L2-vision-ft/score-native.py --model-sha256 "$MODEL_SHA" --input input.png --cameras cameras.json --predictions predictions --view all --output ../native-risk
```

For L2-frozen select its directory and model index hash. Run within an environment containing the pinned dependencies in `requirements.txt`. `score-native.py` performs camera projection, numerical feature extraction, full-backbone inference, thresholding and serialization. The compact `infer.py` entry accepts explicitly supplied native RGB and deployment-feature arrays. Both work without evaluation labels.

## Original-study reconstruction

`release.json` identifies the public source commit, frozen scientific method, completed quantitative report, independently recorded example and each selected model. The training protocols, model artifacts and public numerical reports are distributed in the bundle. The original sealed private training/evaluation records and provider-dependent rendered outputs remain in the owner research archive. An independent project can use the public acquisition recipe to record new inputs and receive its own run identities. Exact reconstruction of this archive uses the recorded inputs from the method, report, authored-example grade and formal-native grade.

`protocol/stage2-inference-parity-v1.json` gives the registered Mac-to-CUDA development comparison. Its hash and the two catalog hashes are checked against the sealed scientific method. The subsequent release readbacks compare fresh CPU inference with the recorded CPU outputs and report each finite-score difference and threshold crossing.

From the pinned public checkout use a Python environment with the project recording dependencies, PyTorch and safetensors. The original author uses the sealed research store and public A example ZIP to run the build. The `--method`, `--report`, `--example-grade` and `--native-grade` arguments take `RUN_ID=MANIFEST_SHA256`. The builder checks the real eight-run design, both complete FP32 exports, the current cohort opening, every numerical table row and the authored example. It reads its own source files from the specified commit.

```sh
python3 scripts/vision_release.py build --version v0.2.0-rc.1 --repository . --commit "$PUBLIC_COMMIT" --store "$STORE" --shelf "$SHELF" --method "$METHOD_PIN" --report "$REPORT_PIN" --example-grade "$EXAMPLE_PIN" --native-grade "$NATIVE_PIN" --rc3-archive "$PUBLIC_A_ZIP" --output output-one
python3 scripts/vision_release.py build --version v0.2.0-rc.1 --repository . --commit "$PUBLIC_COMMIT" --store "$STORE" --shelf "$SHELF" --method "$METHOD_PIN" --report "$REPORT_PIN" --example-grade "$EXAMPLE_PIN" --native-grade "$NATIVE_PIN" --rc3-archive "$PUBLIC_A_ZIP" --output output-two --rebuild-of output-one
```

The release directory contains the public ZIP, checksum list and build receipt. `private-sources.json` is an owner-side inventory. An independently verified original DINOv2 checkpoint may be supplied with `--base-asset PATH` to emit a separate, exact upstream file. Rebuilding requires the same option and input each time. The full public models already contain their backbones.

## Fixed formal native verification

The owner-side comparison uses B0 on the alphabetically first new development family, all six 512 × 512 target views, and three fitted model packages. L1-full is compared with the separately recorded five-method development arrays. The two visual methods use the recorded Mac cold-run maps for their own exact models. Each check covers 4,718,592 method/pixel coordinates. Finite support and strict threshold states must agree exactly. Maximum absolute tolerances are 0.000001 for L1-full and 0.0002 for either visual model. The result retains each view’s supported denominator, error and threshold-state disagreement count. A distinct process reads only the named source image, cameras, prediction metadata, seven predicted images/depth/alpha arrays and released models. Evaluation maps are opened by the comparison process.

From the study checkout and owner environment, prepare the private input packet, then seal the independent Mac CPU grade. The resulting run identity supplies `NATIVE_PIN` for both build commands above.

```sh
python3 scripts/vision_release_native.py prepare --store "$STORE" --shelf "$SHELF" --method "$METHOD_PIN" --catalog protocol/stage2-families-v1.json --output private-native-case
python3 scripts/vision_release_native.py record --case private-native-case --case-sha256 "$CASE_SHA" --store "$STORE" --shelf "$SHELF" --project . --python "$CPU_PYTHON" --state private-native-state
```

The complete case stays in the owner archive. Its public numerical receipt is `evidence/formal-native-validation.json`. The standalone verifier can also repeat the check against the downloaded ZIP wherever the owner makes the case available.

## Anonymous download

Use the verifier from the exact tagged public source checkout. Obtain the full release commit and candidate index SHA-256 from the producer’s independently verified build. The candidate index binds the CPU archive, its internal manifest, four distinct learned parameter assets, numerical attachment and the published release notes. The verifier anonymously reads GitHub's repository, release and Git tree, downloads each declared attachment and obtains the original 88,283,115-byte DINOv2 weights from the release mirror or the original Meta distribution. A fresh CPU environment runs the complete authorized plane example.

```sh
python3 -I -B scripts/verify_vision_public.py --tag v0.2.0-rc.1 --commit "$PUBLIC_COMMIT" --candidate-sha256 "$CANDIDATE_SHA" --output anonymous-receipt
```

An owner with the private formal case can include it in the same clean-environment run on Windows PowerShell.

```powershell
py -3.12 -I -B scripts/verify_vision_public.py --tag v0.2.0-rc.1 --commit $PublicCommit --candidate-sha256 $CandidateSha --native-case $PrivateCase --native-case-sha256 $CaseSha --output anonymous-windows
```

The public release assets are `candidate.json`, `RELEASE_NOTES.md`, `SHA256SUMS`, the CPU ZIP and the numerical ZIP. An optional separate copy of the original upstream checkpoint carries its official hash. Additional owner-approved attachments must be independently pinned with `--attachment NAME=SHA256`. The release body comes from `RELEASE_NOTES.md`, which records the actual scored cohorts, execution failures, chosen models and all primary artifact hashes. `private-sources.json` and the raw formal case are owner records. Anonymous readback emits `public-verification.json`, actual CPU output and, if requested, a local formal equivalence receipt. Failed verification leaves downloaded or partial bytes and a phase-specific failure marker in the new output directory.

The release downloads and dependency installation have separate evidence. A fresh installer profile disables inherited proxy and CA overrides, persistent caches, `.netrc`, keyring and pip configuration files. `installation.json` names the installed versions, the SHA-256 and origin of each selected dependency wheel, and each pip transaction report hash. Online installations use the public package index and the CPU Torch index on Linux and Windows. `--wheelhouse PATH` records local offline installation. The requirements file pins the direct packages. The transaction lists the transitive versions that pip actually resolved on that platform.

For full release acceptance, append `--require-full --attachment "$CONTROL_NAME=$CONTROL_ZIP_SHA" --public-case-asset "$CONTROL_NAME" --public-case-sha256 "$CASE_SHA"` to the anonymous verifier command. Supply the independently known ZIP and case manifest digests. The verifier extracts only the ten declared deployment inputs and `case.json`, runs all seven scoring methods on the six targets in the clean CPU environment, and writes `public-control/control-scores.json`. A separate process receives the Blender targets, recomputes the measurements, and checks the 512-pixel risk maps and fixed figures. This yields `verified_full`. An invocation that scores only the built-in 128-pixel example reports `verified_builtin_only`. `--require-full` rejects that narrower scope.

## Rebuild the separately licensed plane attachment

The owner verifies the two material exports and the raw evaluation with `vision_control_attachment.py materials`. The command authenticates the source of the registered Blender image and cameras, compares all six public reference arrays with their original Blender records and replays the raw measurement. The producer indexes are the sealed case export `20261123T134307201400Z-a5e16b2b6c7f4dd2`, reference export `20261123T134308150391Z-402d42333024458b` and independent raw evaluation `20261123T134826982683Z-1ad06f33c09d4541`. Each command argument combines its run ID and full manifest hash with `=`. The material inventory is an owner-side working product.

```sh
python3 scripts/vision_control_attachment.py materials --store "$STORE" --shelf "$SHELF" --case-export "$CASE_EXPORT" --reference-export "$REFERENCE_EXPORT" --evaluation "$RAW_EVALUATION" --output checked-control-inputs
python3 scripts/vision_control_attachment.py build --repository . --commit "$PUBLIC_COMMIT" --store "$STORE" --shelf "$SHELF" --case-export "$CASE_EXPORT" --reference-export "$REFERENCE_EXPORT" --evaluation "$RAW_EVALUATION" --materials-sha256 "$MATERIALS_SHA" --method "$METHOD_PIN" --scores "$CONTROL_SCORE_PIN" --scored-evaluation "$CONTROL_GRADE_PIN" --output attachment-one
```

Repeat the second command with a new `--output attachment-two` and `--rebuild-of attachment-one` to check both the ZIP and external receipt byte for byte. Each build also exercises the independent evaluator against all six actual released score maps. Public readers can verify the scored control against an extracted CPU bundle and fresh scores by running the following from the source checkout.

```sh
python3 -I -B scripts/vision_control_attachment.py grade-download --archive "$CONTROL_ZIP" --archive-sha256 "$CONTROL_ZIP_SHA" --case-sha256 "$CASE_SHA" --release "$CPU_BUNDLE/release.json" --release-sha256 "$RELEASE_SHA" --scores ../fresh-plane/scores --output checked-plane-evaluation
```

The readback compares the six native maps of every method, finite support, threshold states, the L0 reason codes, RGB/depth measurements and fixed image pixels. Full error counts and thresholds are retained on a numerical discrepancy.
