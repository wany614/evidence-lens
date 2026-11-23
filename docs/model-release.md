# Building and checking the model release

The builder takes separately sealed model, authored example, score and report receipts. It verifies the complete training and development rosters, original class counts, per-fold class counts, exported parameters, and the 12 comparison-run pins across the four held-out families. The model ZIP includes its CPU wheel, frozen model, threshold and fit proofs, original example, expected numerical scores, curated report tables and the committed release texts.

The release commit supplies `protocol/`, `configs/formal-freeze-v1.json`, `LICENSE`, `licenses/MODEL-LICENSE.txt`, `licenses/EXAMPLE-LICENSE.txt`, and `docs/releases/<version>/{MODEL_CARD.md,RELEASE_NOTES.md,README.md}`. The source/parameter/example terms come from the owner-approved committed files. The wheel's package version and the model release version are recorded separately. Build the wheel from the exact release commit into a directory separate from the checkout.

```sh
PYTHONPATH=src python scripts/release_model.py build \
  --store "$STORE" --method "$METHOD_FREEZE" --method-sha256 "$METHOD_SHA" \
  --version "$MODEL_VERSION" --commit "$RELEASE_COMMIT" --wheel "$WHEEL" \
  --example "$EXAMPLE_RUN=$EXAMPLE_MANIFEST_SHA" --example-source "$REFERENCE_RUN=$REFERENCE_MANIFEST_SHA" \
  --scores "$SCORE_RUN=$SCORE_MANIFEST_SHA" \
  --report "$REPORT_RUN=$REPORT_MANIFEST_SHA" --output "$RELEASE_DIR"
```

The example producer uses a single authored input PNG and its camera calibration. Its provenance and recorded inputs must both identify the original scene-render parent. The recorded example scorer must use exactly that example and the frozen threshold run. `proofs/report-public.json` carries the original private report receipt digest and the participating comparison-run and table hashes. The file inventory in `release.json` records each member's origin, byte count and SHA-256. Its module digest covers the wheel's Git-verified Python source. The release command also returns the SHA-256 digests of the wheel, model-release manifest, demo manifest and checksum list.

```sh
PYTHONPATH=src python scripts/release_model.py zip --folder "$RELEASE_DIR" \
  --release-sha256 "$RELEASE_SHA" --checksums-sha256 "$CHECKSUMS_SHA" --output "$MODEL_ZIP"
PYTHONPATH=src python scripts/release_model.py verify --folder "$RELEASE_DIR" \
  --archive "$MODEL_ZIP" --archive-sha256 "$ZIP_SHA" \
  --release-sha256 "$RELEASE_SHA" --checksums-sha256 "$CHECKSUMS_SHA"
```

The ZIP command uses fixed file order, timestamps and permissions, verifies every compressed member against the output directory and publishes `<name>.zip.sha256` beside the ZIP. A recipient can check the separately published container digest and then check the extracted members with `shasum -a 256 -c SHA256SUMS` or `sha256sum -c SHA256SUMS`. The Python verifier additionally checks the scientific and wheel identities. Its optional external pins connect these comparisons to the release originally announced by the publisher.

## Run the distributed example

In a new Python 3.11+ virtual environment, install the wheel supplied in `distribution/`. The installation declares NumPy and Pillow. Their wheels must be available from the package index or a local wheelhouse. Run the following commands from the extracted release, with the versioned wheel filename from `release.json` and the demo hash from the build or verified file inventory.

```sh
python -m pip install "distribution/$WHEEL_NAME"
python -I -m evidence_lens.l1_demo grade --bundle . \
  --manifest-sha256 "$DEMO_SHA" --output "$NEW_GRADE_DIR"
python -I -m evidence_lens.score_maps --scores "$NEW_GRADE_DIR/scores" \
  --method L1-full --view v01 --output "$NEW_MAP_DIR"
```

`grade.json` reports actual clean-environment package versions, numerical differences, mask/state agreement and the number of finite L1 example pixels. To score another compatible bundle, use `python -m evidence_lens.l1 --help`. [Deployable inputs and score fields](l1-training.md)
