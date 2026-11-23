# Building and checking the model release

The builder takes separately sealed model, authored example, score and report receipts. It verifies the complete training and development rosters, original class counts, per-fold class counts, exported parameters, and the 12 comparison-run pins across the four held-out families. The model ZIP includes its CPU wheel, frozen model, threshold and fit proofs, original example, expected numerical scores, curated report tables and the committed release texts.

The release commit supplies `protocol/`, `configs/formal-freeze-v1.json`, the owner’s license files and `docs/releases/<version>/{MODEL_CARD.md,RELEASE_NOTES.md,README.md}`. The wheel's package version and the model release version are recorded separately. Build the wheel from the exact release commit into a directory separate from the checkout.

The owner records four material classes in `licenses/ASSET-SCOPE.json`. The following template illustrates a separate notice for published measurements, provenance and report files. The owner may instead choose the same committed document as one of the other classes. The research-record notice describes the released measurements, hashes and documentation. Referenced private training data and upstream images have their own provenance and terms.

```json
{
  "schema": "el-public-license-scope-1",
  "license_files": {
    "source": "LICENSE",
    "parameters": "licenses/MODEL-LICENSE.txt",
    "example": "licenses/EXAMPLE-LICENSE.txt",
    "research_records": "licenses/REPORT-LICENSE.txt"
  }
}
```

`release.json.license_scope` lists every file under its material class. Source covers the installed wheel, code and geometry specifications, feature layout and user documentation. Parameters cover model indices, fitted coefficients, preprocessing and thresholds. The example class covers the original plane files, demo manifest and expected output. Research records include `training/`, `proofs/`, `results/`, the method and formal freeze receipts, and the two release inventory files. The license and scope documents are identified as notices.

```sh
PYTHONPATH=src python scripts/release_model.py build \
  --store "$STORE" --method "$METHOD_FREEZE" --method-sha256 "$METHOD_SHA" \
  --version "$MODEL_VERSION" --commit "$RELEASE_COMMIT" --wheel "$WHEEL" \
  --example "$EXAMPLE_RUN=$EXAMPLE_MANIFEST_SHA" --example-source "$REFERENCE_RUN=$REFERENCE_MANIFEST_SHA" \
  --scores "$SCORE_RUN=$SCORE_MANIFEST_SHA" \
  --report "$REPORT_RUN=$REPORT_MANIFEST_SHA" --output "$RELEASE_DIR"
```

The example producer uses a single authored input PNG and its camera calibration. Its provenance and recorded inputs must both identify the original scene-render parent. The builder verifies that parent’s native Blender scene and accepted validation receipt. The recorded example scorer must use exactly that example and the frozen threshold run. Both public PNGs are decoded and checked for approved numerical image chunks. Each NumPy archive is restricted to the declared numerical arrays, dtypes and camera dimensions.

`proofs/report-public.json` and `proofs/example-source.json` are projections. Their origins name the sealed run, its manifest, the source path and the source file’s SHA-256. The report projection includes the original B0 and Q1 comparison receipt hashes for each family. A successful Q1 comparison must identify that family’s exact B0 learned-score, prediction and rule-score run/manifest pairs. Every threshold in `family-results.csv` must match the frozen model’s operating point for that method. L0 uses 0.5.

The file inventory in `release.json` records each member's origin, byte count and SHA-256. The comparison digest uses the file roster and bytes at the original method commit. The separate wheel digest covers the distribution built from the release commit, including its Git-verified package code, README and source license. The release command returns the SHA-256 digests of the wheel, model-release manifest, demo manifest and checksum list.

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
