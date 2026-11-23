# Build the private release candidate

The build runs from a source checkout with Python 3.11+, uv, NumPy and Pillow. The operator needs the selected sealed run envelopes and payloads in a local research store. The separate candidate directory contains the ZIP, its SHA-256 sidecar and a compact build receipt. Recipients run the scorer from the extracted ZIP. [Candidate guide](README.md)

## Select source and scope

Use the exact Git commit to be distributed. The builder checks its own scripts and every wheel module against this commit. Set the variables below from the selected original records. Each run pin has the form `run_id=manifest_sha256`. The example is the authored plane derivation. Its source is the independently validated Blender reference, and its score must have the chosen model and plane derivation as its direct parents.

- `STORE` is the local directory containing `runs/`. `METHOD` is its frozen method JSON. `METHOD_SHA` is the original method digest.
- `COMMIT` is the full 40-character release source commit.
- `EXAMPLE`, `EXAMPLE_SOURCE`, `SCORES` and `REPORT` are explicit original run pins.
- `SCOPE` points to the selected public-safe private-candidate JSON below. `SCOPE_SHA` is the independently agreed digest of its bytes.

For a new fit, use its method freeze and the matching deployment score and formal report. The packager checks the historically registered comparison protocol against the method commit. Its content allowlist includes numerical formal records and the authored demonstration. The formal PNG board, SHARP arrays, PLY files, checkpoint and private archive paths belong in the separate owner archive.

```json
{
  "schema": "el-prerelease-scope-1",
  "version": "v0.1.1-rc.2",
  "publication_status": "private_candidate",
  "preview_class": "<selected preview class>",
  "formal_evidence_status": "<selected evidence status>",
  "model_id": "<frozen threshold-selection ID>",
  "model_index_sha256": "<model/model.json SHA-256>",
  "method_freeze_sha256": "<method SHA-256>",
  "example_scores": {"run_id": "<score ID>", "manifest_sha256": "<manifest SHA-256>"},
  "report": {"run_id": "<report ID>", "manifest_sha256": "<manifest SHA-256>"},
  "diagnostics": [{
    "run_id": "<successful diagnostic ID>",
    "manifest_sha256": "<manifest SHA-256>",
    "path": "payload/<recorded-result>.json",
    "sha256": "<original result SHA-256>"
  }],
  "finding": "<supported finding and applicable scope>",
  "limitations": ["<remaining limitation>"]
}
```

Use `technical_preview` or `experimental_research_preview`. Evidence states are `original_frozen_study_rechecked`, `historical_benchmark_regression`, `original_study_under_review` and `historical_results_withdrawn`. Research previews require rechecked original evidence or an explicitly historical regression. The private scope accepts multiple diagnostics and verifies each against its successful sealed producer. The resulting archive carries the selected text and content identifiers. It can be inspected without access to any private diagnostic file.

## Build and reproduce

```sh
python3 scripts/build_prerelease.py --version v0.1.1-rc.2 --commit "$COMMIT" \
  --store "$STORE" --method "$METHOD" --method-sha256 "$METHOD_SHA" \
  --scope "$SCOPE" --scope-sha256 "$SCOPE_SHA" --example "$EXAMPLE" \
  --example-source "$EXAMPLE_SOURCE" --scores "$SCORES" --report "$REPORT" \
  --output candidate

python3 scripts/build_prerelease.py --version v0.1.1-rc.2 --commit "$COMMIT" \
  --store "$STORE" --method "$METHOD" --method-sha256 "$METHOD_SHA" \
  --scope "$SCOPE" --scope-sha256 "$SCOPE_SHA" --example "$EXAMPLE" \
  --example-source "$EXAMPLE_SOURCE" --scores "$SCORES" --report "$REPORT" \
  --rebuild-of candidate/candidate.json --output candidate-rebuild
```

Add `--offline` to use locally cached wheel-building tools. The builder uses the commit time for wheel timestamps and fixed ZIP member timestamps. The rebuild compares the resulting wheel, selected inputs, human-readable scope, computational tools and full ZIP digest to the earlier receipt. The build and rebuild need the same backend and compression tools for byte-for-byte reproducibility.

## Extract and grade independently

```sh
cd candidate
shasum -a 256 -c evidence-lens-v0.1.1-rc.2-cpu.zip.sha256
python3 -m zipfile -e evidence-lens-v0.1.1-rc.2-cpu.zip ../candidate-extracted
cd ../candidate-extracted
python3 -I smoke.py --output ../candidate-cpu-grade
```

For an owner smoke, add `--release-sha256` with the value in the independently recorded candidate receipt. Pass `--archive ../candidate/evidence-lens-v0.1.1-rc.2-cpu.zip --archive-sha256 "$ZIP_SHA"` to bind every ZIP member and its container checksum to the observed platform and results. Supply `--wheelhouse PATH` for an offline pip install or `--offline-uv` for an existing uv cache. Preserve the emitted `smoke.json` and its referenced files with the run evidence. An independent public download requires its own measured receipt after publication is authorized.

To record the owner check as a new study operation, prepare a JSON config with the release tag, model ID, ZIP and manifest digests, and the intended smoke environment. The existing recorder creates the run ID before execution. Use the selected diagnostic run as another `--parent` where applicable.

```sh
python3 -m evidence_lens.recorder --root "$STORE" --source-root "$CHECKOUT" \
  --config "$GRADE_CONFIG" --kind archive_verify --experiment phase1-rc-cpu-grade \
  --trial v0.1.1-rc.2-private-cpu --parent "$REPORT" --parent "$SCORES" \
  --input "release=$EXTRACTED" --input "zip=$ZIP" -- \
  '{python}' -I '{input:release}/smoke.py' --bundle '{input:release}' \
  --archive '{input:zip}' --archive-sha256 "$ZIP_SHA" --release-sha256 "$RELEASE_SHA" \
  --output '{out}/cpu'
```
