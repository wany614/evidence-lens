# Rebuild and verify the Stage C archives

A source checkout and the public ZIPs are enough to validate packaged member hashes, run arbitrary valid native inputs and replay the authorized offset-bell-cage example. The independent raw reference and implementation source included in the companion provide a separate reference-render reconstruction. The numerical archive includes original study, threshold, model and parent identities.

Exact historical packaging additionally needs the immutable recorder originals of the three explicit release pins and their referenced members. From the recorded shipping commit run the following twice to fresh output directories:

```sh
python3 -I -B scripts/stage3_release.py --version v0.3.0-rc.1 --commit "$COMMIT" --repository . \
  --store "$STORE" --shelf "$SHELF" --method "$METHOD_PIN" --report "$NUMERICAL_PIN" \
  --example "$EXAMPLE_PIN" --output build-a
python3 -I -B scripts/stage3_release.py --version v0.3.0-rc.1 --commit "$COMMIT" --repository . \
  --store "$STORE" --shelf "$SHELF" --method "$METHOD_PIN" --report "$NUMERICAL_PIN" \
  --example "$EXAMPLE_PIN" --rebuild-of build-a --output build-b
```

The builder imports the scientific implementation from that Git tree, authenticates the actual recorded method opening and held-out evaluation, verifies each positive-listed public original and compares every ZIP member with its source. candidate.json includes asset and index hashes. private-sources.json is an owner-side audit of consumed original members and stays with the build records.

The public CPU ZIP contains the corrected scientific worker in `evidence/formal-worker.zip` and the original Windows worker in `evidence/formal-worker-03.zip`. They differ in the dictionary key order used to write a model index. The first frozen seed 1729 completed 288 optimizer updates in Windows before its package loader rejected that index. A separate Mac derivation preserved the original weights, all 48 selected all-source maps and the strict development threshold. The derivation ran zero additional optimizer updates. Its one-source and M4 two-source maps were calculated on Mac CPU. The full-window numerical comparison and the original and corrected source identities are in `evidence/MS-frozen/recovery.json`. To repeat the tensor comparison for either released neural model, copy its `evidence/<method>` directory to a new work directory, then copy `models/<method>` there as `package`. Unpack the formal worker into a separate directory. Download the public Stage B CPU package and take its `models/L2-vision-ft` directory as `B_PACKAGE`. The C model records the required B safetensors SHA-256, f372bd34faa1b345ae2486fe6c96aea00d8d9ac5749b0b2c237f9ed08d50f228. With the dependencies from the source repository's neural verification guide, run:

```sh
python3 -I -B formal-worker/scripts/multi_risk_worker.py verify-updates \
  --B-index "$B_PACKAGE/model.json" --B-weights "$B_PACKAGE/weights.safetensors" \
  --training "$WORK_DIR" --training-sha256 "$TRAINING_JSON_SHA" --output new-tensor-proof
```

Compare the resulting verified-updates.json with the per-model original in independent-proof/. The evidence uses all 181 exported tensors and the actual eight before-weight-decay minibatch gradient witnesses. [Published Stage B release](https://github.com/wany614/evidence-lens/releases/tag/v0.2.0-rc.1)

The research repository documents separately named independent data generation, calibration, training and method freezing. Repeating these operations creates new recorder identities. Releasing a separately run comparison requires its own complete research evidence and version.
