# Rebuild and verify the Stage C archives

A source checkout and the public ZIPs are enough to validate packaged member hashes, run arbitrary valid native inputs and replay the authorized offset-bell-cage example. The independent raw reference and implementation source included in the companion provide a separate reference-render reconstruction. The numerical archive includes original study, threshold, model and parent identities.

Exact historical packaging additionally needs the immutable recorder originals of the three release pins and their referenced members. Set `STORE` and `SHELF` to the original research and verified archive locations. From the tagged source, run both builds into fresh output directories.

```sh
COMMIT=$(git rev-parse 'v0.3.0-rc.1^{commit}')
METHOD_PIN=20261123T165423389866Z-923b49318fd54ce9=c43bd48eb13df71b51bbecb24903370ee56be197f4493affaaad3b15b2f432c1
EXAMPLE_PIN=20261123T172802245455Z-0e7b98d153dd4804=e4e065852c00eabb41e86f9c292ede5d395d7f38d7ee271063249718bfa6f07a
NUMERICAL_PIN=20261123T172732453599Z-e27a4fffd04b4e04=eb02c1f0102639f413aa59b07e42d4c8539fe79e18b26a5917339da6d1504e35
python3 -I -B scripts/stage3_release.py --version v0.3.0-rc.1 --commit "$COMMIT" --repository . \
  --store "$STORE" --shelf "$SHELF" --method "$METHOD_PIN" --report "$NUMERICAL_PIN" \
  --example "$EXAMPLE_PIN" --output build-a
python3 -I -B scripts/stage3_release.py --version v0.3.0-rc.1 --commit "$COMMIT" --repository . \
  --store "$STORE" --shelf "$SHELF" --method "$METHOD_PIN" --report "$NUMERICAL_PIN" \
  --example "$EXAMPLE_PIN" --rebuild-of build-a --output build-b
```

The builder imports the scientific implementation from that Git tree, authenticates the actual recorded method opening and held-out evaluation, verifies each positive-listed public original and compares every ZIP member with its source. candidate.json includes asset and index hashes. private-sources.json is an owner-side audit of consumed original members and stays with the build records.

The example allowlist names 477 original members. Its first export stopped when the two-source alias had no second process receipt. The corrected exporter checked that both entries point to the same score file and used the existing timing receipt. The numerical allowlist names 21 members, including the joint report, its original tables and their manifest. [Publication member selector](https://github.com/wany614/evidence-lens/blob/v0.3.0-rc.1/scripts/stage3_public_members.py)

A first local installation check, run `20261123T173017885387Z-9587690934754949`, stopped before inference. The example and numerical archives contained license notices that their indexes omitted. The published builder records the byte count, hash and tagged source of every notice as well as the research assets. A clean install must verify the entire ZIP inventory before running the example.

The public CPU ZIP contains the corrected scientific worker in `evidence/formal-worker.zip` and the original Windows worker in `evidence/formal-worker-03.zip`. They differ in the dictionary key order used to write a model index. The first frozen seed 1729 completed 288 optimizer updates in Windows before its package loader rejected that index. A separate Mac derivation preserved the original weights, all 48 selected all-source maps and the strict development threshold. The derivation ran zero additional optimizer updates. Its one-source and M4 two-source maps were calculated on Mac CPU. The full-window numerical comparison and the original and corrected source identities are in `evidence/MS-frozen/recovery.json`. To repeat the tensor comparison for either released neural model, copy its `evidence/<method>` directory to a new work directory, then copy `models/<method>` there as `package`. Unpack the formal worker into a separate directory. Download the public Stage B CPU package and take its `models/L2-vision-ft` directory as `B_PACKAGE`. The C model records the required B safetensors SHA-256, f372bd34faa1b345ae2486fe6c96aea00d8d9ac5749b0b2c237f9ed08d50f228. With the dependencies from the source repository's neural verification guide, run:

```sh
python3 -I -B formal-worker/scripts/multi_risk_worker.py verify-updates \
  --B-index "$B_PACKAGE/model.json" --B-weights "$B_PACKAGE/weights.safetensors" \
  --training "$WORK_DIR" --training-sha256 "$TRAINING_JSON_SHA" --output new-tensor-proof
```

Compare the resulting verified-updates.json with the per-model original in independent-proof/. The evidence uses all 181 exported tensors and the actual eight before-weight-decay minibatch gradient witnesses. [Published Stage B release](https://github.com/wany614/evidence-lens/releases/tag/v0.2.0-rc.1)

The research repository documents separately named independent data generation, calibration, training and method freezing. Repeating these operations creates new recorder identities. Releasing a separately run comparison requires its own complete research evidence and version.
