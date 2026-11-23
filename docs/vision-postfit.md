# Stage B development, opening and confirmation

The training catalog registers 28 training families, eight development families, eight new confirmation families, and a separate historical regression on four previously studied families. The original five methods use the same training and development sources as the two visual branches. Each family's six target views retain their native 512 × 512 grid. The study uses families as its independent units.

## Development evidence

| Stage | Recorder run | Manifest SHA-256 | Output SHA-256 |
| --- | --- | --- | --- |
| Combined training, 168 views and 688,128 sampled rows | 20261123T130016909864Z-261c749e33704daf | ea0d00e06d3510924e4f98465b93ae02d3b56eafea1f23851785ccd053bf99ba | 01066fd8e8e2ecf2e4439659c8e758bca079c6dcf66454547459f39e8482f879 |
| Train-only preprocessing | 20261123T130122000887Z-174a98bd029f46ca | 0175337a3186669f428b304e411db80b6e146b9383cde8f1d20c6870eacd50d6 | 1023c3747ca62ec6b94752f4e4660fcec698a7632e8c5a8301098e7753732ef4 |
| Combined development, 48 views and 12,582,912 evaluable pixels | 20261123T130645905832Z-4584b3b8abbf4c59 | acf2ff3dc632548925b54ea9dacf0c8d38f3b2df3b331879dbbcb973a9f37c3c | d1ee59da321d77b6777ea97d742ae0b73cada3a6815db46c3553287d19127ffe |
| Logistic fit, both models selected C = 0.1 | 20261123T130437328225Z-b36e63e03b88475b | 1cca54bb3876f56019efb819370345ea54453831280bdd93ce1feb6f3e4daf0d | 097829b442e6b405bed4eaf46edcb85023064daf80ba8d5d09af5ef55ddfe46d |
| Five original native score maps for every development target | 20261123T130648900916Z-70c27fbec6b04e67 | 1b5dc50145d4d50be1037accb43d339e40784fa6aed70be36a24d8b61abf1066 | 7609e2054a4c9b1c65ec2120fed1726e12c98c49233a16e06bfc049b2543b092 |
| Five original development operating points | 20261123T130659497166Z-3099dd87e0d94206 | 98dabd61812a6877d42cf46451719fb3ca46bc7fa0b8e974cf4950b41dfca731 | 232595d6de6d143b6a562623845d59844dc3059aaf084edba606a779ae203fe3 |

The four original development families have a separate engineering regression against their earlier frozen deployment. Over 24 views, 31,457,256 finite score values matched bit for bit. All five state maps agreed, and all four development-selected thresholds were reproduced. The recorder run was `20261123T124038884623Z-94d1137ae3da491a`, with manifest `cc0c78c4b8952eecc9524d3f23bc8fec58de1361bd1b231bf0a81425877a9334` and readback SHA-256 `5be4f84ab34d20e7d8583d717b7e47cf50494d92680de5b53b9af0e8c3c42796`. This regression used the original model. The table above records the newly fitted baselines.

An independent replay on all eight current development families compared 480 defined AP values across each original method’s own finite domain and the five-method common domain. The largest absolute difference between the grouped reducer and scikit-learn was 2.22 × 10⁻¹⁶. An independent per-family histogram calculation reproduced all five strict thresholds and their coverages. The recorder run was `20261123T131443217184Z-0d6f62ab5d23456d`, manifest `fa7456507bd9ae0a3e22f8b474de1a232a95ad5b75c2411db4881903fd058d9b`, and readback SHA-256 `fc8d533ca3b924966eb034e2063832f3468aaa62dfd5a7d70488b4493536e1ef`.

## Run the development freeze

Run these commands from the research checkout with its experiment environment. Each stage records the source revision, declared inputs and parents, and computes its output into a fresh run. The command reuses a successful matching selection after an interruption. A recorded failed attempt can be retried with `--retry STAGE=FAILED_RUN_ID`. Keep the original worker source for an identical retry.

```sh
PYTHONPATH=src .venv-train/bin/python scripts/vision_postfit.py finalize-lr
PYTHONPATH=src .venv-train/bin/python scripts/vision_postfit.py development \
  --candidate-roster "$CANDIDATE_ROSTER" "$CANDIDATE_SHA"
PYTHONPATH=src .venv-train/bin/python scripts/vision_postfit.py freeze \
  --candidate-roster "$CANDIDATE_ROSTER" "$CANDIDATE_SHA" \
  --cost-frozen "$FROZEN_COST_RUN_AND_MANIFEST" --cost-ft "$FT_COST_RUN_AND_MANIFEST" \
  --cost-family "$REGISTERED_DEVELOPMENT_FAMILY"
```

The candidate roster has schema `el-vision-candidate-roster-1`, stage `primary`, and eight explicit entries in `runs`, each with `run_id` and `manifest_sha256`. A cost pin has form `RUN_ID=MANIFEST_SHA256`. Each primary model needs a separate recorded Mac CPU benchmark on the same B0 development family with six native views, one cold process and three warm passes. The freeze checks the original target and camera hashes and the measured RSS and latency. Minor CUDA-to-CPU score differences are recorded with support counts, ULP differences and threshold crossings.

The exporter preserves the trained logistic coefficients, 26-column schema and preprocessing. Its deployment package contains `model/model.json`, `model/feature_schema.json`, `model/L1-full.json`, `model/L1-geometry.json` and `model/thresholds.json`. Reapplying the exported package to all 48 development targets must reproduce the original threshold choices. The neural package contains its complete weights in `package/weights.safetensors`, the architecture, base identity, preprocessing and package index. Each primary neural package retains its seed-A operating point.

The formal worker source ZIP has SHA-256 `19998e7fd69a6ba305633b1c62d5da0d9f09fe1ef5e3cb7b6177f7a9f54cc8b9`. The freeze checks the source index in each of the eight formal training receipts against SHA-256 `ff0230c2edbd66e1e90c9fe9545cfb129b6379449e76ce0aa2890d8607020810`. The freeze hashes each of the 181 individual tensors directly from each safetensors member and compares it with the training table.

The independent replay reads all eight candidates' chosen-epoch score maps saved on the original training device. It compares scikit-learn's tied-score average precision against the recorded per-view results, checks that early stopping selected the earliest maximum, and rebuilds coverage from per-family score histograms. Exact integer counts and rational family coverages resolve a tie at the target of 0.8. The primary replay includes each method's own supported domain, the original five-method common domain for those five, and the seven-method common domain for all seven. A one-class or empty view has undefined AP. The freeze also verifies tensor-wise CPU export, the effective precision settings and the eight task-gradient witnesses.

Three paired seeds enter the registered default decision. Fine-tuning becomes the default only if it exceeds the refitted full logistic model for each seed, the mean paired AP gain over the frozen encoder exceeds the sample standard deviation of the three paired gains, and its full native Mac CPU cost meets both the five-second hot p95 and 2 GiB RSS limits. The other registered outcome is L1-full. Both neural primary models enter confirmation under either outcome.

## Run the fixed evaluation

```sh
PYTHONPATH=src .venv-train/bin/python scripts/vision_postfit.py open
PYTHONPATH=src .venv-train/bin/python scripts/vision_confirm.py new \
  --method "$METHOD_RUN_AND_MANIFEST" --method-sha256 "$METHOD_SHA" \
  --selection-registry "$ACQUISITION_REGISTRY" "$REGISTRY_SHA"
PYTHONPATH=src .venv-train/bin/python scripts/vision_confirm.py historical \
  --method "$METHOD_RUN_AND_MANIFEST" --method-sha256 "$METHOD_SHA"
PYTHONPATH=src .venv-train/bin/python scripts/vision_confirm.py report \
  --method "$METHOD_RUN_AND_MANIFEST" --method-sha256 "$METHOD_SHA"
```

`open` validates the successful joint recorder report and its original model parents, then writes the one catalog-specific opening marker. Keep the selected model bundles in the active store for this operation. The acquisition controller supplies the 24 terminal selections and a byte-pinned registry. The new cohort command compares the registry against every opened trial selection. The historical command binds the twelve family/condition inputs to their previously reported execution chain. It records a distinct transport, three score runs, an evaluation and a private board index for each cohort. The neural score workers use the CPU vision interpreter from `.venv-vision-cpu/bin/python` or the explicit `--vision-python` path. They consume the predicted RGB and deployable features. Reference RGB and labels enter the separate evaluation and private display processes.

Each private board shows v01 at fixed error, risk and decision scales. The board register also records target RGB and seven-method score changes over all six views for same-family R1 and Q1 versus B0. An unavailable condition source leaves its input panel unavailable. Failed views retain the known reference-valid population, E of zero, the pipeline disposition and their planned positions. Explicit engineering failures can be added to the final report using repeated `--engineering RUN_ID=MANIFEST_SHA256` arguments.

The reading copy is `payload/study.md`. The final recorder report has experiment ID `stage2-vision-confirmation-report`. Its `payload/public-numerical` directory contains the five files below and a byte-count and SHA-256 manifest.

| Public file | Content |
| --- | --- |
| `study-summary.json` | Separate new-eight and historical-four summaries, the two groups of four new families, family risk–coverage curves and matched R1/Q1 numerical responses |
| `family-metrics.csv` | One row for each family, condition and method, including raw masks, AP eligibility, missing outputs, rates and frozen thresholds |
| `paired-changes.csv` | Matched family comparisons and equal-family changes, with denominators |
| `execution-costs.json` | Recorded training, CPU inference, acquisition, resources, scientific attempts and separately pinned engineering incidents |
| `selected-method.json` | Frozen default, all thresholds, package hashes, paired development results and cross-device readbacks |

Score maps, Blender references, condition images and the fixed v01 panels remain in the private numerical and display runs. The public tables carry the source hashes and integer denominators needed to interpret their scope.
