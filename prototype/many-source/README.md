# Evidence Lens many-source prototype

This package ranks the risk of photometric error in target renders from a registered static Gaussian scene. It depends on Evidence Lens 0.5.0 and the frozen 378 × 252 analysis grid. A bank carries the reconstruction inputs (R), the permitted source inventory (A), the independent target camera IDs (H) and the fixed coordinate frame. Each result records the source IDs actually queried (Q).

Install the Stage A wheel and this project in the same Python environment. The optional `vision` extra and the separately distributed Stage C model directory enable real MS-ft Q4 transfer.

```sh
python -m pip install /path/to/evidence_lens-0.5.0-py3-none-any.whl ./prototype/many-source
evidence-lens-bank inspect --bank scene
evidence-lens-bank select --bank scene --target h00 --strategy coverage --budget 16
evidence-lens-bank score --bank scene --target h00 --budget 16 --model trained-head.json --out scored
evidence-lens-bank audit-bank --bank scene --target h00 --budget 16 --model trained-head.json --out audited
evidence-lens-bank report --scores scored --out local-report.html
```

The training release supplies the fitted NumPy head. `--model-sha256` binds a specific model file. To run the fixed coverage, alpha/missing or photometric residual rule independently, select `--rule coverage`, `--rule alpha_missing` or `--rule residual`. The score and audit commands emit `scores.json`, `scores.npz` and an offline HTML report. Click a pixel in a figure to inspect its source witnesses. Each report records the model, scope, true source IDs, unavailable material, source comparisons and byte counts. A higher rank means greater estimated RGB error. The new head uses no operating threshold.

## Preparing and inspecting a bank

The external checkpoint adapter produces `bank.json` in the `el-many-source-bank-1` format. Each source has its original photo hash, an analysis-grid RGB8 PNG and a same-checkpoint self-render. Each target entry contains the native predicted FP32 display RGB, opacity and opacity-normalized positive optical-axis Z. Cameras use integer pixel centres, OpenCV camera-to-world transforms and the versioned Stage A crop and K transformations. Every byte reference includes the relative path, size and SHA-256. Frame anchors and the world-unit transform belong to the reconstruction and apply to every query. The unit field may specify metres or scene units.

`inspect` checks metadata, grants, file availability, exact duplicates and close camera poses. Equal original hashes or byte-identical prepared PNGs share one query identity. Nearby distinct photos retain separate IDs. Camera centres within 2% of the median positive rig separation and optical axes within two degrees are reported as near duplicates. Add `--verify-members` to explicitly decode and verify the permitted evidence. `import --bank scene --out copy` makes a portable copy of the declared deployment members. A bare PLY receives a structural inspection and a list of missing inputs. Complete scoring requires verified reconstruction membership and expected-depth semantics.

## Selection and bounded computation

The fixed strategy begins with the camera closest to the rest of the rig and selects diverse additional viewpoints. The nearest strategy combines centre distance in rig units with optical-axis separation. Coverage selection samples one evaluable target-Z point nearest the centre of each cell in a 12 × 8 grid. It projects the points into candidate cameras and rewards coverage of less-observed cells and viewpoint diversity. All policies resolve utility ties by source ID. The selector reads source camera metadata. Only coverage selection decodes the target's Z and alpha.

The stream scans each unique authorized source in its registered order once. It projects target points, checks source opacity and expected camera-Z, and accumulates counts, first source witnesses, residual moments and extrema on fixed-size target arrays. A comparable sample requires valid positive depth and opacity of at least 0.01 at both pixels. Depth support additionally requires both opacities of at least 0.5 and relative depth residual at most 0.10. A supported photo disagreement above 0.10 counts as colour conflict. A valid target may receive a rank from missingness features even when no selected source is comparable.

`features --bank scene --target h00 --budgets 4 8 16 32 all --out features` writes one feature array per distinct actual budget. A clamped budget records its requested aliases. Use `evidence-lens-bank schema` to print all 25 numeric definitions and 19 missing indicators. Source cache capacity defaults to 128 MiB and accepts 0 through 512 MiB. The cache admits immutable source arrays authenticated against camera, bank, source and file identities. Selector costs, actual RGB and depth decoding, projected source counts and pixel comparisons are recorded separately.

A finite query reports its own evidence. `audit-bank` explicitly extends the scan through the unique permitted library, records the full cost separately and identifies an actual source that covers each subset gap. Claims of zero comparable evidence across the accessible library require a complete audit. Missing source files and unevaluable target geometry have separate states. All visibility and conflict results here are conditional on the registered predictions.

## Array API and independent evaluation

```python
from evidence_lens_many import Bank, select_sources, stream_features, simple_ranks

bank = Bank("scene", cache_bytes=64 << 20)
plan = select_sources(bank, bank.target_ids[0], strategy="coverage")
for step in stream_features(bank, bank.target_ids[0], plan, budgets=(4, 8, 16, 32, "all")):
    ranks = simple_ranks(step)
    print(step.actual_k, step.queried_source_ids, step.comparable_count.max())
```

`score_lr(step, model)` applies training-set imputation and standardization followed by an FP64 linear transform and a float32 sigmoid rank. `legacy_q4` runs the original MS-L1 or MS-ft model on exactly the four supplied source IDs and the fixed scene frame. It retains the released threshold and records the external transfer status. The evidence-and-rank arrays include finite status, reason, comparable count, depth and colour conflicts, and the first actual source witness. Witness values index the canonical source roster. A value of −1 means no witness.

The evaluator is a separate command. It opens an explicitly supplied independent measurement and scores the supplied predictions on their declared pixel domain.

```sh
evidence-lens-bank evaluate --prediction native-target.npz --reference held-out.png --scores scored/scores.npz --out evaluation.json
```

The Stage A baseline identity is `sha256:6e8e837afa0818cdf8c100b494672b46c10bdbd22e522f388983543f0a5b0f23`. The wheel supplies the prototype code. Model release identities, training inputs, experimental results and dataset permissions are tracked in their respective release receipts.
