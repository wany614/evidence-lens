# Layered louvers with a four-photo query

This authored development scene has 16 reconstruction photographs and six target cameras. Both reports use the same fitted B-CPU-LR head, predicted target `h00` and four selected photographs. Open either HTML file on your computer. Each file includes its maps and source-lookup data. Click a pixel to see the ID of its first comparable source. The audited report also identifies a source outside the selection wherever the full scan found comparable evidence missed by the four-source query.

- [Four-photo report](score/report.html) · [Source IDs, scope counts and model identity](score/scores.json)
- [Four-photo report with the 16-source audit](audit-bank/report.html) · [Audit counts, source IDs and actual work](audit-bank/scores.json)
- [Native file identities and reproduction archives](../../../docs/research/many-source-v1/deployment/deployment.json) · [CPU measurements and external-bank checks](../../../docs/research/many-source-v1/deployment/README.md)

The input archive `evidence-lens-b04-r16-example.zip` contains the portable deployment bank, the trained `model.json`, a per-file checksum list and frame provenance. The companion `evidence-lens-b04-r16-reports.zip` contains the HTML, JSON and NPZ for both queries. The NPZ preserves the FP32 rank, finite mask, pixel state, reason, counters and integer source indices used in the panels. The example contains deployment predictions for the target cameras. Independent target photographs are held by the evaluator.

Install the Evidence Lens 0.5.0 wheel and the many-source prototype wheel identified in the deployment receipt. From the extracted **input** archive, regenerate both reports with these commands.

```sh
evidence-lens-bank inspect --bank bank
evidence-lens-bank select --bank bank --target h00 --strategy coverage --budget 4
evidence-lens-bank score --bank bank --target h00 --strategy coverage --budget 4 --model model.json --out finite
evidence-lens-bank audit-bank --bank bank --target h00 --strategy coverage --budget 4 --model model.json --out audited
```

Read the rank as relative estimated RGB error. Its operating threshold is unset. Geometric comparability and the reported witnesses use the reconstruction's predicted camera depth. The public reports use the authored b04 material under its registered rights. [Data and rights](../../../docs/research/many-source-v1/rights.md)

The optional historical four-source transfer accepts `--legacy MS-L1 MS-ft --legacy-models /path/to/released/cpu` on `score` with `--budget 4`. It requires the `vision` extra and the original released MS-ft weights. The finite-query B-CPU-LR example above runs with the base CPU installation. [Historical model release](https://github.com/wany614/evidence-lens/releases/tag/v0.3.0-rc.1)

The authored scene, cameras and derived demonstration images are credited to wany614 under CC BY 4.0. The original prototype code and fitted logistic coefficients use Apache-2.0. The upstream trainer and historical transfer weights retain their respective notices. [Rights by component](../../../docs/research/many-source-v1/rights.md)
