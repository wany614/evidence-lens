# Comparing frozen scorers on the recovered reconstruction

Compare the four methods on `within_new` when the question is which method ranks errors on the current reconstruction. Each method sees the same candidate pixels and accepts the same number in each target view. Their rankings can select different pixels. The result below is one previously exposed physical object, DTU set6, M4, with six 256 × 256 target views and 393,216 reference-valid pixels.

| Candidate population | Candidates | Errors among candidates | Mean view prevalence | MS-L1 errors / accepted | MS-ft errors / accepted |
|---|---:|---:|---:|---:|---:|
| Current four methods (`within_new`) | 371,447 | 165,573 | 44.56% | 20,063 / 78,642 (25.51%) | 19,864 / 78,642 (25.26%) |
| Historical and current methods (`paired_old_new`) | 313,790 | 150,350 | 47.89% | 26,925 / 78,642 (34.24%) | 28,178 / 78,642 (35.83%) |

The primary current-domain AP is 0.5890 for MS-L1 and 0.5912 for MS-ft. Both use the first row's 44.56% view-equal prevalence. The second row answers the historical reconstruction diagnostic. Populations change with the reconstruction and the two comparisons give opposite orderings here. The original operating threshold is a third, separate measurement. MS-L1 called 118,537 pixels low risk across its full application domain, with 32,914 errors. With these few exposed objects, the result supports retaining the real MS-ft comparison and the released MS-L1 default pending independent evidence.

B-L1-transfer uses 26 tabular columns from its designated primary source. The three MS methods use 39 tabular columns, including source photometry and self-render residuals. The visual methods also use their released visual components. All feature columns, model parameters, thresholds and historical result rows retain their published identities.

[Audit of all four methods on both historical objects](comparison-domains.csv) · [Recorder receipt and original input hashes](domain-audit.json) · [Historical per-view counts including cutoff ties](../reconstruction-recovery/risk/matched-coverage.csv) · [Original recovery report](../reconstruction-recovery/risk/README.md)
