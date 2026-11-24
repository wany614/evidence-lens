# Frozen multi-source evidence study

Two ETH3D scenes supply the external confirmation. The primary comparison uses the same K16 sources and the same available pixels for both rankings. Each target accepts the lowest-scoring fifth of this common domain, rounded down. The scene-level effects below average targets equally within each frozen checkpoint.

The development-selected simple rule was **residual**. The registered primary endpoint returned **completed_negative**. The downstream release records engineering and budget acceptance separately.

| Scene | Role | Labelled errors / pixels | New errors / accepted | Rule errors / accepted | New minus rule, target-weighted | Coverage Q4 minus fixed Q4 gap |
|---|---|---:|---:|---:|---:|---:|
| b01-staggered-arcade | train | 56,056 / 1,143,072 | 19 / 228,612 | 4 / 228,612 | 0.0066% | -0.0419% |
| b02-terraced-court | train | 44,535 / 1,143,072 | 68 / 228,612 | 38 / 228,612 | 0.0131% | -0.0542% |
| b03-woven-truss | train | 72,281 / 1,143,072 | 80 / 228,612 | 94 / 228,612 | -0.0061% | -0.0233% |
| b04-layered-louvers | development | 57,474 / 1,143,072 | 66 / 228,612 | 19 / 228,612 | 0.0206% | -0.0683% |
| delivery_area | development | 328,488 / 952,560 | 57,864 / 190,510 | 57,319 / 190,510 | 0.2861% | -38.9347% |
| b05-perforated-gallery | confirmation | 62,778 / 1,143,072 | 98 / 228,612 | 23 / 228,612 | 0.0328% | -0.0429% |
| b06-corner-alcoves | confirmation | 50,913 / 1,143,072 | 128 / 228,612 | 59 / 228,612 | 0.0302% | -0.0472% |
| living_room | confirmation | 332,811 / 952,560 | 51,334 / 190,510 | 45,524 / 190,510 | 3.0497% | -19.8946% |
| bridge | confirmation | 341,069 / 952,560 | 44,943 / 190,510 | 46,014 / 190,510 | -0.5622% | -24.3175% |

## Missing comparable evidence

| Scene | Fixed Q4 gaps / correct | Coverage Q4 gaps / correct | Coverage Q16 gaps / correct |
|---|---:|---:|---:|
| b01-staggered-arcade | 1,587 / 1,087,016 | 1,128 / 1,087,016 | 75 / 1,087,016 |
| b02-terraced-court | 1,186 / 1,098,537 | 588 / 1,098,537 | 123 / 1,098,537 |
| b03-woven-truss | 1,834 / 1,070,791 | 1,589 / 1,070,791 | 130 / 1,070,791 |
| b04-layered-louvers | 2,162 / 1,085,598 | 1,412 / 1,085,598 | 159 / 1,085,598 |
| delivery_area | 263,372 / 624,072 | 680 / 624,072 | 0 / 624,072 |
| b05-perforated-gallery | 1,444 / 1,080,294 | 978 / 1,080,294 | 111 / 1,080,294 |
| b06-corner-alcoves | 1,858 / 1,092,159 | 1,338 / 1,092,159 | 139 / 1,092,159 |
| living_room | 141,729 / 619,749 | 153 / 619,749 | 1 / 619,749 |
| bridge | 139,207 / 611,491 | 21 / 611,491 | 0 / 611,491 |

The P1 gap requires a correct independent RGB label, no comparable source in the selection, and a comparable source in the complete audited bank. Its denominator is all correctly predicted labelled pixels in that target. The P2 table includes the accepted share of the complete reference image. For the authored scenes, geometric-visibility.csv compares the model to independent mesh line-of-sight on the registered 54 × 36 grid. nearest-depth-approximation.csv reports the separate dense calculation from sampled depth. Error capture uses eligible positives. The column capture_of_all_labelled_errors includes the errors outside that score domain. View-priority times distinguish shared source selection and evidence, joint simple-rule timing, head execution and target-level aggregation.

[Per-view primary denominators](views.csv) · [Full tables including each method and domain](https://github.com/wany614/evidence-lens/releases/download/many-source-v0.1.0-dev.0/many-source-results.zip) · [Scene comparability](comparability.csv) · [Two-view priority check](view-priority.csv) · [All measured results](endpoints.json)
