# Independent native-pixel check

The published scores and decisions were checked against the original Blender target RGB, the rendered prediction from each condition and the seven frozen risk maps. All 144 new target views and 72 historical target views passed. The two cohorts are reported separately. [Stage B confirmation results](vision-confirmation.md)

| Cohort | Target views | Valid reference pixels | Evaluable pixels | Confirmed errors | Seven-method common pixels | Errors in the common set | AP values checked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| New confirmation | 144 | 37,748,736 | 37,748,736 | 12,687,160 | 37,562,806 | 12,501,230 | 2,736 |
| Historical regression | 72 | 18,874,368 | 18,874,368 | 6,221,429 | 18,800,301 | 6,147,362 | 1,368 |

A confirmed error exceeded 0.05 mean absolute difference across the three display-RGB channels. The original five-method and seven-method common sets had the same counts in both cohorts. The finite, high, low and unknown pixel counts for every method, and the confirmed errors in each decision group, all matched the scientific tables. Each low decision used its frozen threshold and a strict less-than comparison within the evaluable set.

Scikit-learn 1.9.1 independently recomputed average precision (AP), preserving equal-score groups. Every comparison had both positive and negative pixels. Across the 19 comparisons per view, the greatest absolute difference from the scientific evaluator was 3.33 × 10⁻¹⁶ in each cohort. A separate readback also reconciled the 252 published family–condition–method rows with the two pixel checks and recomputed family means using each metric’s defined contributors. [Full counts, definitions and SHA-256 evidence](vision-pixel-audit.json)

## A failed verification attempt

The first independent check of the new cohort stopped while reading its evaluation header. The reader expected a method-hash field used by the historical format. It processed zero pixels. The corrected reader verified the new format’s frozen threshold hash, opening hash, direct method parent and three score-index hashes before reading any images. Under a new recorder ID it then checked all 144 targets. The published worker also reproduced the complete historical readback. The scientific confirmation, model choice and thresholds had been sealed independently of these checks.

| Evidence | Recorder ID |
| --- | --- |
| Frozen method | `20261123T140725019974Z-860cfe35388349e2` |
| Joint scientific report | `20261123T152543832083Z-c80e50ae1e1d40aa` |
| Historical pixel check | `20261123T142356293379Z-ed3d34d163084eba` |
| New-series audit that stopped at the header | `20261123T152450570567Z-160e5db0fec844b6` |
| Corrected new-series pixel check | `20261123T152644361637Z-07b948b4d278467f` |
| Historical replay with the published worker | `20261123T152925433070Z-2f5c4820daa64cd2` |

## Reproducing the independent calculation

The worker opens original runs by manifest hash. Researchers with lawful access to those records can run it in the research environment with NumPy and scikit-learn installed. The active store supplies the recorded opening, and the active and cold stores supply the sealed runs. Use a fresh recorder run for each request. Pass the chosen JSON as a recorder input named `request`, and register its method and stage pins as parents. The following arguments go after the recorder’s `--` separator. Set the two store variables to your local directories. The recorder expands the three brace-delimited paths and supplies `EL_RUN_ID`.

```sh
'{python}' '{source}/scripts/verify_vision_native_metrics.py' --root "$ACTIVE_STORE" --shelf "$COLD_STORE" --request '{input:request}' --out '{out}'
```

The historical request runs through the same worker. The public numerical tables and audit counts can be read independently of the private research originals. [New cohort request](../../configs/vision-audit/new_confirmation.json) · [Historical request](../../configs/vision-audit/historical_regression.json) · [Executed worker](../../scripts/verify_vision_native_metrics.py)
