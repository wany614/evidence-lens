# Reproducing the C6 operating-point calculation

The workflow uses a batch-two recorder store, the prior T0 archive and the original release archive. The first two are named by `B2` and `B1` below. Pass `--history "$HISTORY"` when the release archives are stored outside the configured project history. Use the project's Python environment with NumPy and Pillow.

```sh
python scripts/b2_calibration_record.py lock --store "$B2" --batch-one "$B1"
python scripts/b2_calibration_record.py fingerprints --store "$B2" --batch-one "$B1"
python scripts/b2_calibration_record.py candidates --store "$B2" --batch-one "$B1"
```

The method lock records the scorer release, original threshold, reconstruction recipe, the previously exposed development check, and source hashes of the calculation and reference reader. The incremental command consumes each named C6 score once its reconstruction and four-method score are sealed. It verifies per-view state and finite masks, records the score-map identity and fingerprints the sorted MS-L1 distribution of each physical object and condition. The candidate command requires the closed 12-cell matrix and verifies all fingerprints against a fresh read of the original scores.

The collective target-opening receipt must name the algorithm and candidate commitments and account for 288 planned view-method entries. The intake reader checks that receipt before obtaining the six native photographs for each C6 subject. With `OPENING` set to the verified opening pointer and `REFERENCE_INDEX` set to the six independently converted reference pointers, run the fit and independent numerical readback:

```sh
python scripts/b2_calibration_record.py fit --store "$B2" --batch-one "$B1" \
  --opening "$OPENING" --reference-index "$REFERENCE_INDEX"

python scripts/b2_calibration_audit.py audit --store "$B2" --batch-one "$B1" \
  --fit "$B2/work/controller/b2-C6-calibration/selected.json"
```

The audit reconstructs display-RGB labels from the native prediction and registered reference PNG. It verifies the weighted quantiles with exact mass comparisons immediately below each observed tie and through that tie. For every pixel it finds the first strictly larger cutoff, then uses integer bucket sums to recompute every object-level count. The public tables contain the original four methods, all repeated candidate anchors, all per-object counts and all held-out folds.

The full photographs, predictions and raw per-pixel scores are held privately under their original resource terms. Their numerical identities, the code and the registered counts support independent inspection. A new reproduction with independently obtained raw inputs needs its own recorded acquisition permission and evidence.
