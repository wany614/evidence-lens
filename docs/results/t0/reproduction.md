# Replaying the frozen T0 study

T0 uses the `v0.3.0-rc.1` risk models released at `a65b1caea258d991573bddb7a65f2bc4f9945fd7`. The recovered reconstruction and future-object registration came from `bc8da49815751fa7a6e191d0dc17c9f987f9c825`. R0 and R2 had completed and the R1 eligibility decision was already recorded. The selected 5,000-update recipe has SHA-256 `57730c706aaaa2f88c7f39ff179ba04ff971f31e599bf3409a3d545c61d72c98`. Preserve these separate identities when reproducing the experiment. [Executable protocol](../../../protocol/t0-confirmation-v1.json) · [Reconstruction authority](../reconstruction-recovery/authority/freeze.json)

## Roles and stored evidence

The Mac controller uses a project-specific recorder store on the external volume. The historical store stays readable at its original location. Each new datum identifies its creating run, recorder manifest, relative member path and SHA-256. `runs/` contains completed originals, `quarantine/` and the receive records retain failures, and `t0/` binds individual reconstructions to the common fixture and freeze. A transport directory carries only the source-authorized job or its recorded return. The independent evaluator keeps reference photographs and comparison panels in private payloads. Public source archives exclude DTU images, derived pictures, checkpoints, numerical arrays and machine telemetry.

The official 124-scan ZIP directory was read in validated HTTP byte ranges. Identity selection used the published subject descriptions and DTU rotation groups. No source or target image body was read to choose the list. The intake records the DTU access description, citation instructions and scope of the private research use. A third-party reuser must obtain the official data under the applicable terms. The full crop includes background and incidental props. [All 124 dispositions, exact file counts, rights and identities](intake/README.md)

| Reconstruction | Source photos at light 3 | Target positions used only for rendering before opening |
|---|---|---|
| M2 | s00 = 24, s03 = 22 | 16, 18, 25, 37, 35, 36 |
| M4 | s00 = 24, s01 = 23, s02 = 17, s03 = 22 | 16, 18, 25, 37, 35, 36 |

The source recorder converts the original RGB8 to a fixed central 256 × 256 crop. Right-camera rectified projections and camera-Z share one metric coordinate system. Only source observations contribute to initialization and optimization. Eligibility requires 24 distinct tracks, source-specific track counts and occupied source cells. A failed source gate selects the pre-existing 2,048-point `endpoint_stratified_inverse_depth` hypotheses. Set1/M2 supplied the single forced engineering fixture. The independent readback checks the actual five optimizer checkpoints and the final colour, opacity and expected-depth semantics. [Camera protocol](../../dtu-camera.md) · [Initialization protocol](../../../protocol/recovery-initialization-v1.json)

## Scientific entry points

Run the project recorder with the research dependencies. All inputs are explicit and all published original results have run/member identifiers. Set `STORE` to the new research store and `HISTORY` to the location holding historical recorded inputs. The old release's CPU vision interpreter is required for the four-way scientific scoring. The lighter deployment CLI uses its separate installation instructions.

```sh
python scripts/t0_study.py --store "$STORE" --history "$HISTORY" freeze --roster "$ROSTER_PIN"
python scripts/t0_score_matrix.py --store "$STORE" --history "$HISTORY" --cpu-python "$VISION_PYTHON"
python scripts/t0_study.py --store "$STORE" --history "$HISTORY" lock
python scripts/t0_study.py --store "$STORE" --history "$HISTORY" open --matrix "$STORE/work/controller/closed-matrix.json"
python -B scripts/t0_locked_evaluation.py --store "$STORE" --history "$HISTORY" --references "$REFERENCE_ACCESS_LIST"
```

The operations above span distinct authorities. Before scoring, the intake and reconstruction workers must produce the condition-specific source receipts and authenticated native Cases. The software lock additionally requires the authored evaluation-fixture receipt. Between `open` and `evaluate`, the independent reference reader obtains only the six permitted target members for each registered object. The deployment scorer has no reference parameter. The method-specific scores and states come from the pinned distributed `cpu.py` runtime. [Source and reference intake](../../../scripts/t0_dtu_record.py) · [Reconstruction controller](../../../scripts/t0_reconstruct.py) · [Frozen CPU entry](../../../scripts/dtu_score.py) · [Independent evaluator](../../../scripts/t0_evaluate.py)

### Recorded opening and recovery

The first software lock preceded a cross-store receipt repair. A second lock recorded the exact changes, with the same object list, scoring implementation, evaluator and thresholds. The first attempt to open the targets from the working checkout failed its file check before producing an access receipt. The successful controller ran from the second lock’s archived source. Its recorder sets the worker’s script directory, working directory and Python source path to the new snapshot of those same bytes.

The archival loader’s first preflight also detected one interpreter-created bytecode cache. It was removed from the source tree and retained with its hash in the private incident record. All original files retained their hashes, the entire original manifest verified, and the retry suppressed bytecode writes. The authorized opening preceded all 36 target downloads. The engineering fixture used 24 authored maps and 24 deliberately absent maps to exercise the evaluator. Its surrogate image is unrelated to the six physical subjects.

To replay this incident from **the recorded failed attempt**, use the command below. A fresh study can instead start with its matching frozen tree. The authority returned by an already completed opening is reused.

```sh
python -B scripts/t0_locked_evaluation.py --store "$STORE" --history "$HISTORY" \
  --failed-opening-run 20261124T013330448584Z-042531c306864803
```

[Original freeze](authority/freeze.json) · [Pre-opening software and its transport amendment](authority/software-lock-final.json) · [Collective access receipt](authority/opening.json) · [Independent evaluation](confirmation/evaluation.json)

## Reading the denominators

V counts reference-valid crop pixels. E counts pixels in V with comparable prediction RGB. Y is the count in E where float32 mean absolute display-sRGB error is strictly greater than 0.05. S counts the method's finite scores within E. C intersects E with finite scores from all four methods. At the released threshold, L counts finite scores strictly below the threshold, H contains the other finite scores, and U counts missing scores within E. LY, HY and UY count actual errors in these groups. E = L + H + U and Y = LY + HY + UY. An absent prediction reduces E. An empty accepted group has undefined conditional error.

The four current methods use `within_new` for AP and matched coverage. Per-view exact-k selection uses recorded label-independent ties with the original rounding rule. AP is undefined for an empty or single-class population. Its companion prevalence uses the same defined views. The research tables report each object's numerator and denominator, and the aggregate gives each physical family equal weight separately for M2 and M4. `paired_old_new` is available only for historical two-generation diagnostics. [Domain audit](comparison-domains.md) · [Original AP, coverage and tie rules](../../../protocol/recovery-risk-comparison-v1.json)
