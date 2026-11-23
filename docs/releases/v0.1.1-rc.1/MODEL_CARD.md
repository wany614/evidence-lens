# Evidence Lens model card

L1-full estimates the risk that a target pixel’s mean absolute display-sRGB error exceeds 0.05 relative to an independent reference image. Its score ranks pixels for review. High risk can be a false alarm, and a low-risk pixel can still have a large error. Geometric accuracy needs a separate reference and measurement. [Selected model and scientific status](SCIENTIFIC_SCOPE.md)

## Model and inputs

The full and geometry-only predictors are L2-regularized logistic regressions. Their files include fitted coefficients, intercepts, imputation values and standardization parameters. The feature schema defines names, order and units. Thresholds come from the development families. The wheel also computes a constant baseline, one minus accumulated alpha and the L0 rule. The model index pins the feature schema, parameter files and thresholds. [Model index](model/model.json) · [L1-full parameters](model/L1-full.json) · [Thresholds](model/thresholds.json)

The scorer takes one input RGB image, calibrated perspective cameras, a source self-render, and RGB, optical-axis depth and accumulated alpha for each target view. Expected depth refers to the alpha-weighted rendering convention tested by the depth probe. A successful semantic probe establishes how the renderer encodes depth. Scene-scale and geometric error require measurement against an independent reference. Unknown decisions retain missing or unsupported evidence.

## Study population

The registered dataset contains 24 authored, static opaque Blender scene families. Six nearby views per family share a source image. Sixteen families supplied the grouped training folds, four supplied development thresholds and four formed the original held-out cohort. The formal conditions were a 512-pixel baseline input (B0), an input resized to half resolution (R1), and a colour-only FP16 round trip (Q1). The study used Apple SHARP and gsplat. The table selection identifies every formal comparison run and its enclosed manifest. [Family split](protocol/families-v1.json) · [Training population](training/training_dataset_manifest.json) · [Selected comparisons](results/selection.json)

Apply the evidence classification in the scientific scope when citing the numbers. It distinguishes the original frozen evaluation, historical regression after changes informed by that benchmark, or results under review. A revised model evaluated on the previously examined families cannot establish generalization to a new independent test cohort. The family is the comparison unit because nearby pixels and views share scene geometry. [Formal scope and error decisions](SCIENTIFIC_SCOPE.md) · [Family-equal summaries](results/family-equal-summary.csv) · [Paired comparisons](results/paired-family-equal.csv)

## Reading the denominators

V is the reference-valid population. E is the subpopulation with evaluable RGB. Y is the count of confirmed RGB errors in E. Common-mask AP ranks pixels evaluable for the compared methods. The low-risk error rate divides confirmed errors in the low-risk decision by its accepted evaluable pixels. High-risk recall uses confirmed errors as the denominator. LY counts confirmed errors assigned low risk, UY counts those assigned unknown risk, and LY + UY counts all errors left outside high risk. The scope document derives these counts from the selected report. [Counts by family, condition and method](results/family-results.csv)

Use the model for inspecting nearby synthetic views with compatible inputs. The measured scope is limited for transparent surfaces, real photographs, moving subjects, wider changes of viewpoint and other generators. The separate authored plane example verifies software installation and numerical deployment. Its six-plane-target score comparison supplies no estimate of SHARP depth fidelity or new-family accuracy. [Known issues](KNOWN_ISSUES.md) · [Run the CPU example](README.md)
