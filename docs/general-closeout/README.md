# Can sparse source evidence locate photometric errors in a new view?

Evidence Lens scores each pixel in an upstream reconstruction. Its inputs are the source photographs, source self-renders, calibrated cameras and predicted target RGB, expected camera-Z and opacity. An independent evaluator compares the prediction with a held-out photograph at that camera. The event is mean absolute display-sRGB error strictly above 0.05. The default MS-L1 and the optional visual methods carry the frozen parameters and native thresholds from `v0.3.0-rc.1`. [Training and model identities](../releases/v0.3.0-rc.1/MODEL_CARD.md) · [Input and output contract](../tooling/README.md)

## Which evidence belongs to which question?

| Population | Role and exposure |
|---|---|
| Authored Blender families | Trained or evaluated the risk models in their original, disjoint splits. The four held-out families supplied 48 native target renders. |
| DTU set1 and set6 | The upstream scene recipe was selected using set1. Previously inspected set6 became a regression case. Both were later used for camera-policy development. |
| T0, six physical objects | Confirmed the frozen risk methods with the recovered scene recipe. These results were available before C6 calibration and the T1 freeze. |
| C6, six other physical objects | Fitted and diagnosed layout-specific threshold candidates for MS-L1. Each object provided both layouts. |
| T1, six further physical objects | Prospectively registered fixed-layout scoring and paired view selection. The source broker separates selector decisions, authorized photographs and independent evaluation references. |
| Two original static scenes | Redistributable engineering controls for the CPU tool, native array comparison, camera trajectories and load measurements. |

The full centre crop defines reference-valid pixels V. E counts comparable target predictions, Y counts RGB errors, S counts comparable finite scores, L counts those below the released threshold, and U counts comparable pixels with unknown risk. Low coverage is L/V and error within the low group is LY/L. A missing render leaves the reference denominator and has no comparable target pixels. An absent or empty denominator has an undefined rate. A verified foreground mask was unavailable. [Denominator and camera conventions](../results/t0/confirmation/README.md) · [Prospective endpoints](../results/batch-two/holdout/prospective.json)

## Four frozen models

On the authored confirmation families, common-domain AP for MS-L1 and MS-ft was 0.747 and 0.734 with two sources, and 0.704 and 0.665 with four. The development-selected MS-L1 remains the default. B-L1-transfer consumes only the registered primary source. MS-L1, MS-frozen and MS-ft use all registered sources. [Original confirmation](../results/multisource-confirmation.md)

T0 covered set2, set4, set10, set11, set12 and set31, with all 288 model–view slots scored. For each target, `within_new` restricts ranking to the same current candidates across methods. At fixed 20% coverage the table gives the object-equal mean error among the accepted pixels. In a separate diagnostic, `paired_old_new` restricts to candidates shared with the earlier reconstruction. These two candidate sets can yield different method rankings. [T0 pixel counts by object and view, every quota and tie bound](../results/t0/confirmation/README.md) · [Domain audit on the exposed regression](../results/t0/comparison-domains.md)

| Fixed layout | Method | Common AP | Matched prevalence | Error at 20% coverage |
|---|---|---:|---:|---:|
| M2 | B-L1-transfer | 0.7750 | 64.69% | 48.75% |
| M2 | MS-L1 | 0.8185 | 64.69% | 42.12% |
| M2 | MS-frozen | 0.8126 | 64.69% | 41.81% |
| M2 | MS-ft | 0.8172 | 64.69% | 40.94% |
| M4 | B-L1-transfer | 0.5256 | 38.63% | 21.68% |
| M4 | MS-L1 | 0.6052 | 38.63% | 21.05% |
| M4 | MS-frozen | 0.6073 | 38.63% | 18.60% |
| M4 | MS-ft | 0.6193 | 38.63% | 16.94% |

The six T0 objects support a different comparison from the authored families. The four-source T0 results favor MS-ft on both common AP and matched 20% risk. Reconstruction, lighting, rig, object and the definition of eligible pixels all constrain transfer.

## Threshold adaptation and development camera selection

C6 covered set48, set75, set110, set42, set77 and set52. The rule required at most 5% mean accepted-pixel error, at least 20% mean coverage, and a nonempty accepted group in every object. Of the 201 registered anchors, no cutoff satisfied all three conditions on either layout. Among anchors meeting the coverage and nonempty-group conditions, minimum mean error was 60.69% for M2 and 15.15% for M4. All twelve leave-one-object-out fits also yielded no feasible cutoff. The two published profiles retain `tau: null`. The original map and its native threshold remain available. [Candidate curves, complete denominators and leave-one-out results](../results/batch-two/calibration/README.md)

On the development objects, risk-minus-uniform target MAE was −0.000257 on set1 and exactly zero on set6, where both strategies selected the same photos. The three registered random camera choices have separate observations. T1 uses new physical objects and its own prospective endpoint. [Development policy results and controls](../results/batch-two/active-views/README.md)

## Scope of the tool and study

The NumPy/Pillow default consumes prepared evidence. Optional neural CPU scoring loads a full released encoder and head. Rendering accepted Gaussian assets is a separate upstream CUDA operation. In deployments without evaluation photographs, the HTML displays scores, finite support, states, unknown reasons and applicability. A user needs independent labels to measure prediction accuracy. [Supported input formats and tested renderer](../tooling/assets/README.md) · [Optional model and dependency rights](../multisource-rights.md)

All DTU observations in these studies share the named lighting condition, rig, crop, registered source layouts and reference cameras. Counts are retained per object and view. Aggregate comparisons give each physical object equal weight. The static, synthetic replay measures queueing and service on prepared arrays. Future geometric, semantic, dynamic and embodied use requires independent evidence. [Specialization prerequisites and metadata](specialization-readiness.md) · [Contributions and decision log](research-log.md)
