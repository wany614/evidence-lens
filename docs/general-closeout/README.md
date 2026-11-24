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

## Prospective T1 confirmation

The six registered objects were set118, set76, set45, set103, set61 and set32. All six shared two-source fits reached 5,000 updates. The frozen selector probe required exactly 20,000 Gaussians; set118 ended at 16,251 and set76 at 17,199. Each retained its two-source predictions. These two objects had no authorized candidate observations and their twelve M4 conditions are terminal failures. Four other objects yielded twenty sealed pair choices, with one exact ordered-input fit reuse on set61. In total, 29 distinct optimizations supplied 30 successful logical conditions among 42 scheduled. All 240 available fixed-layout scores and 48 failed slots were sealed before the one independent reference opening. [All method and view dispositions](../results/t1/fixed/sealed-matrix.json) · [Opening receipt](../results/t1/fixed/opening.json)

| Fixed layout | Physical objects with defined AP | B-L1-transfer AP / error at 20% | MS-L1 | MS-frozen | MS-ft |
|---|---:|---:|---:|---:|---:|
| M2 | 6/6 | 0.6661 / 37.50% | 0.7295 / 33.16% | 0.7186 / 32.33% | 0.7276 / 31.23% |
| M4 | 4/6 | 0.4882 / 18.88% | 0.5851 / 15.79% | 0.5903 / 14.44% | 0.6013 / 12.40% |

Each cell contains object-equal common-support AP and the object-equal error among pixels accepted at matched 20% coverage. Same-support prevalence was 53.29% for M2 and 33.30% for M4. At the original MS-L1 threshold, M2 marked 899,205 of 2,359,296 reference pixels low, including 263,961 true errors. For fixed M4 it marked 805,909 of 2,359,296 reference pixels low, including 157,742 errors. The 786,432 reference pixels of the two absent M4 reconstructions contribute to the coverage denominator. M4's 1,572,864 comparable pixels contained 515,384 errors. The M2 total was 1,244,100 errors among 2,359,296 comparable pixels. The native recount checked 288 method–view rows and all 2,592 exact-quota rows, including cutoff ties. [Per-object counts, error rates, AP, unknowns and ties](../results/t1/README.md) · [Independent native recount](../results/t1/fixed/native-recount.json)

At the same coverage MS-ft reduced the object-equal error in both T1 layouts. The ordering varied by object. On set76 M2, for example, MS-ft accepted 25,946 errors and MS-L1 accepted 24,318 errors among the same quota of 78,642. Lower matched risk therefore does not establish a universal winner or a safety guarantee. The default was retained from development.

For the preregistered novel-view endpoint, risk-weighted minus uniform target MAE averaged **−0.006617** over four comparable objects. It decreased on set45 (−0.012344) and set32 (−0.022063), increased on set103 (+0.007938), and equalled zero on set61, where both selected 17 and 31 and shared the same immutable fit. The two probe failures have undefined paired differences. The registered three random seeds, fixed M4, shared M2 and all denominators accompany the pairs. An independent implementation reopened the native arrays and checked 252 logical views, including 180 comparable views. Aliases are identified in these totals. [Object pairs, random means and ranges, and failures](../results/t1/nbv.md)

After the prediction register closed, a separate reader compared the engineering scorer against the original CPU paths without reading reference photographs or errors. At four neural threads and 64 MiB, all 240 available method–view outputs matched including finite masks, states, reasons and exact-quota membership. Risk, validity and state came from the historical-style score files. Unknown reason was regenerated from the frozen engine on its original inputs. [Numerical comparison protocol](../tooling/numerical-bridge.md)

A separately registered comparison then applied the Mac performance selection of eight neural threads, chosen solely on the original bell M4 workload. The eight-thread file and both resident paths matched one another on all 240 available slots. Compared with the frozen four-thread results, 204/240 slots were bit-identical across all methods. MS-ft changed 28 risk values across 21 of its 60 views, with maximum absolute difference 1.1921 × 10⁻⁷. MS-frozen changed 23 across 15 views, at most 8.9407 × 10⁻⁸. Both linear methods matched on every view. There were zero validity, state, unknown-reason or threshold flips and zero changed accepted memberships over the 2,400 attainable quota checks per comparison. The T1 AP figures above are from the original four-thread scores. An eight-thread AP was not estimated. [Prespecified supplement](t1-threads8-preregistration.json) · [Per-view numerical differences](t1-threads8-numerical-bridge.json)

## Scope of the tool and study

The NumPy/Pillow default consumes prepared evidence. Optional neural CPU scoring loads a full released encoder and head. Rendering accepted Gaussian assets is a separate upstream CUDA operation. In deployments without evaluation photographs, the HTML displays scores, finite support, states, unknown reasons and applicability. A user needs independent labels to measure prediction accuracy. [Supported input formats and tested renderer](../tooling/assets/README.md) · [Optional model and dependency rights](../multisource-rights.md)

All DTU observations in these studies share the named lighting condition, rig, crop, registered source layouts and reference cameras. Counts are retained per object and view. Aggregate comparisons give each physical object equal weight. The static, synthetic replay measures queueing and service on prepared arrays. Future geometric, semantic, dynamic and embodied use requires independent evidence. [Specialization prerequisites and metadata](specialization-readiness.md) · [Contributions and decision log](research-log.md)

## Where the CPU time went

On Apple M6 arm64, the dense bell M4 workload supplied 200 measured hot frames per configuration. MS-L1 file service took 92.8 / 100.3 / 102.7 ms at p50 / p95 / p99. With a resident model and cached source evidence, it took 40.1 / 44.6 / 48.2 ms. Median feature construction was 31.7 ms and model evaluation took 6.2 ms. The process averaged 0.99 occupied logical CPUs. MS-ft file service took 203.1 / 224.3 / 263.9 ms. The selected eight-thread resident configuration took 73.5 / 84.9 / 88.0 ms and averaged 1.56 occupied logical CPUs. Its median feature and model times were 32.8 and 39.6 ms. These measurements locate costs in source-projection work and the neural kernel. The effect of adding parallel NumPy feature computation remains unmeasured.

The file comparison includes checking, decoding and serializing a real scored file. The array API uses prepared source data and returns fresh arrays in memory. A repeat still computes the current target feature and model output. The static CUDA stage independently rendered 548 source or target frames. Its worker took 27.66 seconds, of which 17.06 seconds were spent compressing, writing and hashing outputs. Mac verification of the 331,979,380-byte return took 61.56 seconds including local synchronization access and native archival. The elapsed network-only transfer was unobserved. [Timing boundaries and raw measurements](performance/README.md) · [Native render costs](performance/g4-upstream-cost.json)

On the four-source static bell sequence the Mac delivered every frame at 10 Hz for both models (600/600 each). At a 30 Hz input rate it delivered 1,503/1,800 frames for MS-L1 and 809/1,800 for MS-ft, or 25.04 and 13.47 outputs per second over the measured wall time. The bounded queue dropped 297 and 991 frames respectively. At 30 Hz the measured p95 input-ready-to-output latencies were 73.6 ms and 109.4 ms. The six-second map/source revision control delivered 60/60 outputs and registered a cache miss on the initial context and each revision. The inputs in this test were pre-rendered. Its timestamps measure the scoring service and its queue. [Frame dispositions, sample counts and upstream boundaries](performance/README.md)

On Windows the same 24 configurations each completed 200 hot samples and three fresh processes. The native clock recorded 1,360.828 seconds inside the 1,800-second budget. The tested neural settings were one, two, four and eight threads. The four-thread setting alone reproduced the reference on the development sample, so Windows used four. On the dense bell M4, MS-L1 median file and resident-cache service was 283.83 and 127.60 ms. For MS-ft it was 629.09 and 218.17 ms. The corresponding cached p95 times were 141.54 ms and 228.59 ms. The hardware, linear algebra backend and neural runtime differed across the two hosts. [Per-platform timings, backends and raw measurements](performance/README.md)

Windows delivered 466/600 default-model inputs at 10 Hz. The queue dropped 132 and expired two processed results. At 30 Hz it delivered 470/1,800 and dropped 1,330. Its delivered throughput was 7.76 and 7.82 outputs/s respectively. During the separate 60-input revision test, 53 results were delivered, five inputs were dropped and two processed results expired. Both platforms reported every input disposition and verified the output for each processed frame against an offline computation. The upstream renderer and synchronized transfer have their own cost records. [Load and latency tables](performance/README.md) · [Original static output example](../../examples/general-static-replay/report.html)
