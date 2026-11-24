# Model card for the general CPU tools

Tool version 0.4.0 runs the original risk weights from release `v0.3.0-rc.1`, published at commit `a65b1caea258d991573bddb7a65f2bc4f9945fd7`. Each model predicts the risk of a colour error on a new view of a static Gaussian scene. The registered event is mean absolute display-sRGB error strictly above 0.05. The source image and prediction coordinates, expected camera-Z in metres, source self-renders and support semantics are required inputs. Independent target photographs belong to evaluation. [Original training card and gradients](../v0.3.0-rc.1/MODEL_CARD.md) · [Frozen data and feature definitions](../../stage3-risk.md)

| Method | Evidence and parameters | Runtime |
|---|---|---|
| B-L1-transfer | Earlier single-source logistic model, designated primary source only | CPU numerical features |
| MS-L1 (default) | All registered sources, 39-column feature schema, frozen linear coefficients and preprocessing | NumPy and Pillow |
| MS-frozen | Released DINOv2 encoder with trained projection and risk head | Optional CPU PyTorch and safetensors |
| MS-ft | Fine-tuned DINOv2 parameters, projection and risk head | Optional CPU PyTorch and safetensors |

The retained MS-ft fit used seed 1729 and 288 optimizer updates. Training changed 1,775,998 eligible backbone parameters. The original training, selection, train/dev curves and change evidence are preserved. The general study ran zero new risk-model training jobs and zero new calibration fits. T1 Gaussian scene optimization is separately accounted for. [Original optimization evidence](../v0.3.0-rc.1/MODEL_CARD.md) · [Training curves](../../results/stage3-learning-curves/README.md)

The independently downloaded CPU archive was 164,854,827 bytes. Its SHA-256 is `f555956840870309217cc85fe9b339ef53762679bbd2138b7f149d8ea451a7b7`. The four models and their preprocessing are bound by the original 116-member inventory and `release.json` SHA-256 `150b04e702f5386b5808ac17545cd073e3d4d837ea789e6ae167168ac086084d`. Compatible tools load the full declared model and fail on missing parameters. [Verified download and per-model digests](../../general-closeout/status.json) · [CPU example and verification command](README.md)

The authored holdout favored MS-L1 over MS-ft on common-domain AP for both layouts. On the six DTU T0 objects, the four-source matched-20% error was 21.05% for MS-L1 and 16.94% for MS-ft. Original cutoffs address a different coverage question. On the six calibration objects neither layout met the registered operational target, and both profiles retain a null cutoff. A score or low-risk state carries no formal reliability guarantee. [T0 counts by physical object](../../results/t0/confirmation/README.md) · [Calibration and actual acceptance counts](../../results/batch-two/calibration/README.md) · [Current evidence and limits](../../general-closeout/README.md)

The supported uses concern the registered sparse sources, calibrated view family, static reconstructions and photographic error. Missing finite support yields an unknown state. Geometry, semantic decisions, motion separation and physical control have no independent evaluation in this project. The source release and derived imagery use separate license scopes, with attribution for the upstream DINOv2 work. [Rights for methods, weights and examples](../../multisource-rights.md) · [Prerequisites for later specialization](../../general-closeout/specialization-readiness.md)


## Prospective transfer on new physical objects

T1 M2 yielded common AP of 0.7295 (MS-L1) and 0.7276 (MS-ft) on six objects. At matched 20% coverage their object-equal errors were 33.16% and 31.23%. T1 M4 yielded 0.5851 and 0.6013 common AP, and 15.79% and 12.40% matched error, on four comparable objects out of six scheduled. On two other objects the frozen upstream query probe failed, so their candidate photographs were never obtained. Every fixed-layout evidence and failure slot was recorded before the one independent opening. [Complete object-level study and failure denominators](../../results/t1/README.md) · [Numerical bridge with the engineering tool](../../tooling/numerical-bridge.md)


On Mac, the released four-thread file, resident and cached paths reproduced all 240 available T1 outputs bitwise. The faster eight-thread setting selected on an original engineering control was internally consistent across its three paths. Relative to four threads, MS-ft changed 28 individual risk values and MS-frozen changed 23. No native threshold, state or registered matched-coverage membership changed. Reported scientific AP uses the four-thread reference. [Exact environment, counts and limits](../../tooling/numerical-bridge.md)

The CPU engineering replays used an Apple M6 arm64 Mac and a native x86-64 Windows laptop. Both completed 4,800 hot samples and 72 cold launches. At 30 Hz the Mac delivered 1,503 of 1,800 default-model inputs and 809 of 1,800 MS-ft inputs. Windows delivered 470 of 1,800 default-model inputs. The remaining frames and expiration counts are in the per-input records. These latency results apply to the two original static scenes and the specified prepared 256 × 256 evidence. [Configuration timings and input dispositions](../../general-closeout/performance/README.md) · [CPU and rendering boundaries](../../general-closeout/README.md)

## Replaying the distributed wheel

The CPU release at `7051de93f7b8af764c5ded4aeb623fb205b56452` passed 45 targeted regressions from its immutable source ZIP. Mac arm64 installed the anonymous download and replayed all four methods on the original six bell M4 targets. Native Windows x86-64 replayed MS-L1 and the complete MS-ft weights in fresh environments. A network-off arm64 Linux container installed the same downloaded wheel and replayed MS-L1. All six default views were bit-exact on the three platforms. All four Mac methods matched their original arrays. On Windows, the MS-ft output differed in 374,959 risk values, at most 8.1062317e-06. The comparison recorded 0 state changes, 0 threshold crossings and 2 changed quota memberships across 60 attainable comparisons. The published T1 AP and matched risk estimates remain the four-thread scientific results. [Native receipts and original-view numerical differences](../../verification/v0.4.0/README.md)
