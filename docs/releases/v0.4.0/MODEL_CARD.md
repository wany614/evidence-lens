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
