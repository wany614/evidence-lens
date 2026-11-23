# Offset bell cage

Family 057 is the first registered development family. The four-image reconstruction optimized a single Gaussian scene against four calibrated RGB images. The two-image reconstruction used the same endpoint cameras. Six other cameras provided the fixed target views. The public example ZIP includes both source sets and all six predicted views for each condition. [Download the example and CPU models](https://github.com/wany614/evidence-lens/releases/tag/v0.3.0-rc.1)

![Independent target, RGB and depth error, and all four risk methods for the first target](four-method-comparison-v01.png)

The top row contains the source images, target reference, target prediction, absolute RGB error and auxiliary depth error. Each subsequent row keeps the four-image reconstruction fixed and varies the inputs to the scorer. Each method occupies two columns for risk and decision. On this target, all 65,536 pixels were RGB-evaluable and 5,149 exceeded the fixed 0.05 mean absolute display-RGB error threshold. The visual scale for depth error is in metres. Gray denotes missing or unsupported evidence and purple marks pixels outside the indicated evaluation domain.

Each panel preserves its native 256 × 256 pixel grid. A recorded display correction replaced missing font glyphs and checked all 312 panels across the twelve development boards against original numerical files. The CC BY 4.0 images are credited to wany614. [Original member hashes and run IDs](origins.json) · [Published asset rights](../../docs/multisource-rights.md)

## Default model output from the public download

The anonymous macOS run scored the four-source case with MS-L1. The panels below are its original 256 × 256 PNGs. On the state map, teal marks low risk, orange marks high risk and gray marks unknown. The risk map keeps the continuous score. [Anonymous CPU verification](../../docs/verification/v0.3.0-rc.1/README.md)

| Continuous risk | Frozen low, high and unknown states |
| --- | --- |
| ![Default per-pixel risk for target v01](m4-v01-risk.png) | ![Default frozen decision for target v01](m4-v01-state.png) |
