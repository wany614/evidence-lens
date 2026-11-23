# Family risk–coverage curves

Each panel shows one registered family and all seven methods. Markers show attainable coverage after keeping equal scores together. Hollow diamonds use the thresholds frozen on development. The vertical coordinate counts actual RGB errors among accepted pixels. A family with no accepted pixels has an undefined error rate.

| Cohort | Native input (B0) | Half-resolution input (R1) | FP16 colour round-trip (Q1) |
| --- | --- | --- | --- |
| New confirmation (8 families) | [B0](new_confirmation-B0.png) | [R1](new_confirmation-R1.png) | [Q1](new_confirmation-Q1.png) |
| Historical regression (4 families) | [B0](historical_regression-B0.png) | [R1](historical_regression-R1.png) | [Q1](historical_regression-Q1.png) |

The plotted summary has SHA-256 `b4c81223c41ecd0758be5a76352716953e7b391059c90057cd71300783684a31`. Its sealed report is `20261123T152543832083Z-c80e50ae1e1d40aa`. The figure run is `20261123T152720077419Z-3ef0701f629c47ca`. Rebuild all six images with the source checkout and that summary. The published render used Matplotlib 3.11.2 and Pillow 12.3.0. [Rendering command](../../docs/releases/v0.2.0-rc.1/REBUILD.md#draw-the-family-riskcoverage-curves) · [Scientific results](../../docs/results/vision-confirmation.md) · [Download the public tables](https://github.com/wany614/evidence-lens/releases/download/v0.2.0-rc.1/evidence-lens-v0.2.0-rc.1-numerical.zip)

Original plots and grantable measurements © wany614, CC BY 4.0.
