# Evidence Lens 0.5.0

This CPU tool scores prepared two- and four-source evidence on a finite set of image grids. The original source photos, the analysis pixels, the encoder's patch lattice and the display raster have separate descriptions. The four risk models keep their released numerical parameters.

## Score an authored example

Install the wheel in a new Python 3.12 environment. Download the prepared bell examples from the same release, verify the file checksums and extract the example archive. Run these commands from the directory containing the six bundles. Use a fresh output directory for each score.

```sh
python -m pip install ./evidence_lens-0.5.0-py3-none-any.whl
evidence-lens profiles
evidence-lens validate --bundle landscape
evidence-lens score --bundle landscape --model MS-L1 --out landscape-score
evidence-lens report --scores landscape-score --display-size 1920 1080 --out landscape.html
```

The example above computes risk on 448 × 252 pixels. The 1920 × 1080 display uses nearest-neighbour sampling for risk classifications and unknown reasons. Every count in the report comes from the 448 × 252 numerical arrays. The base install uses NumPy and Pillow. The same CLI accepts the original 256 × 256 bundles.

For MS-ft and MS-frozen, install the vision extra from the same wheel. Download the complete CPU model release with the source distribution's verified downloader. Use the extracted directory containing `release.json`.

```sh
python -m pip install './evidence_lens-0.5.0-py3-none-any.whl[vision]'
python scripts/general_release_weights.py --out weights
evidence-lens score --bundle photo --model MS-ft --threads 4 --models weights/cpu --out photo-ft
evidence-lens report --scores photo-ft --out photo-ft.html
```

The finite profile identifiers are `analysis-256x256-v1`, `analysis-448x252-v1`, `analysis-252x448-v1`, `analysis-378x252-v1` and `analysis-375x251-v1`. Profile dimensions are width × height. The prepared example directories use `square`, `landscape`, `portrait`, `photo` and `odd` respectively. The `legacy-letterbox` example holds the same original landscape frustum on the historical 256-square canvas.

## Prepare another input

The v2 bundle includes a checked original pinhole camera, integer crop, half-pixel resize affine and new K for each source and target. GPU Gaussian rendering produces expected camera-Z and accumulated opacity directly on the chosen analysis grid. Original colour photographs use the registered source filter. A zero-opacity pixel has undefined expected depth. The metric camera and depth units must match. The renderer can inspect a supported full asset on the CPU and requires CUDA for Gaussian rendering. Full bundle verification precedes CPU scoring. [Grid implementation](../../../src/evidence_lens_bundle/grids.py) · [Asset contracts](../../tooling/assets/README.md)

Existing consumers may continue to call `load_scorer`, `prepare_sources` and `score_frame`. The v2 source and target bindings cover the analysis profile, original-grid witness, prepared camera, model and actual ordered input content. The frame API returns risk, finite support, state and unknown reason. [Resident API](../../tooling/resident.md) · [Method and rights](MODEL_CARD.md) · [Scope](SCOPE_AND_LIMITATIONS.md)
