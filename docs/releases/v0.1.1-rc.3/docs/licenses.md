# Asset licenses and attribution

Attribute Evidence Lens to **wany614**. The root Apache license covers original software and configuration. The CC BY 4.0 license covers original documentation, eligible study records and authored media. The manifest identifies the assets, exact bytes and material class of each ZIP member. [Apache-2.0](../LICENSE) · [CC BY 4.0](../licenses/CC-BY-4.0.txt) · [NOTICE](../NOTICE) · [Asset inventory](../licenses/ASSET-SCOPE.json)

| Supplied material | Terms and origin |
| --- | --- |
| Python, inference and recording software, wheel code and executable configuration | Apache-2.0. Keep the applicable copyright, license and NOTICE and identify modifications as the license requires. |
| Fitted L1 parameters, preprocessing and operating thresholds | Apache-2.0 for the project’s independently trained logistic regression models. The parameters were fitted on prediction features and independent Blender error labels. |
| Narrative documentation and model card | CC BY 4.0. Credit wany614, link the license and indicate changes when sharing adaptations. The wheel contains Apache-licensed Python and a CC-licensed README. |
| Authored tabletop example | CC BY 4.0 for the project's 128-pixel input, calibrated cameras, two-metre plane projection, numerical expectations and associated documentation. |
| Original research records | CC BY 4.0 to the extent rights exist in the project-created material. Factual values may carry no exclusive rights. The reference planes and model predictions have separate provenance. |

The upstream SHARP software and checkpoint have distinct terms. The model agreement has a non-commercial scientific research restriction for the upstream model and covered derivatives. The selected research used fixed SHARP predictions as data; the separate L1 optimizer updated its own parameters. This package supplies the original CPU scorer, independent coefficients and the project’s authored demonstration. The upstream checkpoint, raw SHARP predictions, Gaussian PLYs and formal SHARP image boards remain with their separate sources. The recorded licenses and acknowledgements for SHARP, gsplat and dependencies are identified in the third-party inventory. NumPy and Pillow are installed separately. [Model notice](../licenses/MODEL-LICENSE.txt) · [Example notice](../licenses/EXAMPLE-LICENSE.txt) · [Research notice](../licenses/RESEARCH-RECORDS.txt) · [Third-party notices](../THIRD_PARTY_NOTICES.md) · [Dependency source inventory](../licenses/DEPENDENCY-SOURCES.json)

The citation file provides a recommended research citation. [Citation metadata](../CITATION.cff) · [Source repository](https://github.com/wany614/evidence-lens) · [Release manifest](../release.json)
