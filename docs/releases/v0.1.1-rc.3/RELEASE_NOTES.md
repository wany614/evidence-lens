# Evidence Lens v0.1.1-rc.3 release notes

This public Experimental Research Preview versions the CPU software and its distribution terms. It ships the frozen rc.2 L1 model, ID `20261123T042819219876Z-27eaa01eba29489d`. The two logistic models, their preprocessing and the selected thresholds retain their original numerical identities. Its committed wheel uses Python package version `0.1.1rc3`. The release manifest identifies the selected method, fit, example, scores, original formal report, source commit and package hashes. The scientific scope records which conclusions the diagnostic evidence supports. [Release manifest](release.json) · [Scientific scope](SCIENTIFIC_SCOPE.md)

The CPU package includes two fitted regressions, their feature preprocessing and schema, the operating thresholds, three comparison methods and readable Python modules in `source/evidence_lens/` that match the wheel. The selected formal results are numerical tables. The image and prediction files in this package come from the original authored tabletop plane example. [Model card](MODEL_CARD.md) · [Example provenance](example/provenance.json)

The clean CPU smoke compares all five methods over all six example targets against scores from the original recorded example. It checks finite masks, low/high/unknown states, L0 reason codes and a maximum absolute score difference of 1e-6. The smoke records the measured system and installed package versions. A fresh archive restoration, a formal scientific case review and the software smoke each have different evidence and populations. [Run instructions](README.md) · [Use terms](docs/licenses.md)

## Distribution terms

The authorized CPU package carries Apache-2.0 for original software and independently fitted parameters, and CC BY 4.0 for the original documentation, authored example and licensable research records. Attribution is wany614. The standalone wheel identifies the licenses covering its source and embedded README. [Per-file scope](release.json) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Release assets](https://github.com/wany614/evidence-lens/releases/tag/v0.1.1-rc.3)
