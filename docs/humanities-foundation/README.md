# Evidence, images and their production

Open the [case viewer](../../examples/evidence-foundation/index.html) and choose a camera. First inspect the reconstructed frame. Switch on the model assessment to see what its score and unknown states say about that frame. The source view identifies the synthetic camera images queried and the procedure that selected them. The independent evaluation opens separately.

六个铃架视角共用一套四源重建。另一个原创案例固定同一张廊道预测，依次查询 4、8、16、32 和 128 张来源。读者可以核对哪些像素多了一份可比较的证据，以及预测图本身是否改变。这两例都用合成相机生成源图。观众如何理解它们、它们能否与历史材料建立联系，仍是开放问题。

| Reading | What you can check |
|---|---|
| [Models and data](MODEL_AND_DATA_BASIS.md) | Which weights changed in previous training, what the new NumPy head learned and how an independent image becomes a label. |
| [Concepts and claims](CONCEPTS_AND_CLAIMS.md) | Seven different propositions about an image, from visual plausibility to historical interpretation, with examples and counterexamples. |
| [Three primary readings](THEORY_SOURCES.md) | The passages actually read, their versions and the limits of applying them here. [Citable bibliography](bibliography.bib) |
| [Case index](../../examples/evidence-foundation/case-index.json) | Original scene, camera, queried sources, prediction identity, reference and reproduction rights. |
| [Decisions and contributions](CONTRIBUTIONS.md) | Who introduced the concern, designed the comparison, implemented it and reviewed the materials. |
| [Open questions](OPEN_QUESTIONS.md) | Competing interpretations which the current materials leave to future discussion. |

## Materials fixed for this edition

The bell uses release **v0.5.0**, commit `c259c987b77296f368d78a9b09ef504226ac6cc3`. Its original image-grid tests and historical performance measures retain their own denominators. [Published baseline](https://github.com/wany614/evidence-lens/releases/tag/v0.5.0) · [Baseline manifest](../releases/v0.5.0/BASELINE_MANIFEST.json)

The gallery uses the separate experimental release **many-source-v0.1.0-dev.0**, commit `19cd7079ee5d4e470af4ca6b2543ad916ef0a4b4`. Its mechanism measurements and negative registered ranking endpoint both belong to this version. The configurable query in the gallery inspects the same saved Gaussian reconstruction. [Experimental release](https://github.com/wany614/evidence-lens/releases/tag/many-source-v0.1.0-dev.0) · [B manifest](../research/many-source-v1/B_MANIFEST.json) · [C registration and fixed cases](registration.json)

In the displayed gallery target, 4,579 of 95,256 pixels exceeded the independent RGB error cutoff. Among the other 90,677, a comparable source existed elsewhere in the bank for 180 pixels at coverage-first K4, 120 at K8, 109 at K16 and 12 at K32. The fixed four-source order missed 74 on this view. At K128 the selection is the entire audited bank, so a gap relative to that bank is zero by definition. Across the full image, 220 positions still had no comparable source in the audited library. The prediction and the reference keep the same identities throughout. [Queries and fixed image digests](../../examples/evidence-foundation/case-index.json)

This edition, **humanities-foundation-v1**, provides materials for reading and teaching the work. Public images come from project-authored scenes. The three primary texts remain at their cited sources. Project-authored images and explanatory writing are CC BY 4.0. The scripts are Apache-2.0. The trained model packages retain their individual notices. [Humanities edition and checksums](https://github.com/wany614/evidence-lens/releases/tag/humanities-foundation-v1) · [Source rights](../research/many-source-v1/rights.md)
