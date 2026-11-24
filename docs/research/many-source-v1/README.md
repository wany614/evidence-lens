# Multi-source evidence audit

The tool checks a fixed target render against a stated selection of source photographs. A limited query produces its risk ranking and identifies which sources are comparable. An explicit full audit can find a source that the limited query missed. Independent RGB measurements supply the error labels in this study.

**Study result: completed_negative**. The registered K16 comparison tested a fitted NumPy head against the `residual` rule. Each of the two external confirmation scenes contributed ten targets. Four independent target sets were opened together after all six confirmation score or failure receipts existed.

| ETH3D confirmation scene | Labelled errors / pixels | New errors / accepted | Rule errors / accepted | Mean target-level difference |
|---|---:|---:|---:|---:|
| living_room | 332,811 / 952,560 | 51,334 / 190,510 | 45,524 / 190,510 | +3.050 percentage points |
| bridge | 341,069 / 952,560 | 44,943 / 190,510 | 46,014 / 190,510 | -0.562 percentage points |

[Open the numerical report](../../../examples/many-source-study/report.html) · [Measured results and P1 gap counts](results.md) · [Every numerical table, including risk–coverage curves](https://github.com/wany614/evidence-lens/releases/download/many-source-v0.1.0-dev.0/many-source-results.zip)

## Run the authored example on CPU

Download the original input ZIP into an empty working directory. Create a Python 3.12 virtual environment with `python -m venv .venv`. On macOS or Linux activate it with `source .venv/bin/activate`. In Windows PowerShell use `.\.venv\Scripts\Activate.ps1`. [Authored input ZIP](https://github.com/wany614/evidence-lens/releases/download/many-source-v0.1.0-dev.0/evidence-lens-b04-r16-example.zip) · [Two prepared reports with JSON and arrays](https://github.com/wany614/evidence-lens/releases/download/many-source-v0.1.0-dev.0/evidence-lens-b04-r16-reports.zip)

```sh
python -m pip install https://github.com/wany614/evidence-lens/releases/download/v0.5.0/evidence_lens-0.5.0-py3-none-any.whl https://github.com/wany614/evidence-lens/releases/download/many-source-v0.1.0-dev.0/evidence_lens_many_source-0.1.0.dev0-py3-none-any.whl
python -m zipfile -e evidence-lens-b04-r16-example.zip case
cd case
evidence-lens-bank inspect --bank bank
evidence-lens-bank score --bank bank --target h00 --budget 4 --strategy coverage --model model.json --model-sha256 b9e6b761f2ff15d2c2aada851aa0e239f550163368107692b0c536843f11c721 --out finite
evidence-lens-bank audit-bank --bank bank --target h00 --budget 4 --strategy coverage --model model.json --out audited
```

Open finite/report.html and audited/report.html. Each report carries the ordered query IDs, witnesses, scope states and the frozen rank. The original b04 target uses 16 source photographs. Its independent static-scene bank and optional CUDA upstream example have separate entry points. [Four-source panel](../../../examples/many-source/b04-layered-louvers/score/report.html) · [Full audit](../../../examples/many-source/b04-layered-louvers/audit-bank/report.html) · [Upstream adapter](../../../prototype/many-source/upstream/README.md)

## Model and review receipts

The trained parameter file is 6,382 bytes. Its SHA-256 is `b9e6b761f2ff15d2c2aada851aa0e239f550163368107692b0c536843f11c721`. The operating point is unset. Mac arm64 and an offline Linux arm64 container ran the original deployment example. Native Windows x86-64 supplied the source-only Gaussian fits. [Model ZIP](https://github.com/wany614/evidence-lens/releases/download/many-source-v0.1.0-dev.0/evidence-lens-many-source-model-0.1.0.dev0.zip) · [Model card](model-card.md) · [CPU platform results](deployment/README.md)

[Study protocol](protocol.md) · [Data roles](data-roles.json) · [Source and calibration](data/README.md) · [Frozen model and rule](freeze-manifest.json) · [Unified opening](opening.json) · [Actual reference access](access-order.json) · [Independent arithmetic](independent-arithmetic.json) · [Upstream recipe](upstream-recipe.json) · [Upstream fits](upstream.json) · [All source image and camera-grid checks](input-alignment.json) · [Checkpoint update counts](checkpoint-updates.json) · [Resource and failure ledger](resources-and-failures.md) · [Scope](limitations.md) · [Rights](rights.md)
