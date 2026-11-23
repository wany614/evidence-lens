# Evidence Lens multi-source CPU replay

This release scores the risk of RGB prediction error for a fixed reconstructed target. The package includes the original single-source logistic transfer, the multi-source logistic model, the frozen-encoder model and the fine-tuned model. The pre-test method receipt in release.json identifies the deployment default and threshold for every model. Both neural packages include full FP32 weights, their training-only preprocessing, feature schema, original licenses and independent parameter checks.

The companion example contains both real reconstructions of the first registered development family, offset bell cage. M2 used the two endpoints. M4 used the same endpoints and two interior images. Both conditions have the same six target cameras. CASE/ contains all authorized deployment inputs. REFERENCE/ contains independently rendered original Blender geometry and RGB evaluation data. EXPECTED/ holds the scientific scores and independent CPU replay results. The separate numerical archive covers all four registered held-out families.

Download `candidate.json`, the CPU ZIP, the example ZIP and `SHA256SUMS` from the same release. Check the ZIPs against the checksum list. Extract them into adjacent directories named `cpu` and `example`, with `candidate.json` alongside those directories. Use Python 3.11 or 3.12. [Release downloads](https://github.com/wany614/evidence-lens/releases/tag/v0.3.0-rc.1)

### Score the first target with NumPy and Pillow

From the extracted CPU directory on macOS or Linux, these commands install the dependencies for the selected MS-L1 model, read the published release identity and score the four-source example. The case identity below identifies the original four-source deployment inputs.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-light.txt
RELEASE_SHA=$(.venv/bin/python -c 'import json; print(json.load(open("../candidate.json"))["assets"]["cpu"]["index_sha256"])')
M4_CASE_SHA=8ab6d51ec5c348fc61121478631a7ead7a9299659337cc8a8a5811922ec1d598
.venv/bin/python -I -B cpu.py run --release-sha256 "$RELEASE_SHA" --case ../example/CASE/M4 \
  --case-sha256 "$M4_CASE_SHA" --view v01 --scope all --output measured-v01
```

Open `measured-v01/risk.png` to see the continuous score, from dark blue at 0 to red at 1. Open `measured-v01/state.png` to see the frozen decision. Teal means a finite score strictly below the threshold, orange-red means at or above it, and gray means unknown in both images. Both PNGs contain one RGB pixel per native target pixel. `score.json` records the actual threshold, palette and SHA-256 of each PNG. `scores.npz` retains the original numerical arrays.

In Windows PowerShell, run these commands from the CPU directory.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-light.txt
$candidate = Get-Content ..\candidate.json -Raw | ConvertFrom-Json
$m4 = "8ab6d51ec5c348fc61121478631a7ead7a9299659337cc8a8a5811922ec1d598"
.\.venv\Scripts\python.exe -I -B cpu.py run --release-sha256 $candidate.assets.cpu.index_sha256 --case ..\example\CASE\M4 --case-sha256 $m4 --view v01 --scope all --output measured-v01
```

### Run the fine-tuned model or replay every method

On Linux and Windows first install the PyTorch CPU wheel with the environment's Python. On macOS the PyPI wheel supports CPU inference.

```sh
# Linux
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
# macOS and Linux
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -I -B cpu.py run --release-sha256 "$RELEASE_SHA" --case ../example/CASE/M4 \
  --case-sha256 "$M4_CASE_SHA" --method MS-ft --view v01 --scope all --output fine-tuned-v01
EXAMPLE_SHA=$(.venv/bin/python -c 'import json; print(json.load(open("../candidate.json"))["assets"]["example"]["index_sha256"])')
.venv/bin/python -I -B cpu.py replay --release-sha256 "$RELEASE_SHA" --example ../example \
  --example-sha256 "$EXAMPLE_SHA" --output independently-replayed
```

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -I -B cpu.py run --release-sha256 $candidate.assets.cpu.index_sha256 --case ..\example\CASE\M4 --case-sha256 $m4 --method MS-ft --view v01 --scope all --output fine-tuned-v01
.\.venv\Scripts\python.exe -I -B cpu.py replay --release-sha256 $candidate.assets.cpu.index_sha256 --example ..\example --example-sha256 $candidate.assets.example.index_sha256 --output independently-replayed
```

The replay checks four methods, two reconstructions, six views and three source scopes. M2 two and all share the same original score. Both the hash-verified scientific maps and the independently recomputed CPU maps enter the comparison. `grade.json` records numerical discrepancies, threshold crossings and independent checks of the 12 original reference-valid, evaluable and high-error grids. Runtime estimates and the two-condition intervention apply to these authored scenes at 256×256.

## Installations verified after publication

The public archives were downloaded anonymously and run on macOS arm64, Windows AMD64 and Debian 13 Linux arm64 under OrbStack. The two arm64 checks used Python 3.12.16 and the Windows check used Python 3.12.10. Each replay compared all four methods in 120 distinct native outputs and independently checked 12 reference targets. Against the published CPU maps, macOS had zero threshold crossings, Windows had two and Linux had one. [Public verification receipts, numerical differences and exact environments](../../verification/v0.3.0-rc.1/README.md)
