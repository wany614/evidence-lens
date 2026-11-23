# Evidence Lens multi-source CPU replay

This release scores the risk of RGB prediction error for a fixed reconstructed target. The package includes the original single-source logistic transfer, the multi-source logistic model, the frozen-encoder model and the fine-tuned model. The pre-test method receipt in release.json identifies the deployment default and threshold for every model. Both neural packages include full FP32 weights, their training-only preprocessing, feature schema, original licenses and independent parameter checks.

The companion example contains both real reconstructions of the first registered development family, offset bell cage. M2 used the two endpoints. M4 used the same endpoints and two interior images. Both conditions have the same six target cameras. CASE/ contains all authorized deployment inputs. REFERENCE/ contains independently rendered original Blender geometry and RGB evaluation data. EXPECTED/ holds the scientific scores and independent CPU replay results. The separate numerical archive covers all four registered held-out families.

Install with Python 3.11 or newer. The CPU command does not require a research-store, network credential, Blender, CUDA or a training-time package. From the extracted CPU directory on macOS/Linux run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -I -B cpu.py run --release-sha256 "$RELEASE_SHA" --case ../example/CASE/M4 \
  --case-sha256 "$M4_CASE_SHA" --view v01 --scope all --output measured-v01
.venv/bin/python -I -B cpu.py replay --release-sha256 "$RELEASE_SHA" --example ../example \
  --example-sha256 "$EXAMPLE_SHA" --output independently-replayed
```

Open `measured-v01/risk.png` to see the continuous score, from dark blue at 0 to red at 1. Open `measured-v01/state.png` to see the frozen decision. Teal means a finite score strictly below the threshold, orange-red means at or above it, and gray means unknown in both images. Both PNGs contain one RGB pixel per native target pixel. `score.json` records the actual threshold, palette and SHA-256 of each PNG. `scores.npz` retains the original numerical arrays.

On Windows create the environment with `py -3 -m venv .venv`, then use `.venv\Scripts\python.exe` for the pip and CPU commands. With PowerShell, put each command on one line and supply the real digest values from the release receipt. release.json and example.json carry the model and per-condition file identities. The independent release asset hashes in SHA256SUMS identify the ZIPs before extraction. A fresh output folder keeps measured results distinct from the supplied references.

The replay checks four methods, two reconstructions, six views and three source scopes. M2 two and all share the same original score. The signed scientific and independently recomputed CPU maps both enter the comparison. grade.json records numerical discrepancies, threshold crossings and independent checks of the 12 original reference-valid, evaluable and high-error grids. Runtime estimates and the two-condition intervention apply to these authored scenes at 256×256.
