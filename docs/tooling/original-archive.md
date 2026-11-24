# Original CPU replays

The four prepared bundles cover two authored development scenes. The helical beads have sparse projected evidence and the offset bell cage has dense support. The two bell layouts use separately fitted reconstructions. Both bead layouts use the same Gaussian asset. The archive contains the authorized source photographs, calibrated cameras, native RGB/Z/opacity predictions, 16 self-contained HTML reports and the arrays for all 64 model–view combinations.

## Run the installed wheel

Verify the archive and wheel against the published SHA-256 manifest. In a clean Python 3.11+ environment install the matching Evidence Lens wheel. The default MS-L1 path needs NumPy and Pillow. For neural models install the wheel’s vision extra, including CPU PyTorch and safetensors. Extract the original v0.3.0-rc.1 CPU release into a directory containing release.json and models/. Its archive SHA-256 is `f555956840870309217cc85fe9b339ef53762679bbd2138b7f149d8ea451a7b7`. [Original CPU weights](https://github.com/wany614/evidence-lens/releases/download/v0.3.0-rc.1/evidence-lens-cpu-v0.3.0-rc.1.zip)

Run these commands inside the unpacked original-replays archive. Each output directory must be new or empty.

```sh
python accept.py --sequence bell-M4 --method MS-L1 --out grade-l1
python accept.py --sequence bell-M4 --method MS-ft --models /path/to/cpu --threads 4 --out grade-ft
python accept.py --sequence bell-M4 --method all --models /path/to/cpu --out grade-all
```

Each command invokes the installed package’s score and report CLI in isolated Python processes. Open the new HTML under the model’s output directory. The acceptance.json file records the real model and grid, FP32 bits including NaNs, signed and absolute differences on jointly finite pixels, threshold crossings, and changes in exact-k accepted pixel sets. Find the 20% row under `20%_canvas`. A single-model run uses historical companion scores to define four-method common support. The all-model run recomputes all four methods. Every successful comparison writes its numerical status. Add `--require-bit-exact` to return exit status 2 when one or more values differ.

Open `expected/bell-M4/MS-ft/report.html` locally for the original six-view numerical reference. The neighboring scores directory contains compressed arrays, predictions, source previews and a hash-checked numerical receipt. Inputs for this example are under `inputs/bell-M4`. The numerical-bridge.json file records the comparison with both resident cache settings. CPU scoring consumes prepared evidence. Rendering new views from a supported Gaussian asset uses the separate CUDA gsplat renderer.

Credit **wany614 / Evidence Lens** for the authored images, cameras, predictions, original example reports and measurements under CC BY 4.0. Software is Apache-2.0. The archive contains the original notices and license texts. Separate model weights retain their release terms.
