# Public CPU replay of v0.2.0-rc.1

On 23 November 2026 an Apple silicon Mac downloaded all seven files from the public GitHub pre-release and installed the scoring dependencies into a fresh Python environment. The downloader used anonymous HTTPS. It sent no authentication or cookies. The verification program checked its own source against the Git tree at the tagged commit. [Release downloads](https://github.com/wany614/evidence-lens/releases/tag/v0.2.0-rc.1) · [Tagged source](https://github.com/wany614/evidence-lens/tree/fb2fbdc232cf2be706ca88e4ad9ddfe884013cc3)

| Mac measurement | Observed result |
| --- | --- |
| Runtime | arm64 CPU, Python 3.12.16, PyTorch 2.8.0, NumPy 2.5.5 |
| 128 × 128 example | Seven methods on six target views, 36 risk and decision PNGs |
| 512 × 512 attachment | Seven methods on six targets, 11,010,048 method–pixel positions |
| Independent Blender comparison | Matched the recorded 512 example reference result |
| Download, installation and both examples | 42.26 seconds |
| Recorder run | `20261123T153606811032Z-20d1b1357efe4d4c` |
| Recorder manifest SHA-256 | `ba0e441ed58252faf373c3bfa0c65967dc4b50bbf3b937c159fd3b8b27fc5e2e` |
| Public replay receipt SHA-256 | `a340da6f49ab361a64bc4fd5400f2f1b4cee2d0487574c9ad73255e75738f9b1` |

The receipt includes the downloaded byte counts and SHA-256 for all seven assets, the sources and digests of the dependency wheels, the deployed model identities and the independent evaluation digest. [Mac replay receipt](macos.json)

## Windows CPU replay

Windows completed the public downloads and produced all risk and state maps for both examples. The 128 × 128 case passed all 42 method–view checks. In the 512 × 512 case, 41 of 42 passed. At pixel (91, 260) of view v05, L2-frozen scored 0.5428744555 on the reference Mac and 0.5428719521 on Windows. Its frozen threshold was 0.5428733230. This changed one high decision to low among 11,010,048 checked method–pixel positions. The finite masks and the source, prediction and measured-error figures agreed. The verifier requires identical states and exited with code 1.

The clean Windows run used Python 3.11.5, PyTorch 2.8.0+cpu and NumPy 2.2.6. It took 134.33 seconds and reached a sampled process-tree peak of 448,319,488 bytes. Recorder `20261123T154033371799Z-78c6f6eaa3dc4162` preserves the attempt. The largest observed native-case numerical discrepancy was 0.00001181. The attached machine-readable files record the affected counts and the exact reference comparison. [Windows run summary](windows.json) · [Original Windows comparison](windows-control-readback.json)

## Repeat on a CPU

Use Python 3.11 or 3.12. Clone the public repository, check out the exact commit below and run the verifier from that checkout. The output path must be empty. The verifier downloads and checks the assets, installs the dependencies, runs both original examples and writes `public-verification.json`. On Windows, run the same arguments with a single-line PowerShell command and the selected Python interpreter.

```sh
git -c core.autocrlf=false clone --branch v0.2.0-rc.1 --depth 1 https://github.com/wany614/evidence-lens.git
cd evidence-lens
git rev-parse HEAD
python3 -I -B scripts/verify_vision_public.py --tag v0.2.0-rc.1 --commit fb2fbdc232cf2be706ca88e4ad9ddfe884013cc3 \
  --candidate-sha256 52a13598378813fca132f122d8f7bc6d8b6ca47a57cff81bc40b1a3e9707289d \
  --attachment evidence-lens-v0.2.0-rc.1-public-plane-512.zip=0a2c80c15e9ad6eb15e15707880bdd269fe70899c07ae91cf9ffc3a13fd7d0db \
  --public-case-asset evidence-lens-v0.2.0-rc.1-public-plane-512.zip \
  --public-case-sha256 1323cd8fd285eea2af58133d2b7f63fefed676fd1e44d6eb43e6795ab85a97e0 \
  --require-full --output ../evidence-lens-public-replay
```

To run only the fitted logistic model in a NumPy and Pillow environment, use the separate lightweight instructions. [CPU scoring options](../../releases/v0.2.0-rc.1/README.md)
