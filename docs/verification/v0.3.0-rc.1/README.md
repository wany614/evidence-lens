# Public CPU verification of v0.3.0-rc.1

These checks downloaded the public pre-release from GitHub without account credentials. Each started in a fresh Python environment, verified the six asset digests and compared the executing verifier and deployed runtime with Git commit `a65b1caea258d991573bddb7a65f2bc4f9945fd7`. The default model ran using only NumPy and Pillow as application dependencies. Its scoring process received only the four-source deployment case and the required model and runtime files. [Download the pre-release](https://github.com/wany614/evidence-lens/releases/tag/v0.3.0-rc.1) · [Browse the exact source](https://github.com/wany614/evidence-lens/tree/a65b1caea258d991573bddb7a65f2bc4f9945fd7)

| Platform | Result | Distinct native comparisons | Independent target-reference checks | Largest absolute difference against CPU readback | Threshold crossings against CPU readback |
| --- | --- | ---: | ---: | ---: | ---: |
| macOS arm64, Python 3.12.16 | passed | 120 | 12 | 0.0 | 0 |
| Windows AMD64, Python 3.12.10 | passed | 120 | 12 | 8.52346420288086e-06 | 2 |
| Debian 13 Linux arm64, Python 3.12.16 (OrbStack on Mac) | passed | 120 | 12 | 8.553266525268555e-06 | 1 |

Each complete replay scheduled 144 method, source and target combinations. The two-source condition reuses the same arrays for its two-source and all-source results, leaving 120 distinct comparisons. The numerical tolerance was 0.0001. Each crossing is counted even when the numerical difference falls within that tolerance. The grade also contains a separate comparison with the original scientific producer maps. [Mac receipt](macos.json) · [Windows receipt](windows.json) · [Linux receipt](linux.json)

## Agreement with the original scientific predictions

The published CPU readback and the original scientific predictions are separate reference maps. This comparison uses each distinct method, target and source subset once. Each pixel contributes once for each of those distinct outputs.

| Platform | Finite method–pixel positions | Largest absolute score difference | Changes across the frozen threshold |
| --- | ---: | ---: | ---: |
| macOS | 7,701,159 | 1.2516975e-05 | 2 |
| Windows | 7,701,159 | 1.1414289e-05 | 0 |
| Debian Linux arm64 | 7,701,159 | 1.6093254e-05 | 1 |

For the standalone default MS-L1 output on four-source target v01, both machines scored the same 65,536 pixels. The largest absolute risk difference between these two installations was 0, with 0 state differences across the full 256 × 256 image. Both machines wrote byte-identical risk and state PNGs. [Continuous risk](../../../examples/offset-bell-cage/m4-v01-risk.png) · [Frozen state](../../../examples/offset-bell-cage/m4-v01-state.png)


Against the public CPU maps, the single Linux crossing was MS-ft, M2 score-one, v06. Against the original study maps it was MS-ft, M4 score-two, v06. For standalone MS-L1 M4 v01 the Linux score, support, state and decoded risk/state images matched the Mac output exactly. The two Linux-encoded PNG files have different byte hashes. [All Linux comparisons and reference checks](linux/grade.json) · [Standalone comparison](linux.json)

The Linux run used a local `python:3.12-slim` Linux arm64 image with a read-only root, an empty temporary filesystem and separate fresh virtual environments for the default and complete checks. It downloaded all six release assets anonymously, 264,632,173 bytes in total. Container-network counters measured 426,466,254 bytes received and 1,237,749 sent across verification and both installs. The complete verifier took 50.57 s. The container peaked at 1,782,894,592 bytes including in-memory temporary files. Its host volume retained at least 34,680,041,472 bytes free. The default installed NumPy 2.5.5 and Pillow 12.3.0. The complete environment added CPU Torch 2.8.0+cpu and safetensors 0.8.0. [Unmodified verifier receipt](linux/verification.json) · [Linux resource counters](linux/linux-resources.json) · [Host and image identity](linux/host.json) · [Lightweight wheels](linux/installation-light.json) · [Complete wheels](linux/installation.json)

## Files and hashes

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| RELEASE_NOTES.md | 2,862 | `b9d05e425046215f8a6e75578f5709a0e6d5cba3dc476050d31f86437d80fae5` |
| SHA256SUMS | 310 | `f77177ab0bab53cc839474d0d2cf0d4118b4218c752f853b8f3442ec7f0fab4d` |
| candidate.json | 3,310 | `7e767f053f9cca09072de1259433f65c0ee550d8c14a9d968579ad1cb9623669` |
| evidence-lens-cpu-v0.3.0-rc.1.zip | 164,854,827 | `f555956840870309217cc85fe9b339ef53762679bbd2138b7f149d8ea451a7b7` |
| evidence-lens-example-v0.3.0-rc.1.zip | 99,402,309 | `14204986e0d598adc613dd1a9ef2e5c1584f5be001d3541531d5d9e709707548` |
| evidence-lens-numerical-v0.3.0-rc.1.zip | 368,555 | `f28f702eb4dbaa3dc86af5d5de396f554318168707d71180fb39041b00797b2b` |

## Reproduce the public check

Clone the tagged source with Git line endings preserved. The verifier creates fresh CPU environments, fetches the three archives and their sidecars, and writes the score, risk image, state image, dependency receipts and full grade into the specified new directory.

```sh
git -c core.autocrlf=false clone --branch v0.3.0-rc.1 --depth 1 https://github.com/wany614/evidence-lens.git
cd evidence-lens
python3 -I -B scripts/verify_stage3_public.py --commit a65b1caea258d991573bddb7a65f2bc4f9945fd7 --candidate-sha256 7e767f053f9cca09072de1259433f65c0ee550d8c14a9d968579ad1cb9623669 --output ../evidence-lens-public-check
```

On Windows PowerShell, use `py -3.12` for the Python command. [Run the models on the public example](../../releases/v0.3.0-rc.1/README.md)

## Recorded origins

| Action | Run ID | Manifest SHA-256 |
| --- | --- | --- |
| Release build | `20261123T173212401848Z-7045443abe424128` | `7af2c73e9fad9a0a85ba09c1789feef7380fc05aa39c6439676f9de431eb94f3` |
| Independent rebuild | `20261123T173221320686Z-8462c41334d74751` | `f694c3af2ef1b5f0d1420f71f7ae9d16dc0d708f4c2ca33a1850c59054aa28ad` |
| Local clean CPU installation | `20261123T173246730789Z-c55444dc1cc14de1` | `af179c9e79775d6fac17d0bdca00ddd5ae494253411a095b1523604b07393db1` |
| Anonymous macOS verification | `20261123T173512903623Z-ef426005a252419e` | `db4a59e5a4790bd8fc4e612c93d373fba60a3675300589fbe978b8e4419c3831` |
| Anonymous Windows verification | `20261123T174113206864Z-ddf9de90b4e34500` | `fd3c40928530570af21b75b20dce7233d3645fe8d03754cb7ca81f2c18238405` |
| D0 Linux setup, host process launch failed | `20261123T193226173907Z-69f75b788afe4c7a` | `18a4db23f55f3aef4a30af75b9e9325fca5f2d58e07ebf35cd398589df2c49c3` |
| D0 anonymous Linux arm64 verification | `20261123T193254087345Z-e06c25cca15b4b9d` | `aae7991a93be393ad2d6723f404dee0bf386e5626d3481ed19e19353d14f5ab8` |

The first Linux setup used a relative host interpreter path that was unavailable from the recorder working directory. It stopped before starting the container or consuming network bytes. The successful command used the absolute host interpreter path and the same fixed public release. [Disposable Linux wrapper](../../../scripts/d0_linux_grade.py)
