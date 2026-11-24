# Install the general CPU tools

The pure-Python wheel installs the `evidence-lens` command. The complete input for scoring is a prepared Evidence Bundle. The original replay archive supplies the prepared bell-cage and helical-bead scenes, reference score arrays and offline HTML. Download the wheel, `evidence-lens-original-replays-v1.zip`, `release-manifest.json` and `SHA256SUMS` from the same release. Compare the manifest digest and asset sizes with the release notes before unpacking. [Public releases](https://github.com/wany614/evidence-lens/releases) · [Positive inventory and anonymous download program](../../../scripts/general_public_download.py)

In a new Python 3.12 environment, install the wheel. Extract the original replay ZIP into its own directory. Run the following from that directory, using the path to the downloaded wheel.

```sh
python -m pip install /path/to/evidence_lens-0.4.0-py3-none-any.whl
evidence-lens validate --bundle inputs/bell-M4
evidence-lens score --bundle inputs/bell-M4 --model MS-L1 --out my-scores
evidence-lens report --scores my-scores --out my-report.html
python accept.py --sequence bell-M4 --method MS-L1 --out checked-l1 --expected-sha256 7aed41037e7c4ec7f0a8583d33bdbe5291d6b02bf0ef8df325a13d337490f9cc
```

Open `my-report.html` locally. `checked-l1/acceptance.json` identifies the installed package and compares the new risk, validity, state and unknown-reason arrays with the released originals. It also compares the original threshold and the identities of pixels selected at the same coverage. The full report records the environment and actual differences. Use a fresh directory for each new output. [Original replay structure and numerical criteria](../../tooling/original-archive.md)

## Choose the download for your task

| Filename | Contents |
|---|---|
| `evidence_lens-0.4.0-py3-none-any.whl` | Installable CLI, resident array API, source cache and the default model. |
| `evidence-lens-original-replays-v1.zip` | Four prepared original workloads, all four models’ frozen score arrays, sixteen offline reports and the executable comparator. |
| `evidence-lens-static-trajectories.zip` | Two authored scenes, M2/M4, four 120-pose sequences and two 24-pose revision sequences. Supply these inputs to the continuous replay. |
| `evidence-lens-cpu-replay-measurements.zip` | The recorded CPU configuration samples, per-frame queue ledger, stage timings, resource samples, output arrays for an authored example and its HTML report. Read the embedded `tables.json` for member digests. |
| `evidence-lens-source-v0.4.0.zip` | The audited code, frozen protocols, permitted numerical research reports, build and reproduction scripts. |
| `release-manifest.json` and `SHA256SUMS` | Exact code and model lineage, lengths and SHA-256 digests. |

The complete four-model weights have a separate original download. The verified downloader below retrieves them. [Installation and original-model provenance](MODEL_CARD.md) · [Open an authored four-model report](../../../examples/general-four-models/README.md)

## Run the genuine fine-tuned model

From the matching source checkout or extracted source ZIP, use the downloader to fetch and verify the complete original CPU release. It validates the archive, every release member and the retained preprocessing and weights. The vision extra installs the optional CPU model dependencies. Install the wheel in the same environment that will run `accept.py`.

```sh
python -m pip install "./evidence_lens-0.4.0-py3-none-any.whl[vision]"
python scripts/general_release_weights.py --out weights
```

In the extracted original replay directory, pass the absolute path to the weights directory that contains `release.json`.

```sh
python accept.py --sequence bell-M4 --method MS-ft --models /path/to/weights/cpu --threads 4 --out checked-ft --expected-sha256 7aed41037e7c4ec7f0a8583d33bdbe5291d6b02bf0ef8df325a13d337490f9cc
```

Set `--method all` to replay B-L1-transfer, MS-L1, MS-frozen and MS-ft and obtain all four HTML reports. The `--threads` option controls PyTorch's neural CPU path. The NumPy path has no equivalent setting in that option. Both layout-specific C6 profiles have an undefined adapted threshold. The original MS-L1 cutoff is 0.06428983435034752. [Full model identities and license scope](MODEL_CARD.md) · [C6 M2 profile](../../results/batch-two/calibration/profiles/M2.json) · [C6 M4 profile](../../results/batch-two/calibration/profiles/M4.json)

## Reuse and verify

For repeated calls, `load_scorer`, `prepare_sources` and `score_frame` operate on identity-bound arrays. The CLI retains its original file reference path. Use the bounded static replay program with the separate original trajectory asset. Camera and frame timestamps, completed results, drops, stale outputs, map/source revisions, feature time, service time and queueing have separate records. [Resident array API](../../tooling/resident.md) · [Numerical bridge](../../tooling/numerical-bridge.md) · [Static replay and timing](../../general-closeout/performance/README.md)

Rendering a supported Gaussian asset requires its separate CUDA renderer. The risk scoring path uses the prepared numerical evidence. The published source includes positive manifests, registration, numerical study results and reproduction scripts. The DTU photographs and their derivative panels remain in the private research archive. [Input support](../../tooling/assets/README.md) · [Research report](../../general-closeout/README.md) · [Release asset rights](RIGHTS.md)

## Checks on the actual downloadable wheel

The current release was downloaded anonymously and replayed on Mac arm64, native Windows x86-64 and an arm64 Linux container. The platform report includes the exact code/weight identities and observed numerical differences. [Public verification and per-view tables](../../verification/v0.4.0/README.md) · [Study completion matrix](../../general-closeout/completion-final.json)

The embedded Windows test interpreter initially reported TLS issuer verification failure. A new recorded run used the CA bundle from its existing pip installation, checked each HTTPS hostname and successfully downloaded the public files. The exact CA digest, interpreter client and server URLs appear in the independent readback. [Windows trust and time receipts](../../verification/v0.4.0/chronology.json)
