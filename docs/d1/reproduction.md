# Reproducing the DTU input registration

The public input replay retrieves the official archive index, its ReadMe and calibration files by HTTP range. It recreates the 256-pixel camera grid from the original projection matrices and the registered synthetic camera layout. The command needs no photographs, reference scan, GPU or Blender. Python 3.12.16 and SciPy 1.18.1 were used on arm64 macOS. The image and score paths use their separate dependencies. [DTU source](https://roboimagedata.compute.dtu.dk/?page_id=36) · [Frozen study](../../protocol/dtu-external-v1.json)

From the repository root, with 30 GiB of free space remaining after the run, use an empty research directory outside the checkout.

```sh
python3.12 -m venv .venv-dtu
. .venv-dtu/bin/activate
python -m pip install -e '.[record]' 'scipy==1.18.1'
python scripts/dtu_replay_inputs.py --store ../evidence-lens-dtu-replay
```

Each step prints its recorder receipt. The last line identifies the new camera directory and the expected and measured SHA-256 of `camera-plan.json`. The independent Mac replay recovered `e688ed923a57c65bc73be6a2ca794e8cc23ea554474ee825d65a92450fddbdbd`. Different linear algebra libraries can change the last digits of serialized camera values. The projection and checkerboard receipts give errors in pixels and metres for physical comparison. [Camera conversion and probes](../dtu-camera.md)

To retrieve only the four real development source photographs under the recorder, use the same environment. They remain in your private store. The command checks the official archive identity, the four selected member names and their CRCs, and records a SHA-256 for each downloaded member.

```sh
python -m evidence_lens.recorder --root ../evidence-lens-dtu-replay --source-root . \
  --kind archive_verify --experiment d1-local-development --trial set1-source \
  --config configs/dtu-v1/set1-source.json \
  --input view-set=configs/dtu-v1/view-set.json \
  -- '{python}' '{source}/scripts/dtu_sample.py' --request '{config}' \
  --view-set '{input:view-set}' --out '{out}'
```

The independent target download uses the `set1-truth.json` template and its own trial. The original study evaluated set6 once. Repeating it is a reproduction on an exposed object. The acquisition program requires both the frozen study and a source-opening receipt for those four source photographs. Its six target photographs use a distinct reference-release receipt derived from both terminal scoring conditions. [Evaluation inputs, exact scores and independent checks](evaluation.md)

The Windows worker batch contains the source PNGs, source and render cameras, optimizer configuration and versioned worker code. The Mac case builder consumes the sealed returned reconstruction and renders. The four CPU score paths come from the original release ZIP. The reference converter and evaluator run separately. The historical recorder IDs and the full-resolution photographs, returned NPZs, checkpoints and boards belong to the research archive. The public tables contain the role and hash information required to identify them. [Published scorer and original synthetic example](https://github.com/wany614/evidence-lens/releases/tag/v0.3.0-rc.1)
