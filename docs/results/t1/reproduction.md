# Reproducing the T1 evidence chain

The scientific baseline is commit `cdf2267a8775030cf26616aeaa472f38ec5b2f7f`. The tested risk weights are the four released Stage B/C packages identified in the T1 freeze. The fit uses the registered Windows CUDA/gsplat worker, the cold-start recipe with seed 43171, the original source-only 24-track gate and the 2,048-point fallback. The recorded optimizer runs have 5,000 updates and five issued checkpoints. [Frozen T1 method](../batch-two/holdout/t1-freeze.json) · [Endpoint protocol](../../../protocol/b2-t1-confirmation-v1.json)

## Obtain the sealed scientific implementation

From a checkout of the general release, create a separate worktree for the programs used by the study. Commit `c19b4235beedd905f7bb0e38c824efaf6b3bcc0e` sits directly on the baseline and includes the controllers and evaluation wrappers whose bytes were sealed before reference access. The engineering release contains cache optimizations in two feature implementations bearing the same names. The worktree selects the registered scientific versions for all numerical imports. The later resource exporter can be copied separately from the general release.

```sh
general_release_commit=$(git rev-parse HEAD)
git worktree add --detach ../evidence-t1-science c19b4235beedd905f7bb0e38c824efaf6b3bcc0e
git show "$general_release_commit:scripts/t1_resources.py" > ../evidence-t1-science/scripts/t1_resources.py
cd ../evidence-t1-science
```

The resource exporter's SHA-256 is `48ab4347228121e27e980f8ffb44c3bfa3dd641c8d0b29c725250ac51b2bc2d9`. The original evaluation and independent verification run records each retain their full scientific source snapshot.

## Private prerequisites

The Mac control store contains the current study's `runs/` and `work/controller/`. Its `history/runs/` resolves the signed members of the historical Evidence Lens, batch-one and batch-two stores. The Windows compute store and the private cross-device handoff and return directories are separate. Access to the original DTU light-3 archive and the frozen camera/weight records is necessary for photographic reconstruction and independent reference evaluation. The public numerical tables contain run identities, input hashes and counts. DTU photographs, trained scene checkpoints and photo-derived prediction arrays stay with the private records.

Use the environment containing the original scorer and full real model packages. Set `PYTHONPATH` to the scientific checkout's `src` and `scripts`. Each child recorder uses its own archived source snapshot. Pass the private study root explicitly with `--store` or set `EVIDENCE_LENS_STUDY_ROOT`. The original triangulation interpreter can be supplied through `EVIDENCE_LENS_GEOMETRY_PYTHON`. The CPU Torch interpreter for checkpoint auditing can be supplied through `EVIDENCE_LENS_READBACK_PYTHON`.

## Original stage order

1. Verify the historical identities and the new G0 entry. The original batch-two freeze continues to record its zero-access cutoff. The G0 entry authorizes opening the two S0 source images in the order 24, 22.
2. On the Mac, `t1_study.py m2` acquires S0, runs the original source-only initializer and writes six ordered jobs. `t1_probe_kit.py issue` packages the frozen probe and its 14 permitted camera poses. Each job binds a source manifest, exact ordered images, both camera files, initializer arrays and metadata, seed, numerical worker and recipe.
3. On Windows, run the issued, SHA-verified `b2_worker_queue.py` against the job registry. The original optimizer sees only the registered source camera file, observations, initializer and recipe. Its separate endpoint renderer requires six independent-camera predictions whenever it writes the source self renders. Only the two native source self renders can enter the selector. Run the issued `t1_probe_windows.py` against the final M2 generation, then return all original archives, side receipts and cumulative ledgers through the private exchange.
4. On the Mac, `t1_queue_intake.py` verifies every return, checks the five checkpoint states and native output semantics, seals each strategy decision and prepares authorized M4 jobs. The formal fixed order is 24, 23, 17, 22. The selected order is 24, lower selected ID, higher selected ID, 22. A distinct receipt grants each successful trajectory its own observations. A digest match for the entire scientific input and configuration permits sharing one immutable fit.
5. The frozen probe tests for exactly 20,000 final Gaussians. An otherwise valid M2 below this count can still be scored at the six independent cameras. A sealed probe failure gives its object zero candidate-photo authority. The five strategy terminal receipts and all six dependent M4 conditions retain their planned denominators. The three hash-derived random assignments remain precomputable registration information. The failed selector made no actual selection. [Frozen probe](../../../scripts/b2_probe_worker.py)
6. Seal the 42 condition terminals, the fixed-layout score slots, and the evaluation program. The joint opening verifies the common 42-condition and 288-slot roster. Only this one opening enables acquisition of all 36 reference photographs in the registered order for each object.
7. On the existing closed study, `t1_nbv_evaluate.py evaluate` reads the original target predictions and reference PNGs for all 252 scheduled logical views. `t1_nbv_independent.py audit` reads the original FP32 arrays, counts and error sums separately. `g1_evaluate.py` and `g1_recount.py` evaluate and independently check the formal four-method fixed-layout matrix. `t1_resources.py census` inventories the actual original attempts and charges shared fits once.

For an existing completed study, use its selected four-field run pins to recover exact bytes. Further calculations on its opened targets inherit their observed status. The recorded object order is set118, set76, set45, set103, set61, set32. [Frozen camera selection](../../../protocol/b2-nbv-v1.json)

## Read back the closed study

Run the following from the scientific checkout using the study's original environment. Set `EVIDENCE_LENS_STUDY_ROOT` to the private study directory containing its verified runs and history. The first two commands validate and return the sealed evaluation pins. The last copies the signed numerical tables and trajectory records into the result directory and verifies existing copies.

```sh
export PYTHONPATH="$PWD/src:$PWD/scripts"
python scripts/t1_nbv_evaluate.py evaluate --store "$EVIDENCE_LENS_STUDY_ROOT"
python scripts/t1_nbv_independent.py audit --store "$EVIDENCE_LENS_STUDY_ROOT"
python scripts/t1_resources.py publish --store "$EVIDENCE_LENS_STUDY_ROOT" --dest docs/results/t1
```

The primary endpoint was recorded by run `20261124T044825489277Z-1ca8f9b1522541fe`. The separate raw-array recount was recorded by run `20261124T044829368776Z-3f0b55a69aa44e2d`. Both identify the same joint opening, `20261124T044751135267Z-7eafdcf5d2a44153`. [Endpoint receipt](nbv.json) · [Independent recount receipt](independent-nbv.json)
