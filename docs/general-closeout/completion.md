# Completion matrix

The final matrix records `software_release_ready=true` and `general_phase_complete=true` for v0.4.0 at 05:48:11 UTC on 24 November 2026. It contains 37 items. Thirty-one are complete, and four measured targets had negative outcomes. These were the two C6 thresholds recorded together, full-frame 30 Hz delivery on Mac, and full-frame 10 Hz and 30 Hz delivery on Windows. One item retains the two scientific probe failures, and one marks additional training and embodied control outside this study. [Final matrix](completion-final.json) · [Public release verification](../verification/v0.4.0/README.md) · [Final storage census](resources/final/resources.json)

All six target views of the original bell-cage M4 example reproduced the default MS-L1 output on Mac, Windows and the Linux arm64 container. Windows MS-ft had 374,959 changed risk values over those six views, with a maximum absolute difference of 8.1062 × 10⁻⁶. Finite masks, reasons and threshold states matched. At two of 60 attainable view–quotas, one accepted pixel replaced another. The affected quotas were 1% on v02 and 20% on v06. The initial Windows TLS attempt and subsequent successful verified download remain individually traceable. [Windows numerical differences](../verification/v0.4.0/windows/numerical-differences.csv) · [Public timing and certificate verification](../verification/v0.4.0/chronology.json)

## Prepublication record

The candidate matrix records the evidence available on 24 November 2026 at 05:18:12 UTC. Each item lists its status, public evidence paths, model and source identities, recorder runs and remaining limitation. The public archive verification and clean-platform checks had their own pending rows at this cutoff. The two failed T1 probes remain scientific terminal outcomes. [Candidate matrix](completion-candidate.json) · [Storage ledger](resources/README.md)

The `recorder_evidence` object binds each cited original to its run, native manifest SHA-256, member path, member SHA-256, recorder source snapshot and recorded Git commit where available. A row whose code version starts with `sha256:` points to its recorded source snapshot. The top-level repository-file digests describe the documentation and implementation as they existed when the controller bound its inputs. Later report revisions can have different digests. The release readiness flag requires the current public artifacts and their clean platform readbacks. The general completion flag also covers the hash-sealed performance and replay tables, the T1 dispositions and the documentation checks recorded at that cutoff.

## Reproduce a recorded cutoff

The study store contains the hash-sealed deployment records and the historical run index. Run the following with the repository's recorder environment. The controller saves one set of input bindings for each phase under that store. Repeating a completed phase verifies and returns its original recorder pin.

```sh
PYTHONPATH=src:scripts python scripts/general_resource_ledger.py --store "$STORE" --record candidate
PYTHONPATH=src:scripts python scripts/general_resource_ledger.py --store "$STORE" --export candidate
PYTHONPATH=src:scripts python scripts/general_closeout_matrix.py record --phase candidate --store "$STORE"
```

The final phase uses `--record final`, `--export final` and `--phase final`. The controller resolves the released source build, manifest, GitHub publication, anonymous download, clean Mac and Linux runs, Windows intake and native result, and post-public verification. It also reads both native CPU measurement pairs, the combined public measurement archive and the historical documentation check. The optional `--facts FILE` can bind a role to a controller name or a four-field recorder pin under `pins`, associate a public projection with the same role under `public_evidence`, or describe a real missing dependency under `notes`. Every cited native member is checked against its sealing manifest. [Matrix controller](../../scripts/general_closeout_matrix.py) · [Storage census](../../scripts/general_resource_ledger.py)
