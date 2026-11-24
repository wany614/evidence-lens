# Completion matrix

The candidate matrix records the evidence available on 24 November 2026 at 05:18:12 UTC. Each item lists its status, public evidence paths, model and source identities, recorder runs and remaining limitation. The public archive verification and clean-platform checks had their own pending rows at this cutoff. The two failed T1 probes remain scientific terminal outcomes. [Candidate matrix](completion-candidate.json) · [Storage ledger](resources/README.md)

The `recorder_evidence` object binds each cited original to its run, native manifest SHA-256, member path, member SHA-256, recorder source snapshot and recorded Git commit where available. A row whose code version starts with `sha256:` points to its recorded source snapshot. The top-level repository-file digests describe the documentation and implementation as they existed when the controller bound its inputs. Later report revisions can have different digests. The release readiness flag requires the current public artifacts and their clean platform readbacks. The general completion flag also covers the hash-sealed performance and replay tables, the T1 dispositions and the Research Log.

## Reproduce a recorded cutoff

The study store contains the hash-sealed deployment records and the historical run index. Run the following with the repository's recorder environment. The controller saves one set of input bindings for each phase under that store. Repeating a completed phase verifies and returns its original recorder pin.

```sh
PYTHONPATH=src:scripts python scripts/general_resource_ledger.py --store "$STORE" --record candidate
PYTHONPATH=src:scripts python scripts/general_resource_ledger.py --store "$STORE" --export candidate
PYTHONPATH=src:scripts python scripts/general_closeout_matrix.py record --phase candidate --store "$STORE"
```

The final phase uses `--record final`, `--export final` and `--phase final`. The controller resolves the released source build, manifest, GitHub publication, anonymous download, clean Mac and Linux runs, Windows intake and native result, and post-public verification. It also reads both native CPU measurement pairs, the combined public measurement archive and the final companion-document readback. The optional `--facts FILE` can bind a role to a controller name or a four-field recorder pin under `pins`, associate a public projection with the same role under `public_evidence`, or describe a real missing dependency under `notes`. Every cited native member is checked against its sealing manifest. [Matrix controller](../../scripts/general_closeout_matrix.py) · [Storage census](../../scripts/general_resource_ledger.py)
