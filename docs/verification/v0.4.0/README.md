# Public release verification for v0.4.0

The GitHub release resolves to commit `7051de93f7b8af764c5ded4aeb623fb205b56452`. The frozen model parameters come from `v0.3.0-rc.1`. The wheel, audited source, prepared original replays, static trajectories and CPU measurements were retrieved anonymously and verified against the published manifest. [Release and checksums](https://github.com/wany614/evidence-lens/releases/tag/v0.4.0) · [Positive inventory](release-manifest.json) · [Source build](build.json)

Mac downloaded all seven files. Windows independently downloaded the wheel, original replays, manifest and checksum list. An arm64 Linux container received the same anonymously verified wheel and examples from the Mac and ran with its network disabled. Each replay installed the candidate wheel in a new environment and used the original bell M4 example with six target views. The optional methods loaded their original parameters. [Receipts and run identities](verification.json)

| Platform and installation | Method | Bit-exact targets / 6 | Changed risk pixels | Largest absolute difference | State changes | Threshold crossings | Changed quota memberships / attainable |
|---|---|---:|---:|---:|---:|---:|---:|
| linux aarch64 · default | MS-L1 | 6 / 6 | 0 | 0 | 0 | 0 | 0 / 60 |
| mac arm64 · default | MS-L1 | 6 / 6 | 0 | 0 | 0 | 0 | 0 / 60 |
| mac arm64 · vision | B-L1-transfer | 6 / 6 | 0 | 0 | 0 | 0 | 0 / 60 |
| mac arm64 · vision | MS-L1 | 6 / 6 | 0 | 0 | 0 | 0 | 0 / 60 |
| mac arm64 · vision | MS-frozen | 6 / 6 | 0 | 0 | 0 | 0 | 0 / 60 |
| mac arm64 · vision | MS-ft | 6 / 6 | 0 | 0 | 0 | 0 | 0 / 60 |
| windows AMD64 · default | MS-L1 | 6 / 6 | 0 | 0 | 0 | 0 | 0 / 60 |
| windows AMD64 · vision | MS-ft | 0 / 6 | 374,959 | 8.1062317e-06 | 0 | 0 | 2 / 60 |

The per-view tables include signed numerical differences, accepted-pixel membership, support and reason changes. Repeated methods in separate environments document two actual installs. Cross-platform figures concern this original prepared example. The frozen scientific population and camera-policy study have separate denominators. [Mac](mac/numerical-differences.csv) · [Windows](windows/numerical-differences.csv) · [Linux](linux/numerical-differences.csv) · [Prospective study](../../results/t1/README.md)

The source archive also passed 45 targeted regression tests. They cover bundle and format handling, source evidence and ordering, cache invalidation, frame and queue disposition, and independent evaluation controls. [Test run](tests.json) · [How to install and reproduce](../../releases/v0.4.0/README.md)

The initial Windows public manifest request failed TLS issuer-chain verification before a replay environment was started. The failed native run and its Mac intake disposition remain in the research archive. The recovery used the project interpreter’s existing pip CA bundle and kept hostname and certificate-chain verification enabled. The subsequent grade comes from a separately recorded anonymous download that passed HTTPS certificate, length and SHA-256 verification. [First public Windows attempt](windows/tls-first-attempt.json)

The recorded UTC start of each public download or replay follows the verified GitHub publication time. The Mac and Linux grades each identify the completed anonymous download in their native parent configuration. The Windows grade and its HTTP log share one public run. The passing Windows client used CPython urllib and its existing pip-vendored CA bundle with certificate and hostname verification enabled. [Native chronology and trust-store digest](chronology.json)

## Recorded operations

| Operation | Native run |
|---|---|
| inventory | `20261124T052727770618Z-f2df90e4ec4d44a7` |
| linux | `20261124T053334466132Z-7833aef37f6e419f` |
| mac | `20261124T053317420913Z-a2222e0c95c84aab` |
| mac anon | `20261124T053239837552Z-8ec25846c9824287` |
| publication | `20261124T053215645657Z-9531fab1637b4892` |
| regressions | `20261124T052734882331Z-389d71c83fd34246` |
| windows anon | `20261124T053921931741Z-2a518ee89cfa4186` |
| windows grade | `20261124T053921931741Z-2a518ee89cfa4186` |
| windows intake | `20261124T054246100437Z-2d21ce6c9efa4107` |
| Final cross-platform reconciliation | `20261124T054331477446Z-297fca1fb02c47eb` |
