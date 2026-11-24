# Installed CPU platform verification

Each host installed the audited 0.4.0 wheel, replayed the six original bell M4 views, installed 0.5.0 in the same environment, and repeated that replay. Risks, validity, state and unknown-reason NPY records agreed byte for byte across versions on each tested host. Each new installation also scored six prepared bundles, checked its file results against the resident API with its source cache off and on, and rendered an HTML report for each bundle and method. [Platform, dependency and grid receipts](platforms.json)

| Native CPU and install | Methods | Original method–views exact · channels | Six-profile method–views | Resident checks | HTML reports |
|---|---|---:|---:|---:|---:|
| Mac arm64 · default | MS-L1 | 6 · 24 | 36 | 72 | 6 |
| Mac arm64 · vision | B-L1-transfer, MS-L1, MS-frozen, MS-ft | 24 · 96 | 144 | 288 | 24 |
| Linux aarch64 · default | MS-L1 | 6 · 24 | 36 | 72 | 6 |
| Windows AMD64 · default | MS-L1 | 6 · 24 | 36 | 72 | 6 |
| Windows AMD64 · vision | B-L1-transfer, MS-L1, MS-frozen, MS-ft | 24 · 96 | 144 | 288 | 24 |

The Linux arm64 container ran with a read-only root, its network disabled and loopback as its only visible interface. Its fresh default installation used NumPy and Pillow. Optional neural installation on Linux remains unmeasured. Mac and Windows installed the vision extra and loaded the complete original MS-frozen and MS-ft tensors. [Mac official-model checks](../../releases/v0.5.0/NEURAL_GRID.md) · [Windows official-model checks](neural-windows.json)

The Windows CPU recorder ran for 321.375 seconds. Its window overlapped a separate GPU control for 4.368 seconds. The platform table gives observed scoring costs for these installation checks. [Native timing and dependency evidence](platforms.json)

## Direct comparison with Mac

Each method contributes 42 paired views, comprising six historical controls and six views on each of the six prepared routes. The comparison reads the two platforms’ original risk, validity, state and unknown-reason arrays. Differences use host minus Mac on common finite support. The two quotas select the lowest 10% and 20% of eligible pixels separately in each view, breaking equal-risk ties by row-major pixel index. The downloadable per-view records also give each four-method common-support comparison where available. [Linux per-view differences](views-linux.json) · [Windows per-view differences](views-windows.json)

| Host · method | Bit-exact views | Common finite pixels | Changed risk words | Max absolute risk difference | Valid / state / reason changes | Threshold crossings | Changed quotas / tests |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linux · MS-L1 | 42 / 42 | 3,442,559 | 0 | 0 | 0 / 0 / 0 | 0 | 0 / 84 |
| Windows · B-L1-transfer | 42 / 42 | 3,492,889 | 0 | 0 | 0 / 0 / 0 | 0 | 0 / 84 |
| Windows · MS-L1 | 42 / 42 | 3,442,559 | 0 | 0 | 0 / 0 / 0 | 0 | 0 / 84 |
| Windows · MS-frozen | 0 / 42 | 3,442,559 | 3,346,406 | 1.573563e-05 | 0 / 0 / 0 | 0 | 11 / 84 |
| Windows · MS-ft | 0 / 42 | 3,442,559 | 3,328,480 | 1.436472e-05 | 0 / 2 / 0 | 2 | 7 / 84 |

The numerical archive records per-grid signed ranges, means, counts and both quota memberships. These comparisons use one authored bell and its fixed deployment evidence. They measure software and numerical reproducibility on the tested hosts. [All per-grid figures and recorder pins](platforms.json)
