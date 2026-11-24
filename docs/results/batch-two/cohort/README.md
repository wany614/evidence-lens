# DTU physical families for calibration and confirmation

Sixteen documented principal subjects were available after the original six-object T0 confirmation. Every family passed the same check for 16 distinct official camera captures under DTU light 3. The allocator used archive-directory entries and right-camera matrices. It read zero photograph bodies. The grouping retains the documented rotations and potential repeats from T0. Subjects share the laboratory setup and may share incidental accessories. [Physical identity evidence](../../t0/intake/README.md) · [Full numerical cohort](source-intake/physical-families.csv)

| Role | Official scans in registered sequence |
| --- | --- |
| C6, empirical calibration | 48, 75, 110, 42, 77, 52 |
| T1, reserved for subsequent confirmation | 118, 76, 45, 103, 61, 32 |
| Reserves | 37, 54, 34, 62 |

The allocator ranked all eligible physical-family identifiers by the SHA-256 of the UTF-8 string `evidence-lens-b2-split-v1|137031|` followed by the identifier. The numeric scan ID breaks an equal digest. The first six went to C6, the next six to T1, and the rest to reserve. The register retains the pre-existing grouped identities, including scans 110–113 for the bronze statue and scans 118–125 for possible stone-statue recurrence. No outcome-based substitution was permitted. [Prior scan-by-scan decisions](../../t0/intake/physical-eligibility.csv) · [Allocation and geometry receipt](source-intake/cohort-summary.json)

## Fixed acquisition and camera eligibility

M2 uses endpoint views 24 and 22. M4 uses 24, 23, 17 and 22. The eight camera candidates are 7, 10, 13, 17, 20, 23, 31 and 40. Evaluation uses 16, 18, 25, 37, 35 and 36. The maximum discrepancy between the original native projection and the 256-square cropped metric rig was 0.00002339 pixels over the registered positions and depths, within the 0.005-pixel check. All six evaluation positions were excluded from the candidate-camera file supplied to the selector. [Rig checks](source-intake/rig-checks.csv) · [Metric camera and crop](../../../dtu-camera.md)

The 12 C6 source conversions used 24 different original photographs. The 84 exact HTTP requests returned 43,966,976 response-body bytes, including ZIP headers. Extracted source PNGs occupied 43,965,736 bytes. The M4 inputs reused 22,648,839 bytes of the sealed M2 endpoint photographs. The two manifest entries for each shared endpoint carry identical original and cropped hashes. [Source transfers](source-intake/transfers.csv) · [Source photograph identities](source-intake/photograph-hashes.csv)

DTU asks users to cite Jensen, Dahl, Vogiatzis, Tola and Aanæs, *Large Scale Multi-view Stereopsis Evaluation*, CVPR 2014. This directory distributes counts and file identities. Photographs and native crops remain in the private research archive. [Official DTU catalogue](https://roboimagedata.compute.dtu.dk/?page_id=36) · [DTU paper](https://roboimagedata2.compute.dtu.dk/data/text/multiViewCVPR2014.pdf)

The collective C6 reference opening followed all 288 planned view-method outputs and the score-only commitment to 2,814 full or leave-one-out candidate cutoffs. Six recorded acquisitions then obtained the 36 target photographs using 114 HTTP ranges and 68,212,120 response-body bytes. Across sources and targets, 60 different DTU photographs were retrieved. T1 photograph reads remained zero. [Complete transfer and camera receipt](complete-intake/cohort-summary.json) · [All registered photograph hashes](complete-intake/photograph-hashes.csv) · [Empirical calibration findings](../calibration/README.md)
