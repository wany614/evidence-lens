# T0 object and photograph intake

The frozen cohort contains six independently named DTU subjects. The study uses all pixels in the registered crop, including the common laboratory background and any incidental props. The independent authors’ object labels describe the principal subject in each scene. The identities support a six-family comparison under one calibrated acquisition rig. The common laboratory setting and possible accessory reuse limit the scope of that comparison. [DTU dataset](https://roboimagedata.compute.dtu.dk/?page_id=36) · [Kim et al., Appendix Table 12](https://arxiv.org/html/2610.06472v1#Pt0.A2.T12)

| Official scan | Documented principal subject |
| ---: | --- |
| 2 | Rabbit statue |
| 4 | Bird doll |
| 10 | Cardboard box |
| 11 | Power supply |
| 12 | Beverage cans |
| 31 | Pumpkin |

## Physical identity and ordering

The v1.2 protocol determines the order by ascending official scan number within eligible physical families. The authors of the DTU paper describe 80 “different scenes” on page 1 and give category counts on page 4. Their evaluation script identifies the 80 scan numbers. The physical identity of individual specimens remains unspecified in those two sources. We cross-checked the published principal-subject labels against the complete rectified archive, the explicit rotation groups in the DTU ReadMe and the two previously exposed objects. Earlier work had identified scan 1 as a painted tapered container and scan 6 as an architectural miniature. [Original registration](../../../../protocol/reconstruction-recovery-v1.json) · [DTU journal paper](https://roboimagedata2.compute.dtu.dk/data/text/IJCV_MVS_Stereo_data_2016.pdf) · [Earlier subject record](../../dtu-external/README.md)

All 124 archive IDs contain the ten prescribed members. Six were selected, sixteen additional documented families exceeded the registered budget, fifteen entries shared a documented rotation group or a potentially repeated subject, and eighty-five remained unresolved. The two previously exposed subjects were excluded. Scans 3, 5, 7, 8 and 9 lacked independent subject-level identification in the textual evidence. The same criterion applied to the later unresolved IDs. This sample therefore favors subjects with a published identity mapping. A future study could establish more identities before registering its own cohort. The acquisition-side lookup used the scan number and object columns of the independently authored table. Its method-specific measurements played no part in ordering. [Full scan-by-scan decisions](physical-eligibility.csv) · [Authors’ configuration table](https://arxiv.org/html/2610.06472v1#Pt0.A2.T12)

The official ReadMe lists the inclusive ranges 55–58, 65–68, 69–73, 106–109, 110–113, 114–117, 118–121 and 122–125. The 69–73 range contains five IDs as printed. We retained it as one group. We also grouped uncertain recurrences involving bird dolls, boxes, cans, brick subjects and stone statues. Roster membership was fixed before triangulation, reconstruction and evaluation. [Full scan-by-scan decisions](physical-eligibility.csv) · [Official SampleSet and ReadMe](https://roboimagedata.compute.dtu.dk/?page_id=36)

## Camera and transfer evidence

All six objects use light 3 and the fixed 256 × 256 display-RGB conversion. The four source captures are 24, 23, 17 and 22. M2 takes the endpoints 24 and 22. M4 takes all four. The six held-out positions are 16, 18, 25, 37, 35 and 36. The rectified right-camera calibration was tied back to the earlier validated metric rig through its camera files and twenty cross-archive size/CRC matches on the two exposed objects. [Camera protocol](../../../dtu-camera.md) · [Source intake receipt](source-intake.json)

The official archive contains 54,373 directory records and occupies 129,593,443,783 bytes. The first request retrieved 29,237 bytes of official catalogue HTML. Exact ZIP64 metadata ranges yielded the location and CRC of each registered member. Twelve source acquisitions returned 24 unique photographs through 84 HTTP 206 responses. Response bodies totalled 63,049,695 bytes, including archive headers. The 24 original PNGs occupied 63,048,047 bytes. M4 reused 31,621,394 bytes of sealed M2 endpoint PNGs. All twelve condition acquisitions succeeded on their first attempts. Independent verification checked original CRC, SHA-256, exact requested byte ranges and equality of the shared endpoints before and after cropping. [Transfer ledger](source-transfers.csv) · [Source hashes](source-hashes.csv)

Before the first network request, an initial recorder invocation rejected a missing parent. Supplying the verified v1.2 registration allowed the first official HTTP request to start under the recorder. The metadata ledger retains three sealed engineering failures. One parser initially used an incompatible private Python ZIP64 interface. One publication parser matched identically headed tables in two datasets. One attempt to bind the unsuccessful first HTML parse was rejected by the recorder before a new request. The successful replacement used the versioned author publication and selected only DTU rows. All three occurred before the object register and acquisition freeze. Recorded metadata requests consumed 48,868,726 bytes in total, including the failed attempts. The HTTP ledger counts response-body bytes. [Metadata operations](metadata-operations.csv)
