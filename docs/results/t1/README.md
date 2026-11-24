# Prospective T1 on six DTU objects

The jointly sealed cohort supplied 240 scored slots among 288 scheduled. The source evidence, immutable fits and terminal failures had all been hash-sealed when the independent reader opened the 36 registered target photographs. The 42 logical reconstruction conditions have a separate observation and computation ledger. [Pre-opening matrix](fixed/sealed-matrix.json) · [Collective opening](fixed/opening.json) · [Native independent recount](fixed/native-recount.json)

## Per-object fixed-layout reconstruction and released MS-L1 threshold

| Object | Layout | Views scored / planned | RGB errors Y / E | Low L / V | Errors LY / L | Unknown U / E | Target MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| set118 | M2 | 6/6 | 138114/393216 | 198047/393216 | 49390/198047 | 14796/393216 | 0.0505 |
| set118 | M4 | 0/6 | 0/0 | 0/393216 | 0/0 | 0/0 | N/A |
| set76 | M2 | 6/6 | 210091/393216 | 172457/393216 | 61617/172457 | 19967/393216 | 0.1557 |
| set76 | M4 | 0/6 | 0/0 | 0/393216 | 0/0 | 0/0 | N/A |
| set45 | M2 | 6/6 | 178008/393216 | 154959/393216 | 31251/154959 | 19406/393216 | 0.1210 |
| set45 | M4 | 6/6 | 109482/393216 | 194046/393216 | 25126/194046 | 24074/393216 | 0.0684 |
| set103 | M2 | 6/6 | 346663/393216 | 34247/393216 | 27812/34247 | 7414/393216 | 0.2886 |
| set103 | M4 | 6/6 | 164123/393216 | 199114/393216 | 66958/199114 | 38725/393216 | 0.0822 |
| set61 | M2 | 6/6 | 171200/393216 | 165535/393216 | 42505/165535 | 18990/393216 | 0.1246 |
| set61 | M4 | 6/6 | 114886/393216 | 215763/393216 | 36113/215763 | 32815/393216 | 0.0763 |
| set32 | M2 | 6/6 | 200024/393216 | 173960/393216 | 51386/173960 | 30081/393216 | 0.1446 |
| set32 | M4 | 6/6 | 126893/393216 | 196986/393216 | 29545/196986 | 27760/393216 | 0.0767 |

A zero denominator makes the corresponding rate undefined. Missing M4 predictions retain every reference pixel in V. Their L/V equals zero and their error prevalence and LY/L are undefined. Finite support S, common support C, error counts and the absolute MAE sum accompany every method and view. [All 288 rows](fixed/method-views.csv) · [All 48 object–layout–method totals](fixed/method-conditions.csv) · [Unknown causes](fixed/unknown-reasons.csv)

## Four methods on identical current candidates

Each physical object contributes equally to a defined mean. Common AP and prevalence use the same defined views. The 20% column uses the registered exact per-view quota from the full crop, restricted to common score support. Coverage at the original thresholds uses each method's native application domain.

| Layout | Method | Objects with defined AP / planned | Common AP | Prevalence | Errors at 20% | Original L/V | Original LY/L |
|---|---|---:|---:|---:|---:|---:|---:|
| M2 | B-L1-transfer | 6/6 | 0.6661 | 0.5329 | 0.3750 | 0.2918 | 0.3824 |
| M2 | MS-L1 | 6/6 | 0.7295 | 0.5329 | 0.3316 | 0.3811 | 0.3621 |
| M2 | MS-frozen | 6/6 | 0.7186 | 0.5329 | 0.3233 | 0.2449 | 0.3300 |
| M2 | MS-ft | 6/6 | 0.7276 | 0.5329 | 0.3123 | 0.2856 | 0.3283 |
| M4 | B-L1-transfer | 4/6 | 0.4882 | 0.3330 | 0.1888 | 0.2339 | 0.2092 |
| M4 | MS-L1 | 4/6 | 0.5851 | 0.3330 | 0.1579 | 0.3416 | 0.1958 |
| M4 | MS-frozen | 4/6 | 0.5903 | 0.3330 | 0.1444 | 0.1945 | 0.1663 |
| M4 | MS-ft | 4/6 | 0.6013 | 0.3330 | 0.1240 | 0.2440 | 0.1642 |

## Errors among the accepted pixels at 20%

| Object | Layout | B-L1-transfer | MS-L1 | MS-frozen | MS-ft |
|---|---|---:|---:|---:|---:|
| set118 | M2 | 23191/78642 | 17959/78642 | 17365/78642 | 17276/78642 |
| set118 | M4 | N/A | N/A | N/A | N/A |
| set76 | M2 | 30097/78642 | 24318/78642 | 26888/78642 | 25946/78642 |
| set76 | M4 | N/A | N/A | N/A | N/A |
| set45 | M2 | 12251/78642 | 9992/78642 | 7205/78642 | 6601/78642 |
| set45 | M4 | 7264/78642 | 7240/78642 | 5813/78642 | 4928/78642 |
| set103 | M2 | 67527/78642 | 68280/78642 | 66800/78642 | 65960/78642 |
| set103 | M4 | 29377/78642 | 23873/78642 | 23466/78642 | 22406/78642 |
| set61 | M2 | 27290/78642 | 19731/78642 | 22416/78642 | 20148/78642 |
| set61 | M4 | 15248/78642 | 9606/78642 | 12384/78642 | 9635/78642 |
| set32 | M2 | 16590/78642 | 16202/78642 | 11878/78642 | 11436/78642 |
| set32 | M4 | 7500/78642 | 8960/78642 | 3748/78642 | 2031/78642 |

Scores tied at a cutoff use the original label-independent pixel order. Every attainable and unattainable quota, cutoff group and optimistic/pessimistic tie count is retained. The two C6 profiles have no feasible operating point. [All matched quotas](fixed/matched-views.csv) · [Object-equal denominators](fixed/object-equal.csv) · [Source self-render errors](fixed/source-fit.csv) · [Prediction and reference hashes](fixed/evaluation.json)

[Camera selection, failures and paired independent target results](nbv.md) · [Reproduction and licensed access](reproduction.md) · [Study synthesis and T0 comparison](../../general-closeout/README.md)
