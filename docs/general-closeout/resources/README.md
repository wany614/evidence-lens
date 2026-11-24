# General-stage storage census

## Final census

Between 05:47:19.622 and 05:47:21.037 UTC on 24 November 2026, the Mac counted 293 sealed native records. Of these, 286 succeeded and seven failed. The 158,040 regular-file paths in those records held 8,736,596,818 logical bytes. The seven failures include the two frozen T1 probes, the diagnostic-parent check, an original-archive smoke attempt, a research-log readback attempt, the first Windows public TLS request and the Mac intake that rejected its unsuccessful result. The later Windows public replay and independent verification have their own successful native records. The census cutoff precedes its own seal and the final completion matrix. [Final native run table](final/runs.csv) · [Final census and file digests](final/resources.json) · [Recorder origin](final/origin.json)

The whole local general store contained 159,946 regular-file paths and 9,816,679,787 logical bytes, with 10,238,738,432 bytes in per-path allocated blocks. Counting distinct inodes once gave 7,134,841,857 logical bytes and 7,508,586,496 allocated bytes. The scan skipped 1,468 symbolic links and observed zero entries vanish during enumeration. The volume had 1,007,782,342,656 bytes free. APFS extent sharing can reduce the storage underlying different inodes. [Final component table](final/components.csv)

## Prepublication census

The prepublication Mac census counted 257 sealed native records between 05:11:22.803 and 05:11:23.926 UTC on 24 November 2026. Of these records, 253 succeeded and four failed. Their 133,719 regular-file paths held 5,742,500,840 logical bytes. The run table gives the native completion manifest, status, recorder wall interval and sampled peak process-tree RSS where available. Overlapping recorder intervals include concurrent and nested work. [Census](candidate/resources.json) · [Native run table](candidate/runs.csv) · [Recorder origin](candidate/origin.json)

The local general store also held shared assets, transport archives, controller work and indexes. Across those components, 135,436 regular-file paths held 6,784,131,894 logical bytes. Counting each inode once gave 5,282,608,650 logical bytes and 5,596,258,304 bytes in allocated file blocks. APFS can share extents between different inodes. The census skipped 1,465 symbolic links, including historical-store links, and excluded its own active recorder directory. The volume had 1,009,774,370,816 bytes free after the scan. [Component table](candidate/components.csv)

The T1 study ledger separately reports 29 physical fits, shared work, failed attempts, GPU memory, transfers and access costs. The prior two research batches retain separate censuses. [T1 resources](../../results/t1/resources.md) · [First batch](../../results/t0/resources/resources.json) · [Second batch](../../results/batch-two/resources/README.md)
