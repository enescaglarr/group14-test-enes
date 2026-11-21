# NIGHT_LOG

## Steps

| Step | gen | holdout | mean (delta) | FinTabNet clean / degraded TEDS | mean s/table | max s/table | Decision | Note |
|---|---|---|---|---|---|---|---|---|
| 0 baseline | 0.9486 | 0.9333 | 0.9409  | 0.383 / 0.201 | 4.42 | 27.8 | REFERENCE | commit 7bcc951; timings: 8-core laptop, sequential, one table at a time; max 27.8 s >= 15 s (ftnd) |
| 1a text_height | 0.9486 | 0.9361 | 0.9424 +0.0015 | 0.366 / 0.442 | 5.02 | 12.3 | ACCEPT | fused-letter blobs + clamp 12-80 px + Tesseract word-height cross-check; FTN-deg 0.201->0.442, worst 27.8->12.3 s |
| 1b router >=40% vline | 0.9486 | 0.9361 | 0.9424 +0.0000 | 0.366 / 0.442 | 3.52 | 8.3 | ACCEPT | scores identical on all 4 sets (1a already removed the false vlines); time change = machine-load noise (mean 5.0 -> 3.5 s with same code paths) |
| 1c bordered->borderless fallback | 0.9486 | 0.9361 | 0.9424 +0.0000 | 0.366 / 0.442 | 3.38 | 8.4 | ACCEPT | never triggered on these 300 tables after 1a/1b (safety net only); scores identical |
| 1d 12 s budget guard | 0.9486 | 0.9361 | 0.9424 +0.0000 | 0.366 / 0.442 | 3.37 | 8.3 | ACCEPT | no table reached the 12 s budget here (max 8.3 s), scores identical; adapter logs s/table + max |
