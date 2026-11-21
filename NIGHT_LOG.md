# NIGHT_LOG

## Steps

| Step | gen | holdout | mean (delta) | FinTabNet clean / degraded TEDS | mean s/table | max s/table | Decision | Note |
|---|---|---|---|---|---|---|---|---|
| 0 baseline | 0.9486 | 0.9333 | 0.9409  | 0.383 / 0.201 | 4.42 | 27.8 | REFERENCE | commit 7bcc951; timings: 8-core laptop, sequential, one table at a time; max 27.8 s >= 15 s (ftnd) |
| 1a text_height | 0.9486 | 0.9361 | 0.9424 +0.0015 | 0.366 / 0.442 | 5.02 | 12.3 | ACCEPT | fused-letter blobs + clamp 12-80 px + Tesseract word-height cross-check; FTN-deg 0.201->0.442, worst 27.8->12.3 s |
