# NIGHT_LOG

## Steps

| Step | gen | holdout | mean (delta) | FinTabNet clean / degraded TEDS | mean s/table | max s/table | Decision | Note |
|---|---|---|---|---|---|---|---|---|
| 0 baseline | 0.9486 | 0.9333 | 0.9409  | 0.383 / 0.201 | 4.42 | 27.8 | REFERENCE | commit 7bcc951; timings: 8-core laptop, sequential, one table at a time; max 27.8 s >= 15 s (ftnd) |
| 1a text_height | 0.9486 | 0.9361 | 0.9424 +0.0015 | 0.366 / 0.442 | 5.02 | 12.3 | ACCEPT | fused-letter blobs + clamp 12-80 px + Tesseract word-height cross-check; FTN-deg 0.201->0.442, worst 27.8->12.3 s |
| 1b router >=40% vline | 0.9486 | 0.9361 | 0.9424 +0.0000 | 0.366 / 0.442 | 3.52 | 8.3 | ACCEPT | scores identical on all 4 sets (1a already removed the false vlines); time change = machine-load noise (mean 5.0 -> 3.5 s with same code paths) |
| 1c bordered->borderless fallback | 0.9486 | 0.9361 | 0.9424 +0.0000 | 0.366 / 0.442 | 3.38 | 8.4 | ACCEPT | never triggered on these 300 tables after 1a/1b (safety net only); scores identical |
| 1d 12 s budget guard | 0.9486 | 0.9361 | 0.9424 +0.0000 | 0.366 / 0.442 | 3.37 | 8.3 | ACCEPT | no table reached the 12 s budget here (max 8.3 s), scores identical; adapter logs s/table + max |
| 2 white margin before OCR | 0.9491 | 0.9385 | 0.9438 +0.0014 | 0.463 / 0.454 | 3.36 | 7.6 | ACCEPT | crop pad >= 0.5x text height, page OCR padded; plus crop_dark_frame no longer trims frameless edges (generic bug that ate edge glyphs); FTN clean 0.366->0.463 |
| 3 rebuild numbers from digits | 0.9560 | 0.9502 | 0.9531 +0.0093 | 0.463 / 0.453 | 3.36 | 7.8 | ACCEPT | clear vote (>=80%) -> off-format / junk number cells rebuilt in the table's format ('repaired_raw', adapter writes it); QA gen 0.870->0.894, holdout 0.834->0.882 |
| 4 snap cutoff 0.8 + umlaut only with German hint | 0.9548 | 0.9512 | 0.9530 -0.0001 | 0.463 / 0.453 | 3.38 | 7.6 | ACCEPT | neutral within the 0.002 rule (gen -0.0012, holdout +0.0010): short OCR-damaged headers (0kt) no longer snap, English labels no longer get umlauts |
| B triage fast path (T0/T1 4 s, T2/T3 12 s) | 0.9548 | 0.9512 | 0.9530 +0.0000 | 0.459 / 0.453 | 2.96 | 7.6 | ACCEPT | depth-4 tree on specks/bg_spread/lap_var/skew, CV acc 0.955, T3->T0 0% (CV, gen, holdout); T0/T1 ~30% faster (gen 3.85->2.62 s), scores unchanged; FTN clean 0.463->0.459 |
| C 3rd reader (PP-OCRv5 latin rec, ONNX) | 0.9583 | 0.9538 | 0.9561 +0.0031 | 0.459 / 0.454 | 3.01 | 8.2 | ACCEPT | only suspicious number cells (conf<70 / flagged / format mismatch); 3 reads -> digits+sign, digit-wise majority, printed in table format; QA gen 0.888->0.904, holdout 0.888->0.898 |

## A: before / after (baseline commit 7bcc951 -> step 4 commit)

| Metric | Before | After | Change |
|---|---|---|---|
| gen FINAL (seed 2026, 100 tables) | 0.9486 | 0.9548 | +0.0062 |
| holdout FINAL (seed 7, 100 tables) | 0.9333 | 0.9512 | +0.0179 |
| gen QA | 0.8500 | 0.8880 | +0.0380 |
| holdout QA | 0.8420 | 0.8880 | +0.0460 |
| FinTabNet clean TEDS (50) | 0.383 | 0.463 | +0.0804 |
| FinTabNet degraded TEDS (50) | 0.201 | 0.453 | +0.2519 |
| mean s/table (all 300) | 4.4 | 3.4 | -1.0 |
| max s/table (all 300) | 27.8 | 7.6 | -20.2 |

Timing: 8-core laptop, one process, one table at a time, Python 3.12, Tesseract 5 (deu+eng, tessdata_fast); other apps running, so +-30% noise between runs.

## C: third reader -- model and license

- Model: PaddleOCR **PP-OCRv5 latin mobile recognition** (`latin_PP-OCRv5_mobile_rec`), recognition only (no detection), ONNX export.
- Files: `models/paddleocr_rec_latin/` (`rec.onnx` 7.9 MB, `dict.txt`, `config.json`, `SOURCE_README.md`), committed to the repo; no network at run time.
- Source: Hugging Face `monkt/paddleocr-onnx`, revision `7b02d0a30a07ba2b92ad1ff5a8941ae2c633de65`, path `languages/latin/`; converted from PaddlePaddle PP-OCRv5 (github.com/PaddlePaddle/PaddleOCR).
- License: **Apache License 2.0** (model card of the source repo and PaddleOCR). Apache-2.0 allows use, modification and redistribution; keep the license notice (SOURCE_README.md is kept next to the weights).
- Runtime: onnxruntime 1.32.0 (CPU), ~50 ms per cell; input height 48 (the config.json says 32, the graph wants 48), CTC decoding over blank + 502 dict chars + space.
