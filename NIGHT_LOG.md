# NIGHT_LOG

## Summary (10 lines)

1. Branch `night`: baseline commit 7bcc951, one commit per accepted step (1a, 1b, 1c, 1d, 2, 3, 4, B, C), the Docker dry run and this log; all 9 steps passed the rule (mean gen/holdout drop <= 0.002, max < 15 s), none reverted; nothing pushed.
2. Final check, `data/final` (seed 123, 100 new tables): FINAL **0.9262 -> 0.9573** (+0.031), TEDS-w 0.947 -> 0.968, QA 0.810 -> 0.900, CellF1 0.844 -> 0.905.
3. Per tier TEDS (final set): T0 0.972 -> 0.989, T1 0.986 -> 0.985, T2 0.964 -> 0.962, T3 0.908 -> 0.957.
4. gen 0.9486 -> 0.9583, holdout 0.9333 -> 0.9538; FinTabNet TEDS clean 0.383 -> 0.459, degraded 0.201 -> 0.454 (no FinTabNet-specific tuning).
5. Longest table: 27.8 s -> 8.2 s (all 300 eval tables) and 7.4 s on the final set; Docker (dev, offline): mean 3.7 s, max 6.5 s, image 732 MB (arm64) / 836 MB (amd64).
6. Biggest wins: 3 number rebuild from digits (+0.0093), C third reader with digit voting (+0.0031), 1a text height (FinTabNet degraded x2.2, worst time 27.8 -> 12.3 s), 2 white margin + frame-crop fix (FTN clean +0.10).
7. Neutral (kept as safety nets / speed): 1b router 40 %, 1c fallback, 1d 12 s budget (never triggered on these sets), 4 snap 0.8 (-0.0001), B triage (T0/T1 30 % faster, T3 -> T0 = 0 %).
8. Remaining error 1 -- header structure in two-row headers: group-header spans wrong (31 header cells) and an extra header/body row (6 tables); 20 final tables still have a wrong grid (4 of them an extra column).
9. Remaining error 2 -- merged body cells: '—' / 'n/a' / 'see note 3' merged vs single decided wrong (19 body cells).
10. Remaining error 3 -- wrong digits that survive the voting (e.g. 1/4, 5/9, 6/8; 23 cells in 17 tables); value errors overall fell 229 -> 51 cells.

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

## D: Docker dry run (not a submission)

`submission_template/Dockerfile.tableparser` (the official template `Dockerfile` is unchanged): `python:3.12-slim`, Debian `tesseract-ocr` + `deu` + `eng`, pinned pip packages (numpy 2.5.5, opencv-python-headless 5.1.0.95, pytesseract 0.3.13, onnxruntime 1.32.0, pillow 13.0.0), code + all weights (`models/`) copied into the image, entrypoint = `submission_template/tableparser_adapter.py /data/images /data/predictions`.

| Run (data/dev, 24 tables, `--network none`) | Total | mean s/table | max s/table | dev FINAL | Image |
|---|---|---|---|---|---|
| arm64, first try | 189 s | 7.8 | **15.4** (table_003) | 0.9685 | 732 MB |
| arm64, `OMP_THREAD_LIMIT=1` (now in the Dockerfile) | 89 s | 3.7 | 6.5 (table_007) | **0.9723** | 732 MB |
| linux/amd64 (QEMU emulation, 2-table smoke test) | -- | 9.4 | 12.3 | output identical to arm64 | 836 MB |

- Build: 55 s (arm64), 79 s (amd64). Docker Desktop: 8 CPUs, 4 GB RAM.
- Problem found: first run exceeded 15 s on one table. Cause: Debian's Tesseract is built with OpenMP and the pipeline runs up to 8 OCR calls in parallel -> CPU oversubscription. Fix tried: `OMP_THREAD_LIMIT=1` -> mean 7.8 -> 3.7 s, max 15.4 -> 6.5 s, and no table hits the time budget any more (score 0.9685 -> 0.9723).
- amd64 timings come from emulation and say nothing about the scoring machine (16 CPUs, native amd64); they only show that the image builds and runs on that platform.

## E: final check on unseen data (`data/final`, seed 123, 100 tables, never used before)

| | FINAL | TEDS-w | QA | CellF1 | T0 TEDS | T1 TEDS | T2 TEDS | T3 TEDS | mean s | max s |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (7bcc951) | 0.9262 | 0.9467 | 0.810 | 0.8436 | 0.972 | 0.986 | 0.964 | 0.908 | 3.72 | 6.5 |
| night (049f0e1) | **0.9573** | 0.9675 | 0.900 | 0.9050 | 0.989 | 0.985 | 0.962 | 0.957 | 3.57 | 7.4 |

CellF1 per tier (baseline -> night): T0 0.847 -> 0.964, T1 0.944 -> 0.954, T2 0.855 -> 0.840, T3 0.729 -> 0.862.

Error cells (final set, aligned to ground truth; cells in structure-wrong tables counted as structure):

| Type | baseline | night |
|---|---|---|
| structure (shape / span / shifted / extra) | 897 cells, 20 tables | 577 cells, 20 tables |
| value: separator / punctuation | 99 | 2 |
| value: format only | 62 | 6 |
| value: wrong digits | 38 | 23 |
| value: sign | 16 | 4 |
| label | 14 | 11 |
| header | 5 | 9 |

Structure-wrong final tables (night): 20; 4 with an extra column, 6 with an extra row, 1 with a missing row; span errors: 31 header cells (group-header spans), 19 body cells (merged markers).

Worst final tables (night): table_017 (T1 borderless, 7x4 -> 8x5), table_055 (T3 hlines, group-header spans), table_079 (T3 hlines, header depth), table_080 (T0 borderless, group-header spans), table_014 (T2 borderless, extra column), table_066 (T2 grid, merged '—' split).

Notes: seeds used tonight -- 42 dev (untouched), 2026 gen, 7 holdout, 99 triage training, 15 FinTabNet selection, 123 final. Data under `data/` is gitignored (not committed).

## Official submission path

`Dockerfile.tableparser` was folded into the official `submission_template/Dockerfile` and removed; `predict.py` (the template entrypoint, contract unchanged) now calls `tableparser_adapter.main`. Tested exactly as README §7, built for the scoring platform:

    python flatten_dataset.py data/dev flat
    docker build --platform linux/amd64 -t my-tablesalt -f submission_template/Dockerfile .
    docker run --rm --network none -v "$PWD/flat:/data/images:ro" -v "$PWD/out:/data/predictions" my-tablesalt
    python score_all.py --gt data/dev --pred out

| | value |
|---|---|
| dev FINAL (score_all) | 0.9724 (TEDSw 0.9852, QA 0.900, CellF1 0.9419) |
| clean build (no cache) | 76 s |
| run, 24 tables | 177 s wall, mean 7.3 s, max 12.7 s (table_006) -- amd64 emulated on Apple Silicon |
| image | 221 MB compressed, ~594 MB on disk (du in container) |
| format | 24/24 files, keys exactly {text, r, c, rs, cs}, 28 empty cells written, no overlaps or holes |

The amd64 timings are under emulation on an arm64 Mac; the native arm64 run earlier was mean 3.7 s, max 6.5 s. The scoring laptop runs amd64 natively.
