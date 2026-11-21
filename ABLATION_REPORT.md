# Ablation: from the Tesseract floor to tableparser, one part at a time

Branch `ocrbench`. Measurement only: no rule was changed for this report. The switches that turn parts off
(`TABLEPARSER_ABLATE`, `tableparser/ablation.py`) are off by default, and the submitted behaviour is
unchanged. With the switches off, the dev and final outputs are byte-identical to the submitted system.

| version | what runs |
|---|---|
| **V1** | plain Tesseract floor (`baselines/baseline_tesseract.py`) on the original image |
| **V2** | tableparser's preprocessing (dark-frame crop, upscale, deskew, shear, light normalization, despeckle) on the whole page, then exactly V1 |
| **V3** | no preprocessing, but tableparser's structure + cell reading + value repair + assemble; triage and third reader off (`preprocess,triage,third`) |
| **V4** | preprocessing + structure + post-processing; triage and third reader off (`triage,third`) |
| **V5** | the full system as submitted (`main`, eb7dab3 behaviour) |

V6 (PaddleOCR as the *main* cell reader, Tesseract as fallback) was dropped on request before it was
measured, and its code is removed (b585918). PaddleOCR remains the third reader on suspicious numbers, as
in the submission.

## Results

TableSalt sets report the official `score_all.py` FINAL. FinTabNet and OCRBench report mean TEDS, with
TEDS-S in parentheses. OCRBench-100 is a fixed subset of 100 samples (seed 100) of the 400 English
table-parsing samples, scored with the official `eval.py`. Seconds come from the single-process timing
sample described below.

| version | dev FINAL | final FINAL | FinTabNet clean | FinTabNet degraded | OCRBench-100 | OCRBench-400 | s / table (mean / max) |
|---|---|---|---|---|---|---|---|
| V1 Tesseract | 0.342 | 0.301 | 0.304 (0.579) | 0.120 (0.308) | 0.305 (0.528) | 0.311 | 0.20 / 0.5 |
| V2 + preprocessing | 0.358 | 0.313 | **0.503** (0.597) | 0.317 (0.455) | 0.305 (0.387) | 0.299 | 0.63 / 1.2 |
| V3 structure, no preprocessing | 0.628 | 0.685 | 0.447 (0.583) | 0.193 (0.401) | 0.345 (0.506) | 0.337 | 3.46 / 9.0 |
| V4 V3 + preprocessing | 0.966 | 0.955 | 0.463 (0.599) | 0.453 (0.633) | 0.343 (0.505) | — | 3.28 / 8.7 |
| **V5** full (submitted) | **0.970** | **0.957** | 0.459 (0.591) | 0.454 (0.633) | 0.335 (0.497) | 0.333 | 3.16 / 8.8 |

### TableSalt, tier by tier (TEDS)

| version | dev T0 | T1 | T2 | T3 | dev QA | final T0 | T1 | T2 | T3 | final QA |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 | 0.539 | 0.476 | 0.256 | 0.323 | 0.267 | 0.493 | 0.455 | 0.265 | 0.229 | 0.242 |
| V2 | 0.473 | 0.330 | 0.359 | 0.367 | 0.300 | 0.420 | 0.362 | 0.336 | 0.307 | 0.174 |
| V3 | 0.978 | 1.000 | 0.329 | 0.616 | 0.542 | 0.984 | 0.992 | 0.527 | 0.617 | 0.590 |
| V4 | 0.999 | 0.995 | 0.980 | 0.973 | 0.875 | 0.989 | 0.991 | 0.962 | 0.954 | 0.886 |
| V5 | 0.999 | 0.995 | 0.981 | 0.978 | 0.892 | 0.989 | 0.985 | 0.962 | 0.957 | 0.900 |

## What each part contributes

- **Structure + cell reading (V1 → V3)** is the largest single gain on TableSalt: final FINAL goes from
  0.30 to 0.69. Clean tiers are almost solved by structure alone (T0/T1 0.98–1.00). Without
  preprocessing, photos and faxes are not: T2 is 0.33–0.53 and T3 0.62.
- **Preprocessing** helps a lot once structure exists (V3 → V4): final 0.685 → 0.955, T2 0.53 → 0.96,
  T3 0.62 → 0.95. On FinTabNet degraded it more than doubles TEDS (0.193 → 0.453). Applied to the
  Tesseract floor alone (V1 → V2), it barely moves TableSalt (+0.01). The floor's problem there is grid
  structure, not image quality. On small, tight FinTabNet crops, though, V2 still gains +0.20 TEDS
  through upscaling and the white margin.
- **Triage + third reader (V4 → V5)** add +0.004 on dev and +0.002 on final, mostly QA (number digits:
  0.875 → 0.892, 0.886 → 0.900) and T3. They are not free elsewhere: on FinTabNet clean
  (0.463 → 0.459) and OCRBench-100 (0.343 → 0.335), V5 is slightly *below* V4. The fast path skips reads
  that those real-document tables still need. On the 30-table timing sample, mean time drops only
  3.28 → 3.16 s.
- **OCRBench** stays near the floor for every version (0.31–0.34). Its main losses are not about OCR.
  They are the generator-specific assumptions described in `OCRBENCH_REPORT.md`: number formats,
  tables without numbers, and `$` columns.

## Protocol

- **Scores.** Four runner processes in parallel (`tools/ablation_run.py --shard I/4`) with
  `TABLEPARSER_TIME_SCALE=4` and `--lean-threads`. The scale stretches every time budget, so four
  processes sharing 8 cores do the same work per table as one process. Two checks backed this up:
  - Without the scale, dev FINAL drops 0.9661 → 0.9583 (budget-cut reads).
  - With the scale, all 24 dev outputs are byte-identical to the sequential run.
- **Threads.**
  - `OMP_THREAD_LIMIT=1`, as in the container. Without it, load average reached 130–145. The dev outputs
    were identical with and without it.
  - `--lean-threads` (OMP_NUM_THREADS=1, `cv2.setNumThreads(1)`, onnxruntime one intra-op thread) left
    outputs byte-identical on all 24 dev tables. It was only about 5 % faster (75 s vs 77 s for dev) and
    lowered the 1-min load from 56 to 46. The bottleneck is one Tesseract process per cell read, not
    library threads.
- **Seconds.** A fixed sample of 30 tables (seed 30): 10 TableSalt final, 5 FinTabNet clean,
  5 FinTabNet degraded, 10 OCRBench. Each was run in one process with no flags, on an otherwise idle
  machine. OCRBench Rotowire_test_table_44 and _154 were timed separately (see `OCRBENCH_REPORT.md`).
- **Budget check.** Every table's meta (`budget_skipped`, `cells_unread`) was recorded. Any table where
  the budget cut work was measured again with 3 processes (scale 3). Of the 252 affected tables, 234
  were clean on re-measurement. The 18 left over are real, not parallel artefacts:
  - 16 are large OCRBench tables in runs from before the large-table fix (13 in base, 1 in fix1, 2 in
    fix2).
  - 2 TableSalt tables (holdout table_052 in fix2, gen table_032 in fix5) also exceed the 4 s fast-path
    budget in a single process.
- **Exceptions.**
  - **V3** is the earlier sequential run (one process, before `OMP_THREAD_LIMIT=1`), without budget meta.
    By request it was not re-measured.
  - **V1 and V2** have no time budget (Tesseract only).
  - **OCRBench-400** for V1–V3 and V5 comes from full runs. V4 was measured on the 100-sample subset only.

## Reproduce

```
python tools/ablation_run.py V1..V5 <flat images | ocrbench:<root>> <out> [--shard I/4] [--lean-threads]
TABLEPARSER_TIME_SCALE=4 ...                       # for parallel score runs only
python tools/ablation_score.py <dir with V1..V5/<set>> out.json --ocb-res <official eval outputs>
```
