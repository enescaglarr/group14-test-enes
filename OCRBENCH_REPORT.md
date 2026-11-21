# tableparser on OCRBench v2 — English table parsing

Measurement only: no tableparser rule was changed for this report. Branch `ocrbench`.

**Result.** On the 400 English table-parsing samples of OCRBench v2, scored with the benchmark's own code:

- tableparser: TEDS **0.333**, TEDS-S 0.479.
- Tesseract floor: TEDS 0.311, TEDS-S 0.536.
- InternVL2.5-26B, from its official prediction file: TEDS 0.611, TEDS-S 0.787.

The single biggest loss is that tableparser returns an **empty table on 87 of 400 samples (22 %)**. Its body-row detector only accepts TableSalt's number formats and rejects one-decimal values like `500.0` or `22.2`. It also needs at least one number per table, so text-only tables come back empty too. Both are generator-specific assumptions that do not hold on real documents.

**After the fixes in section 6** (branch `ocrbench`, TableSalt scores unchanged): TEDS 0.333 → **0.502**,
TEDS-S 0.479 → 0.636, empty tables 87 → 2.

## 1. The data

`~/Downloads/OCRBench_v2/OCRBench_v2.json` holds 10,000 samples with the keys `dataset_name`, `type`, `id`, `image_path`, `question` and `answers`. They cover 30 task types:

| type | n | type | n |
|---|---|---|---|
| APP agent en | 300 | math QA en | 300 |
| ASCII art classification en | 200 | reasoning VQA en | 600 |
| chart parsing en | 400 | science QA en | 300 |
| cognition VQA en | 800 | **table parsing en** | **400** |
| diagram QA en | 300 | text counting en | 200 |
| document classification en | 200 | text grounding en | 200 |
| document parsing en | 400 | text recognition en | 800 |
| fine-grained text recognition en | 200 | text spotting en | 200 |
| formula recognition en | 400 | VQA with position en | 300 |
| full-page OCR en | 200 | key information extraction en | 400 |
| key information mapping en | 300 | | |
| cognition VQA cn | 200 | handwritten answer extraction cn | 200 |
| document parsing cn | 300 | key information extraction cn | 400 |
| formula recognition cn | 200 | reasoning VQA cn | 400 |
| full-page OCR cn | 200 | **table parsing cn** | **300** (skipped: no Chinese Tesseract data) |
| text translation cn | 400 | | |

**`table parsing en`: 400 samples, all images present.**

| source | n | images | question | ground truth |
|---|---|---|---|---|
| `table_to_html` | 316 | crops of tables from annual reports (e.g. `PNR_2016_page_43_table_0.jpg`); median 548×142 px | always "Please represent this table with the HTML-format in text." | full HTML page; header rows are `<thead><th>…</th></thead>` (no `<tr>`), body `<tr><td>`, `rowspan` / `colspan` in 219 of 316 |
| `MMTab` | 84 | rendered tables from TABMWP 15, WikiBIO 15, InfoTabs 13, TAT-QA 11, FeTaQA 9, TabFact 9, ToTTo 4, HiTab 4, Rotowire 4; median 1432×423 px | 24 different phrasings: 65 ask for HTML, 19 for Markdown | 65 HTML (`<table border="1">`, `<th>`/`<td>`, spans in 41), 19 flat Markdown tables (`\| a \| b \|`, no spans) |

Example question and answer (`table_to_html`, id 7801):

```
Q: Please represent this table with the HTML-format in text.
A: <html> … <table frame="hsides" rules="groups" width="60%"><thead><th></th><th rowspan="2">Actual 2008</th>
   <th colspan="5">Projected</th></thead><thead><th></th><th>2009</th> … </thead><tr><td …
```

## 2. Official scoring

The scoring code is `Yuliang-Liu/MultimodalOCR`, folder `OCRBench_v2/eval_scripts`, commit `5770b5c93855027207b3d6511f11d108853a18c3`. Its `eval.py` runs **unchanged**. The predictions go in a `predict` field of the sample list, as in the repo README.

**How the `table parsing en` branch scores a sample:**
- A question containing "html" goes to the HTML branch. That branch takes everything from `<body` (or `<table`) onwards, wraps it with `wrap_html_table`, and computes TEDS. If there is no `<table`, the score is 0.
- A question containing "markdown" goes to the Markdown branch. Both prediction and answer pass through `convert_markdown_table_to_html` before TEDS.
- In this TEDS only `<td>` is a cell with content. A `<th>` counts as a node, but its text is ignored, and `th` against `td` costs a full rename.
- TEDS-S is not reported by `eval.py`. It comes from the same `TEDS` class with `structure_only=True` and the identical preprocessing (`tools/ocrbench_score.py teds_s`).

**Sanity check.** With the ground truth itself as the prediction, all 400 samples score 1.000, mean **1.000**.

## 3. Runs

- **tableparser** (`tools/ocrbench_run.py`): `parse_table` followed by the submission adapter's `to_cells`, exactly as in the container. The cell list then becomes:
  - for the 381 HTML questions, an HTML `<table>` with every cell as `<td>` and the spans kept;
  - for the 19 Markdown questions, a Markdown table (no spans in Markdown, so a spanning cell's text sits at its top-left position).
- **Tesseract floor**: `baselines/baseline_tesseract.py` `to_html`, converted the same way.
- **Environment**: sequential runs, one process, Python 3.12, Tesseract 5 with deu+eng (tessdata_fast), 60 s hard timeout per sample.

| | mean s | median s | p95 s | max s | > 15 s | errors | timeouts |
|---|---|---|---|---|---|---|---|
| tableparser | 2.17 | 1.43 | 4.66 | 18.4 | 2 | 0 | 0 |
| Tesseract floor | 0.24 | 0.17 | 0.58 | 3.1 | 0 | 0 | 0 |

The two slow tableparser samples are large box-score tables: `EN_part/MMTab/Rotowire_test_table_44.jpg` at 18.4 s and `…_154.jpg` at 15.3 s. The 12 s budget guard bounds the optional re-reads only. The mandatory reading of a very large grid can still go past it.

## 4. Results

| system | TEDS | TEDS-S | table_to_html (316) TEDS | MMTab HTML (65) TEDS | MMTab Markdown (19) TEDS | s / sample |
|---|---|---|---|---|---|---|
| **tableparser** (cells as `<td>`) | **0.333** | **0.479** | 0.337 | 0.283 | 0.434 | 2.17 |
| tableparser, header rows as `<th>` (sensitivity check) | 0.372 | 0.486 | 0.395 | 0.242 | 0.434 | — |
| Tesseract floor | 0.311 | 0.536 | 0.316 | 0.289 | 0.305 | 0.24 |
| InternVL2.5-26B (official prediction file, scored here) | 0.611 | 0.787 | 0.534 | 0.891 | 0.925 | GPU |

**The `<td>` row is the result.**
- Writing header rows as `<th>` gains +0.04 TEDS without reading anything better. The `table_to_html` ground truth marks headers with `<th>`, and this TEDS ignores `<th>` text, so a `th` that matches a `th` costs nothing.
- On MMTab the `<th>` variant loses (0.242 against 0.283), because MMTab uses `<th>` differently (row headers, titles).
- This is a scorer convention, not table recognition, so the main number stays with plain `<td>`.

### Published scores for this task

| source | what is reported | numbers |
|---|---|---|
| **InternVL2.5-26B predictions** (`OCRBench_v2/pred_folder/internvl2_5_26b.json` in the official repo, generated by the authors with VLMEvalKit) | per-task score for these 400 samples, official code | TEDS **0.611**, TEDS-S 0.787 (computed here) — the only directly comparable published model output |
| OCRBench v2 paper (arXiv 2501.00321v2, Table 2; v1 Table 2) | **no per-task table-parsing score.** Only the "Parsing" category: the mean of table parsing, chart parsing, document parsing and formula recognition, 0–100 | e.g. Ovis2-8B 44.8, Gemini-Pro 39.2, InternVL3-14B 38.4, Claude3.5-sonnet 37.8, Step-1V 37.2, GPT-4o 36.3, InternVL2.5-26B 36.2 (v1), Qwen2.5-VL-7B 30.4, Molmo-7B 7.6 |
| OCRBench v2 leaderboard (project page and HF space `ling99/OCRBench-v2-leaderboard`, `OCRBench_en.csv`) | the same category-level "Parsing" / "Element Parsing" column, no per-task split | — |
| Xiao et al., "From Diagnosis to Correction: Benchmarking and Improving Real-World Table Parsing" (arXiv 2608.09842), Table 1 | an "OCRBenchv2" column on their *Consensus-Hard* set: samples kept only where ≥ 2 parsers score TEDS < 0.90. Not the full English subset, and the language mix and HTML/Markdown split are not stated. **Not comparable** | TEDS/TEDS-S: MinerU2.5-Pro 84.49/90.25, GLM-OCR 79.52/86.57, PaddleOCR-VL-1.6 75.72/82.49, Qwen3.5-397B 73.22/78.63, Qwen3.5-35B 70.20/76.25 |

The paper's "Parsing" numbers average four tasks, so they cannot go in the same column as a table-only TEDS. They are listed for scale only. InternVL2.5-26B scores 36.2 there, against 61.1 on table parsing alone, which suggests table parsing is the easiest of the four parsing tasks for these models.

## 5. Error types (tableparser)

399 of 400 samples score TEDS < 0.9. Each gets one primary type, checked in this order (`tools/ocrbench_analyze.py`); "any" counts every sample the type applies to.

| type | primary | mean TEDS | any | table_to_html / MMTab (primary) | generator-specific? |
|---|---|---|---|---|---|
| non-Latin script | 0 | — | 0 | 0 / 0 | — (the English subset has none) |
| table not found / text around it | 98 | 0.079 | 98 | 69 / 29 | **yes**, see below |
| multi-line cell (a wrapped cell split into rows) | 57 | 0.461 | 62 | 55 / 2 | **yes**: TableSalt cells are one line |
| columns wrong | 113 | 0.302 | 222 | 81 / 32 | general: column splitting by gaps; `$` signs in their own column |
| header depth | 35 | 0.414 | 136 | 35 / 0 | partly: TableSalt headers are 1–2 rows, these have up to 3 and empty header cells |
| merged cells | 25 | 0.587 | 31 | 20 / 5 | general |
| other (structure right, text / tags) | 71 | 0.492 | 71 | 56 / 15 | mostly the `th`/`td` convention above, plus text |

**Why tables come back empty (87 samples with a 1×1 prediction):**

| cause | n |
|---|---|
| numbers in a format the value parser does not accept: one decimal (`500.0`, `2,096.2`, `22.2`), so no row counts as a body row | 45 |
| no numbers at all (ratings, Wikipedia infoboxes, text tables; year-only columns count as headers) | 23 |
| numbers that parse, still no table | 19 |

Overall, only 70 % of the ground truth's number-like cells parse as numbers for tableparser (6,533 of 9,353). The decimal check (`values.parse_cell`) and the body-row rule (`assemble.is_real_number`, "a row holds a real number") both encode TableSalt's grammar: two decimals or integers. They are the main reason tableparser falls to roughly the Tesseract floor here.

### The 5 worst samples

| id | TEDS | source | ground truth → prediction | why | image |
|---|---|---|---|---|---|
| 7814 | 0.006 | MMTab / TabFact | 30×5 → 1×1 | text table whose only numbers are years, which count as headers → no body row | `EN_part/MMTab/TabFact_2-1805191-33.html.csv.jpg` |
| 7866 | 0.007 | table_to_html | 41×3 → 1×1 | all values one-decimal (`$ 126.3`, `1,167.7`) → unparsed → no body row | `EN_part/table_parsing_to_html/PNR_2015_page_74_table_0.jpg` |
| 7844 | 0.007 | table_to_html | 40×3 → 1×1 | same (`$ 238.5`, `764.0`) | `EN_part/table_parsing_to_html/PNR_2016_page_55_table_0.jpg` |
| 7996 | 0.007 | table_to_html | 38×3 → 1×1 | same (`$ 433.0`, `249.2`) | `EN_part/table_parsing_to_html/PNR_2015_page_91_table_0.jpg` |
| 8065 | 0.010 | table_to_html | 44×4 → 1×1 | credit-ratings table (`Baa2`, `BBB+`, `STABLE`), no numbers at all | `EN_part/table_parsing_to_html/PPL_2008_page_58_table_0.jpg` |

All paths are relative to the OCRBench v2 root. Sample 8181 (`PNR_2016_page_43_table_0.jpg`) is a clean 8×8 contractual-obligations table and fails the same way. The page OCR finds its words (95; the 40 checked by hand are all right, `Debt obligations 0.8 $ 500.0 $ 2,096.2 …`), but every value has one decimal.

## 6. Fixing the general errors (fix1–fix5)

Sections 1–5 describe the submitted system. This section covers five rule changes made after them,
one commit each, now on branch `ocrbench` (`main` is untouched).

**Acceptance rule:** a step is kept only if none of dev, gen, holdout or final drops by more than 0.002
FINAL against the previous kept step. All five were kept.

Protocol (details in `ABLATION_REPORT.md`):
- **Scores:** 4 parallel processes with the time budget scaled by 4. Outputs are byte-identical to a
  single-process run.
- **Budget re-runs:** tables where the time budget cut work were measured again with 3 processes.
- **Seconds:** one process on a fixed 30-table sample, plus the two Rotowire tables timed separately.
- **OCRBench:** the fixed 100-sample subset (seed 100) for every step. The full 400 for the base and
  for the last kept step.

| step | commit | change | dev / gen / holdout / final FINAL | OCRBench-100 TEDS (TEDS-S) | empty tables (of 100) | s / table mean / max | Rotowire_44 / _154 s | kept |
|---|---|---|---|---|---|---|---|---|
| base | eb7dab3 | submitted system | 0.9704 / 0.9583 / 0.9539 / 0.9573 | 0.335 (0.497) | 19 | 3.16 / 8.8 | 18.2 / 14.5 | — |
| fix1 | 2fa34bf | numbers with 1 or 2 decimals; decimal vote over 0/1/2 | same | 0.405 (0.564) | 12 | 3.22 / 7.5 | 18.4 / 15.7 | yes |
| fix2 | 7ff7856 | tables without numbers: header from the top row(s) above the first rule, never an empty table | same | 0.433 (0.607) | 1 | 3.38 / 9.3 | 18.4 / 14.5 | yes |
| fix3 | 177aab2 | > 150 cells: first read only; 12 s hard limit | same | 0.433 (0.607) | 1 | 3.11 / 7.1 | **12.3** / **13.0** | yes |
| fix4 | d61ff7b | a lone `$ € £` before a number joins it as its unit | same | **0.478** (0.622) | 1 | 3.38 / 8.0 | 12.4 / 13.3 | yes |
| fix5 | 9d63eb0 | repairs map OCR letter look-alikes to digits first (`1O7.00` → 107.00), incl. `; : · '` | same | 0.478 (0.622) | 1 | 3.36 / 7.9 | 12.6 / 14.0 | yes |

"Same" means identical FINAL on all four sets. One exception: final TEDS-weighted is 0.9674 for fix5
against 0.9675 for the base (cell F1 0.9046 against 0.9050), and FINAL is 0.9573 for both. The
timing-sample differences between steps (3.1–3.4 s mean) are within run-to-run noise. Only the
Rotowire column moves.

### OCRBench, full 400: before and after

| | base | fix5 | InternVL2.5-26B |
|---|---|---|---|
| TEDS | 0.333 | **0.502** | 0.611 |
| TEDS-S | 0.479 | **0.636** | 0.787 |
| empty tables (1×1) | 87 | **2** | — |
| table_to_html (316) TEDS | 0.337 | 0.529 | 0.534 |
| MMTab HTML (65) TEDS | 0.283 | 0.380 | 0.891 |
| MMTab Markdown (19) TEDS | 0.433 | 0.470 | 0.925 |

- **Per sample:** 273 of 400 improve, 12 get worse, and 115 are unchanged. The largest drop is ToTTo
  sample 8115 (−0.39). It goes from 2×3 to 2×4 against a 2×5 truth: closer in column count, but the
  cells shift and their text no longer lines up.
- **Versus InternVL:** on the annual-report crops (`table_to_html`), tableparser now matches
  InternVL2.5-26B (0.529 vs 0.534). The MMTab gap stays large: those are rendered web and text tables,
  with layouts TableSalt never has.

**Error types (primary type, TEDS < 0.9, same rules as section 5):**

| type | base | fix5 |
|---|---|---|
| table not found / text around it | 98 | **10** |
| multi-line cell | 57 | 95 |
| columns wrong | 113 | 81 |
| header depth | 35 | 34 |
| merged cells | 25 | 49 |
| other (structure right) | 71 | 128 |

The not-found errors are almost gone. Multi-line cells and merged cells now come first: tables that used
to be empty are found, and their next problem shows. These two are the obvious next steps. Neither was
in scope here.

**Worst 5 after the fixes:**

| id | TEDS | why | image |
|---|---|---|---|
| 7829 | 0.024 | InfoTabs infobox (14×2) still comes back empty, unchanged from the base; cause not investigated | `EN_part/table_parsing_en_MMTab/InfoTabs_T1179.jpg` |
| 8182 | 0.048 | TabFact 44×6 → 151×3: wrapped lines become rows, columns merge | `EN_part/MMTab/TabFact_2-17916431-2.html.csv.jpg` |
| 7919 | 0.066 | WikiBIO infobox 12×2 → 2×2, unchanged from the base: only a colour-code row (`bscolour`, `6495ed`) is kept | `EN_part/table_parsing_en_MMTab/WikiBIO_test_29612196.jpg` |
| 8084 | 0.075 | InfoTabs infobox 11×2: empty in the base, now 2×2, most rows still missing | `EN_part/table_parsing_en_MMTab/InfoTabs_T2103.jpg` |
| 7865 | 0.077 | annual-report table 11×4: empty in the base, now 3×1 | `EN_part/table_parsing_to_html/PPL_2018_page_148_table_0.jpg` |

### Time (fix3)

Rotowire_test_table_44 is a 30×17 grid with 510 cells. It took 18.2 s in the submitted system and now
ends at 12.3–12.6 s. The 12 s hard limit leaves 120–157 cells unread; the number depends on the run.
Rotowire_test_table_154 drops from 14.5 to 13.0–14.0 s with all cells read. Before the fix, the first
table went past the official 15 s limit and would have been lost entirely. Now it returns in time,
partly read.

Both tables are still slow because the reader starts one Tesseract process per cell. Reading a large
grid with one page-level OCR pass and assigning the words to the cells would remove the unread cells.
That was not in scope.

On TableSalt, no table reaches 150 cells, so fix3 never triggers there.

**Follow-up (1e3fe22): the hard limit is for large tables only.** In the first version of fix3 the 12 s
limit applied to every table. The amd64 container test under emulation on an Apple Silicon Mac makes
every Tesseract call about twice as slow, and there 6 dev tables hit the limit with cells unread:
container FINAL 0.918, against 0.967 for v1. The official limit is a total (5 min + 15 s per table), so
normal tables are now always read to the end; only tables with more than 150 cells keep the 12 s limit.

| | FINAL | dev / gen / holdout / final |
|---|---|---|
| container, amd64 emulated: v1 / v2 | 0.9666 / **0.9686** | — |
| container, native arm64: v1 / v2 | 0.9723 / 0.9723 (all 24 outputs byte-identical) | — |
| local runs (protocol above), against fix5 | — | 0.9704 / 0.9583 / 0.9535 / 0.9573 |

Against fix5, only holdout table_052 differs. It exceeds the 4 s fast-path budget in a single process
too, so its optional reads vary from run to run. The resulting −0.0004 on holdout is within the 0.002
rule. Rotowire_44 / _154 still end at 12.4 / 12.2 s. On the 30-table timing sample: mean 3.49 s,
max 8.0 s.

### fix5: what it changed

On gen + final, fix4 → fix5 changed **0 cells**, and the `; : · '` extension (fix5 against its narrow
variant) also changed **0 cells**. Dev, holdout and OCRBench-100 show no change either. The repair
path never met a number with a letter look-alike in these runs: only 5 of the 17,981 output cells of
fix4 contain one, and none of them goes through the repair path. fix5 is kept because it costs nothing.
It fixes the case shown in the unit checks (`1O7.00`, `1O7;00` → 107.00, with `Sulfur`, `see note 3`,
`n/a` and `Ablation 3` unchanged), but these sets do not show a measurable effect.

### Budget-limited tables after re-measurement

After the 3-process re-runs, the time budget still cuts work in 2 TableSalt tables (holdout table_052 in
fix2, gen table_032 in fix5). Both also exceed the 4 s fast-path budget in a single process, and only
the optional second reads are skipped. On OCRBench-400 with fix5, 6 large tables remain affected,
including both Rotowire tables.

## Reproduce

```
python tools/ocrbench_run.py tableparser <OCRBench_v2> runs/tp        # also: tesseract
python tools/ocrbench_score.py build <OCRBench_v2> runs/tp tableparser td pred_tp.json
cd <MultimodalOCR>/OCRBench_v2/eval_scripts && python eval.py --input_path pred_tp.json --output_path res_tp.json
python tools/ocrbench_score.py teds_s <eval_scripts> pred_tp.json teds_s_tp.json
python tools/ocrbench_analyze.py res_tp.json runs/tp analysis_tp.json
```

Evaluation dependencies are those in `OCRBench_v2/requirements.txt`, plus `ipdb`, which `vqa_metric.py` imports.
