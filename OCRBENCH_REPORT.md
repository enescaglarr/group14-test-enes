# tableparser on OCRBench v2 — English table parsing

Measurement only: no tableparser rule was changed for this report. Branch `ocrbench`.

**Result.** On the 400 English table-parsing samples of OCRBench v2, scored with the benchmark's own code:

- tableparser: TEDS **0.333**, TEDS-S 0.479.
- Tesseract floor: TEDS 0.311, TEDS-S 0.536.
- InternVL2.5-26B, from its official prediction file: TEDS 0.611, TEDS-S 0.787.

The single biggest loss is that tableparser returns an **empty table on 87 of 400 samples (22 %)**. Its body-row detector only accepts TableSalt's number formats and rejects one-decimal values like `500.0` or `22.2`. It also needs at least one number per table, so text-only tables come back empty too. Both are generator-specific assumptions that do not hold on real documents.

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

## Reproduce

```
python tools/ocrbench_run.py tableparser <OCRBench_v2> runs/tp        # also: tesseract
python tools/ocrbench_score.py build <OCRBench_v2> runs/tp tableparser td pred_tp.json
cd <MultimodalOCR>/OCRBench_v2/eval_scripts && python eval.py --input_path pred_tp.json --output_path res_tp.json
python tools/ocrbench_score.py teds_s <eval_scripts> pred_tp.json teds_s_tp.json
python tools/ocrbench_analyze.py res_tp.json runs/tp analysis_tp.json
```

Evaluation dependencies are those in `OCRBench_v2/requirements.txt`, plus `ipdb`, which `vqa_metric.py` imports.
