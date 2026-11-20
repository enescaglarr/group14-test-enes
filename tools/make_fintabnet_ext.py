#!/usr/bin/env python3
"""
External robustness set from FinTabNet (docling-project/FinTabNet_OTSL).

Writes two datasets in the data/dev layout, so evaluate.py / score_all.py run
on them unchanged:

  <out>/fintabnet/           clean: the original FinTabNet PNG crops
  <out>/fintabnet_degraded/  same tables, same gt/, degraded image.jpg

  python tools/make_fintabnet_ext.py --parquet test-00000-....parquet \
      --out data/ext --n 50 --seed 15

Ground truth: built from the OTSL tokens (fcel/ecel start a cell, lcel extends
colspan, ucel rowspan, xcel both); cell text = the cell's tokens joined. The
i-th fcel/ecel is the i-th entry of `cells[0]` (= i-th <td> of the HTML).

FinTabNet has no header annotation. `header` in cells.json is a heuristic
(leading rows with no non-year numbers in the value columns); the scorers
ignore it. qa.json holds up to 5 lookup questions, kept only if the scorer's
own auto_answer gets them right on the ground truth itself.

Degradation (seeded per table): upscale x3 (FinTabNet text is ~8 px high, too
small to survive damage otherwise — like printing and rescanning), skew 2-6 deg
either way, mild perspective, Gaussian blur, fax specks, JPEG quality 40.
"""
import argparse
import importlib.util
import io
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev, qa = _load("evaluate"), _load("eval_qa")
YEAR = re.compile(r"^\(?(19|20)\d{2}\)?$")


def otsl_to_cells(otsl, cell_texts):
    """OTSL tokens + per-cell texts -> [{text,r,c,rs,cs}]."""
    grid, cells, r, c, k = {}, [], 0, 0, 0
    for tok in otsl:
        if tok == "nl":
            r, c = r + 1, 0
            continue
        if tok in ("fcel", "ecel"):
            cell = {"text": cell_texts[k], "r": r, "c": c, "rs": 1, "cs": 1}
            k += 1
            cells.append(cell)
            grid[(r, c)] = cell
        elif tok == "lcel":
            grid[(r, c)] = grid[(r, c - 1)]
            if grid[(r, c)]["r"] == r:
                grid[(r, c)]["cs"] = c - grid[(r, c)]["c"] + 1
        elif tok == "ucel":
            grid[(r, c)] = grid[(r - 1, c)]
            grid[(r, c)]["rs"] = r - grid[(r, c)]["r"] + 1
        elif tok == "xcel":
            grid[(r, c)] = grid[(r - 1, c)]
        c += 1
    if k != len(cell_texts):
        raise ValueError(f"{k} cell starts vs {len(cell_texts)} texts")
    return cells


def mark_header(cells):
    n_rows = max(c["r"] + c["rs"] for c in cells)
    n_head = 0
    for r in range(min(3, n_rows - 1)):
        vals = [c["text"] for c in cells if c["r"] == r and c["c"] > 0 and c["text"].strip()]
        if any(qa.nums_in(v) and not YEAR.match(v.strip()) for v in vals):
            break
        n_head = r + 1
    for c in cells:
        c["header"] = c["r"] < n_head
    return n_head


def make_qa(cells, n_head, rng, k=5):
    if n_head == 0:
        return []
    grid = {}
    for c in cells:
        for dr in range(c["rs"]):
            for dc in range(c["cs"]):
                grid[(c["r"] + dr, c["c"] + dc)] = c
    labels = [grid[(r, 0)]["text"] for r in range(n_head, 1 + max(r for r, _ in grid))
              if (r, 0) in grid]
    norm_grid = {pos: qa.norm(c["text"]) for pos, c in grid.items()}
    cand = []
    for c in cells:
        if c["header"] or c["c"] == 0 or not qa.nums_in(c["text"]):
            continue
        lab = grid[(c["r"], 0)]["text"].strip()
        if not lab or labels.count(grid[(c["r"], 0)]["text"]) != 1:
            continue
        heads = [grid[(h, c["c"])]["text"].strip() for h in range(n_head)
                 if (h, c["c"]) in grid and grid[(h, c["c"])]["text"].strip()]
        if not heads:
            continue
        q = {"type": "lookup",
             "question": f'What is the value for "{lab}" in column "{heads[-1]}"?',
             "answer": c["text"], "answer_numeric": qa.nums_in(c["text"])[0]}
        if qa.grade_one(q, qa.auto_answer(q, norm_grid)):   # answerable from gt itself
            cand.append(q)
    rng.shuffle(cand)
    return cand[:k]


def degrade(png_bytes, rng: np.random.Generator):
    img = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    pad = int(0.08 * max(img.shape))
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    h, w = img.shape
    ang = rng.uniform(2, 6) * rng.choice([-1, 1])
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + rng.uniform(-0.03, 0.03, (4, 2)) * [w, h]).astype(np.float32)
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                              flags=cv2.INTER_LINEAR, borderValue=255)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.7, 1.2))
    noise = rng.random(img.shape)
    img[noise < 0.0015] = 0                              # black fax specks
    img[noise > 0.9985] = 255                            # dropouts
    img = np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])
    return buf.tobytes(), round(float(ang), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", default="data/ext")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--rows", default="4-15")
    ap.add_argument("--cols", default="3-8")
    args = ap.parse_args()
    rmin, rmax = map(int, args.rows.split("-"))
    cmin, cmax = map(int, args.cols.split("-"))

    meta = pq.read_table(args.parquet, columns=["rows", "cols"]).to_pylist()
    eligible = [i for i, m in enumerate(meta)
                if rmin <= m["rows"] <= rmax and cmin <= m["cols"] <= cmax]
    picks = sorted(random.Random(args.seed).sample(eligible, args.n))
    full = pq.read_table(args.parquet, columns=["filename", "otsl", "cells", "image"])

    out = Path(args.out)
    clean, deg = out / "fintabnet", out / "fintabnet_degraded"
    man_c, man_d = [], []
    for n, idx in enumerate(picks):
        row = full.slice(idx, 1).to_pylist()[0]
        tid = f"table_{n:03d}"
        # tokens carry inline HTML (<b>, <sup>, ...): keep the text, drop the tags
        texts = [re.sub(r"<[^>]+>", "", "".join(c["tokens"])) for c in row["cells"][0]]
        cells = otsl_to_cells(row["otsl"], texts)
        n_head = mark_header(cells)
        rng = random.Random(args.seed * 1000 + n)
        qs = make_qa(cells, n_head, rng)
        png = row["image"]["bytes"]
        jpg, ang = degrade(png, np.random.default_rng(args.seed * 1000 + n))
        n_rows = max(c["r"] + c["rs"] for c in cells)
        n_cols = max(c["c"] + c["cs"] for c in cells)
        for base, name, data, man, tier in ((clean, "image.png", png, man_c, "clean"),
                                            (deg, "image.jpg", jpg, man_d, "degraded")):
            d = base / tid
            (d / "gt").mkdir(parents=True, exist_ok=True)
            (d / name).write_bytes(data)
            (d / "gt" / "cells.json").write_text(json.dumps(cells, ensure_ascii=False, indent=1))
            (d / "gt" / "table.html").write_text(ev.cells_json_to_html(cells), encoding="utf-8")
            (d / "gt" / "qa.json").write_text(json.dumps(qs, ensure_ascii=False, indent=1))
            w, h = Image.open(io.BytesIO(data)).size
            man.append({"id": tid, "image": f"{tid}/{name}", "style": "fintabnet",
                        "tier": tier, "source": row["filename"], "source_index": idx,
                        "has_spans": any(c["rs"] > 1 or c["cs"] > 1 for c in cells),
                        "n_rows": n_rows, "n_cols": n_cols, "header_rows_heuristic": n_head,
                        "n_qa": len(qs), "size": [w, h],
                        **({"skew_deg": ang} if tier == "degraded" else {})})
    for base, man in ((clean, man_c), (deg, man_d)):
        (base / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print(f"eligible {len(eligible)}/{len(meta)}, wrote {len(picks)} tables to {clean} and {deg}")
    print(f"tables with spans: {sum(m['has_spans'] for m in man_c)}, "
          f"qa questions: {sum(m['n_qa'] for m in man_c)} "
          f"({sum(1 for m in man_c if m['n_qa'] == 0)} tables without qa)")


if __name__ == "__main__":
    main()
