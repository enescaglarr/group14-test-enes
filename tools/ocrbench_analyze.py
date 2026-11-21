#!/usr/bin/env python3
"""Error types of tableparser on OCRBench v2 "table parsing en" (diagnosis only).

    python tools/ocrbench_analyze.py <res.json> <run_dir> <out.json>

<res.json>  official eval.py output for tableparser (per-sample "score" = TEDS)
<run_dir>   tools/ocrbench_run.py output (<id>.json cell lists)

Every sample with TEDS < 0.9 gets one primary type, checked in this order:
  non_latin     > 20 % of the ground truth's letters are not Latin
  not_found     no table found (<= 1 row or <= 1 column), or the prediction is
                mostly text that is not in the ground truth (surrounding text,
                titles, notes read as table rows)
  multiline     more rows than the ground truth AND some extra row continues a
                ground-truth cell (a wrapped cell split into two rows)
  columns       number of columns differs
  header_depth  header depth differs (only where the ground truth marks a header:
                <thead> rows or a top-left rowspan)
  spans         rows and columns right, but the set of merged cells differs
  other         structure right: text errors
plus all types that apply (multi-label counts).
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

from lxml import html as lhtml


def norm(t):
    return re.sub(r"\s+", " ", (t or "").replace("\\n", " ")).strip().lower()


def gt_grid(answer, markdown):
    """-> dict(R, C, H, spans, texts) from the ground truth (HTML or Markdown)."""
    if markdown:
        lines = [l for l in answer.strip().split("\n") if l.strip()]
        rows = [[c.strip() for c in l.strip().strip("|").split("|")] for i, l in enumerate(lines) if i != 1]
        return {"R": len(rows), "C": max(map(len, rows)) if rows else 0, "H": None, "spans": set(),
                "texts": [norm(c) for r in rows for c in r if norm(c)]}
    t = lhtml.fromstring(answer.replace("\\n", " "))
    tables = t.xpath("//table") or [t]
    tab = tables[0]
    rows, thead_rows = [], 0
    for el in tab.iter():
        kids = [k for k in el if k.tag in ("td", "th")]
        if el.tag == "tr" or (el.tag in ("thead", "tbody") and kids):
            rows.append((el, kids))
            if el.tag == "thead" or (el.getparent() is not None and el.getparent().tag == "thead"):
                thead_rows += 1
    occ, spans, texts = set(), set(), []
    C = 0
    for r, (_, kids) in enumerate(rows):
        c = 0
        for k in kids:
            while (r, c) in occ:
                c += 1
            rs, cs = int(k.get("rowspan", 1) or 1), int(k.get("colspan", 1) or 1)
            occ.update((r + i, c + j) for i in range(rs) for j in range(cs))
            if rs > 1 or cs > 1:
                spans.add((r, c, rs, cs))
            if norm(k.text_content()):
                texts.append(norm(k.text_content()))
            c += cs
        C = max(C, c)
    H = None
    if thead_rows:
        H = thead_rows
    elif rows and rows[0][1] and int(rows[0][1][0].get("rowspan", 1) or 1) > 1:
        H = int(rows[0][1][0].get("rowspan", 1))
    R = max((r for r, _ in occ), default=-1) + 1
    return {"R": R, "C": C, "H": H, "spans": spans, "texts": texts}


def pred_grid(cells):
    if not cells:
        return {"R": 0, "C": 0, "H": 1, "spans": set(), "rows": {}}
    R = max(x["r"] + x.get("rs", 1) for x in cells)
    C = max(x["c"] + x.get("cs", 1) for x in cells)
    H = next((x.get("rs", 1) for x in cells if x["r"] == 0 and x["c"] == 0), 1)
    rows = {}
    for x in cells:
        if norm(x["text"]):
            rows.setdefault(x["r"], []).append(norm(x["text"]))
    spans = {(x["r"], x["c"], x.get("rs", 1), x.get("cs", 1)) for x in cells
             if x.get("rs", 1) > 1 or x.get("cs", 1) > 1}
    return {"R": R, "C": C, "H": H, "spans": spans, "rows": rows}


def non_latin_share(texts):
    letters = [ch for t in texts for ch in t if ch.isalpha()]
    if not letters:
        return 0.0
    return sum("LATIN" not in unicodedata.name(ch, "") for ch in letters) / len(letters)


def classify(g, p):
    types = []
    if non_latin_share(g["texts"]) > 0.2:
        types.append("non_latin")
    gt_all = " | ".join(g["texts"])
    ptexts = [t for r in p["rows"].values() for t in r if len(t) >= 3]
    foreign = sum(t not in gt_all for t in ptexts) / len(ptexts) if ptexts else 1.0
    if p["R"] <= 1 or p["C"] <= 1 or (foreign > 0.5 and p["R"] > g["R"]):
        types.append("not_found")
    if p["R"] > g["R"]:
        gset = set(g["texts"])

        def part_of_gt_cell(t):           # a proper piece of a longer ground-truth cell
            return t not in gset and any(t in x for x in g["texts"])
        if any(r and all(part_of_gt_cell(t) for t in r) for r in p["rows"].values()):
            types.append("multiline")
    if p["C"] != g["C"]:
        types.append("columns")
    if g["H"] is not None and p["H"] != g["H"]:
        types.append("header_depth")
    if p["R"] == g["R"] and p["C"] == g["C"] and p["spans"] != g["spans"]:
        types.append("spans")
    return types or ["other"]


ORDER = ["non_latin", "not_found", "multiline", "columns", "header_depth", "spans", "other"]


def main(res_path, run_dir, out):
    res = json.loads(Path(res_path).read_text(encoding="utf-8"))
    run_dir = Path(run_dir)
    rows = []
    for x in res:
        if x["type"] != "table parsing en":
            continue
        sid = str(x["id"])
        md = "markdown" in x["question"].lower() and "html" not in x["question"].lower()
        g = gt_grid(x["answers"][0], md)
        f = run_dir / f"{sid}.json"
        p = pred_grid(json.loads(f.read_text(encoding="utf-8")) if f.exists() else [])
        types = classify(g, p)
        primary = min(types, key=ORDER.index)
        rows.append({"id": sid, "dataset": x["dataset_name"], "image": x["image_path"],
                     "teds": round(x["score"], 4), "gt_RxC": f"{g['R']}x{g['C']}",
                     "pred_RxC": f"{p['R']}x{p['C']}", "gt_H": g["H"], "pred_H": p["H"],
                     "gt_spans": len(g["spans"]), "pred_spans": len(p["spans"]),
                     "types": types, "primary": primary if x["score"] < 0.9 else None})
    Path(out).write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    bad = [r for r in rows if r["primary"]]
    print(f"{len(bad)} of {len(rows)} samples with TEDS < 0.9")
    for t in ORDER:
        prim = [r for r in bad if r["primary"] == t]
        multi = sum(t in r["types"] for r in bad)
        if prim or multi:
            mean = sum(r["teds"] for r in prim) / len(prim) if prim else float("nan")
            print(f"  {t:13s} primary {len(prim):3d}  (mean TEDS {mean:.3f})  any {multi:3d}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
