#!/usr/bin/env python3
"""Build OCRBench v2 prediction files from tools/ocrbench_run.py outputs and score
them with the OFFICIAL evaluation code (Yuliang-Liu/MultimodalOCR, OCRBench_v2/eval_scripts).

    python tools/ocrbench_score.py build <OCRBench_v2 root> <run_dir> <system> <variant> <out.json>
        system  tableparser (reads <id>.json cell lists) | tesseract (reads <id>.html)
        variant td  all cells as <td>            (main result)
                th  header rows (r < header depth) as <th> in <thead>  (sensitivity check)
        Writes the official prediction format: the subset's items + a "predict" string.
        Items whose question asks for Markdown get a Markdown table (the official scorer
        parses those with convert_markdown_table_to_html); all others get HTML.
        A sample without output (error / timeout) gets predict "" -> official score 0.

    python tools/ocrbench_score.py teds_s <eval_scripts dir> <pred.json> <out.json>
        Structure-only TEDS (TEDS-S) with the official TEDS class (structure_only=True)
        and exactly the preprocessing of eval.py's "table parsing en" branch.
        (The official TEDS itself comes from running eval.py unchanged.)
"""
import html
import json
import re
import sys
from pathlib import Path

from lxml import html as lhtml


def subset(root):
    d = json.loads((Path(root) / "OCRBench_v2.json").read_text(encoding="utf-8"))
    return [x for x in d if x["type"] == "table parsing en"]


def html_to_cells(s):
    """Tesseract baseline HTML -> cell list (it never writes spans, but be general)."""
    t = lhtml.fromstring(s) if s.strip() else None
    cells, occ = [], set()
    if t is None:
        return cells
    for r, tr in enumerate(t.iter("tr")):
        c = 0
        for td in tr:
            if td.tag not in ("td", "th"):
                continue
            while (r, c) in occ:
                c += 1
            rs, cs = int(td.get("rowspan", 1)), int(td.get("colspan", 1))
            cells.append({"text": td.text_content(), "r": r, "c": c, "rs": rs, "cs": cs})
            occ.update((r + i, c + j) for i in range(rs) for j in range(cs))
            c += cs
    return cells


def header_depth(cells):
    """Header rows = the rows covered by the top-left cell (adapter: rs = H)."""
    for x in cells:
        if x["r"] == 0 and x["c"] == 0:
            return x.get("rs", 1)
    return 1


def to_html(cells, th=False):
    if not cells:
        return "<table></table>"
    H = header_depth(cells) if th else 0
    rows = {}
    for x in cells:
        rows.setdefault(x["r"], []).append(x)
    R = max(rows) + 1
    out = ["<table>"]
    if H:
        out.append("<thead>")
    for r in range(R):
        if H and r == H:
            out.append("</thead><tbody>")
        tag = "th" if r < H else "td"
        tds = []
        for x in sorted(rows.get(r, []), key=lambda y: y["c"]):
            att = ""
            if x.get("rs", 1) > 1:
                att += f' rowspan="{x["rs"]}"'
            if x.get("cs", 1) > 1:
                att += f' colspan="{x["cs"]}"'
            tds.append(f"<{tag}{att}>{html.escape(x['text'])}</{tag}>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    if H:
        out.append("</tbody>" if R > H else "</thead>")
    out.append("</table>")
    return "".join(out)


def to_markdown(cells):
    """Markdown has no spans: a spanning cell's text goes to its top-left position,
    the positions it covers stay empty. Row 0 is the Markdown header row."""
    if not cells:
        return ""
    R = max(x["r"] + x.get("rs", 1) for x in cells)
    C = max(x["c"] + x.get("cs", 1) for x in cells)
    g = [[""] * C for _ in range(R)]
    for x in cells:
        g[x["r"]][x["c"]] = x["text"].replace("|", "\\|").replace("\n", " ")
    line = lambda row: "| " + " | ".join(row) + " |"
    return "\n".join([line(g[0]), "| " + " | ".join(["---"] * C) + " |"] + [line(r) for r in g[1:]])


def build(root, run_dir, system, variant, out):
    run_dir = Path(run_dir)
    runs = json.loads((run_dir / "_runs.json").read_text())
    res = []
    for x in subset(root):
        sid = str(x["id"])
        cells = None
        if runs.get(sid, {}).get("status") == "ok":
            if system == "tableparser":
                cells = json.loads((run_dir / f"{sid}.json").read_text(encoding="utf-8"))
            else:
                cells = html_to_cells((run_dir / f"{sid}.html").read_text(encoding="utf-8"))
        if cells is None:
            pred = ""
        elif "markdown" in x["question"].lower() and "html" not in x["question"].lower():
            pred = to_markdown(cells)
        else:
            pred = to_html(cells, th=(variant == "th"))
        res.append({**x, "predict": pred})
    Path(out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"{out}: {len(res)} items, {sum(1 for r in res if not r['predict'])} empty")


def teds_s(eval_dir, pred, out):
    """Mirror of eval.py's "table parsing en" branch with TEDS(structure_only=True)."""
    sys.path.insert(0, str(eval_dir))
    from TEDS_metric import TEDS, convert_markdown_table_to_html, wrap_html_table
    teds = TEDS(structure_only=True, n_jobs=1)
    scores = {}
    for d in json.loads(Path(pred).read_text(encoding="utf-8")):
        q, p = d["question"].lower(), d["predict"]
        s = 0.0
        if not isinstance(p, str):
            s = 0.0
        elif "html" in q:
            t = p.replace("\n", "")
            if "<body" in t:
                t = re.findall("<body.*", t)[0]
            elif "<table" in t:
                t = re.findall("<table.*", t)[0]
            else:
                t = None
            if t is not None:
                try:
                    s = teds.evaluate(wrap_html_table(t), wrap_html_table(d["answers"][0]))
                except Exception:
                    s = 0.0
        elif "markdown" in q:
            s = teds.evaluate(convert_markdown_table_to_html(p),
                              convert_markdown_table_to_html(d["answers"][0]))
        scores[str(d["id"])] = s
    Path(out).write_text(json.dumps(scores, indent=1))
    print(f"{out}: TEDS-S mean {sum(scores.values()) / len(scores):.4f} over {len(scores)}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(*sys.argv[2:7])
    elif cmd == "teds_s":
        teds_s(*sys.argv[2:5])
    else:
        sys.exit(__doc__)
