"""Entry point: image -> structured, LLM-ready table."""
import json
import sys
import time

from . import budget
from .assemble import assemble
from .preprocess import preprocess


MAX_PLAUSIBLE_COLS = 12


def plausibility(table):
    """How table-like an assembled result is: enough body rows, a sane number
    of columns, rows that fill about the same number of cells, few blanks."""
    R, C = len(table["rows"]), len(table["columns"])
    if R == 0 or C == 0:
        return 0.0
    filled = [0] * R
    blank = 0
    for rec in table["cells"]:
        if 0 <= rec["row_index"] < R:
            if rec["kind"] == "empty" and "blank" in rec["flags"]:
                blank += 1
            else:
                filled[rec["row_index"]] += len(rec["col_index"])
    fill = 1.0 - blank / max(1, len(table["cells"]))
    mean = sum(filled) / R
    spread = (sum((f - mean) ** 2 for f in filled) / R) ** 0.5
    consistency = max(0.0, 1.0 - spread / max(1.0, C))
    rows_ok = min(1.0, R / 2.0)
    cols_ok = 1.0 if C <= MAX_PLAUSIBLE_COLS else MAX_PLAUSIBLE_COLS / C
    return fill * consistency * rows_ok * cols_ok


def _third_reader(pre, path_kind):
    """Crop reader for the ONNX recognizer (None if it is unavailable).
    Bordered bboxes are already inset from the lines; borderless ones are word
    boxes and get the same margin as the phrase re-read."""
    from . import rec_onnx
    from .ablation import off
    if off("third") or not rec_onnx.available():
        return None
    gray, th = pre["gray"], pre["th"]
    H, W = gray.shape
    m = 0 if path_kind == "bordered" else 0.3 * th

    def read(bbox):
        x0, y0, x1, y1 = bbox
        crop = gray[max(0, int(y0 - m)):min(H, int(y1 + m)), max(0, int(x0 - m)):min(W, int(x1 + m))]
        return rec_onnx.read(crop) if crop.size else ("", 0.0)
    return read


def _run(path_kind, pre, meta):
    if path_kind == "bordered":
        from .grid import bordered_cells
        cells, R, C = bordered_cells(pre)
    else:
        from .borderless import borderless_cells
        cells, R, C = borderless_cells(pre)
    return assemble(cells, R, C, {**meta, "path": path_kind},
                    third_reader=_third_reader(pre, path_kind))


def parse_table(path):
    t0 = time.time()
    budget.start()
    pre = preprocess(path)
    # triage: clean scans (T0/T1) -> fast path, 4 s; photos / faxes -> full path, 12 s
    budget.set_path(pre.get("tier") in ("T0", "T1"))
    meta = {"source": str(path), "rotation_deg": round(pre["angle"], 2),
            "shear_deg": round(pre["shear"], 2), "text_height_px": pre["th"],
            "tier_guess": pre.get("tier"), "fast_path": budget.fast()}
    bordered = len(pre["vlines"]) >= 3     # router (ch. 7-11): needs vertical lines
    table = _run("bordered" if bordered else "borderless", pre, meta)
    if bordered and (len(table["rows"]) < 2 or len(table["columns"]) > MAX_PLAUSIBLE_COLS) \
            and not budget.hard_expired():
        # fallback: the line grid gave an implausible table (false vertical lines
        # from aligned glyph stems, or a grid without body rows) -> try borderless
        alt = _run("borderless", pre, meta)
        sb, sa = plausibility(table), plausibility(alt)
        chosen = alt if sa > sb else table
        chosen["meta"]["fallback"] = {"tried": "borderless", "bordered_score": round(sb, 3),
                                      "borderless_score": round(sa, 3),
                                      "chosen": chosen["meta"]["path"]}
        table = chosen
    table["meta"]["seconds"] = round(time.time() - t0, 1)
    table["meta"]["budget_skipped"] = budget.skipped()
    table["meta"]["cells_unread"] = budget.n_unread()
    table["meta"]["large_table"] = budget.large()
    return table


def main():
    if len(sys.argv) < 2:
        print("usage: python -m tableparser.pipeline IMAGE [IMAGE ...]")
        sys.exit(1)
    for p in sys.argv[1:]:
        t = parse_table(p)
        print(f"\n## {p}  ({t['meta']['path']}, {t['meta']['seconds']}s, "
              f"{t['meta']['n_flagged']} flagged)\n")
        print(t["markdown"])
        out = p.rsplit(".", 1)[0] + ".table.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(t, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
