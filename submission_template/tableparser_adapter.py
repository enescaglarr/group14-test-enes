#!/usr/bin/env python3
"""
Adapter: tableparser -> TableSalt submission format.

    python submission_template/tableparser_adapter.py <input_dir> <output_dir>

Runs tableparser.parse_table on every <table_id>.png|.jpg in <input_dir> and
writes <output_dir>/<table_id>.json as a cell list [{"text","r","c","rs","cs"}]
(README §5). Same contract as predict.py, so it can be dropped in as the
container entrypoint. Each file is written as soon as its table is done.

tableparser's output is LLM-oriented (header *paths* per column, one record per
body cell); this module rebuilds the grid from it:
  * header: one row per header level (depth = longest header path); equal consecutive path prefixes become a
    colspan, a path that ends early gets a rowspan down to the body
  * top-left cell spans all header rows; tableparser's placeholders
    ("label", "colN") become empty text
  * body: label in column 0, each value at col_index[0]+1 with cs=len(col_index);
    any grid position not covered gets an empty cell (README: write every cell)
  * text is tableparser's "raw" string (as printed), not the parsed value --
    except for cells tableparser rebuilt in the table's own number format
    ("repaired_raw": junk characters / broken separators from OCR)
"""
import json
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tableparser.pipeline import parse_table  # noqa: E402

_PLACEHOLDER_COL = re.compile(r"col\d+")


def to_cells(t: dict) -> list:
    cols = [[] if len(p) == 1 and _PLACEHOLDER_COL.fullmatch(p[0]) else list(p)
            for p in t["columns"]]
    # header depth from the paths, not meta["header_rows"]: that count can include
    # an empty header row (dev table_022), which would add a phantom row
    H = max([1] + [len(p) for p in cols])
    C = len(cols)
    label = "" if t["label_header"] == "label" else t["label_header"]
    out = [{"text": label, "r": 0, "c": 0, "rs": H, "cs": 1}]

    for lvl in range(H):
        k = 0
        while k < C:
            p = cols[k]
            if lvl > 0 and lvl >= len(p):          # covered by a rowspan from above
                k += 1
                continue
            last = len(p) == 0 or lvl == len(p) - 1
            j = k + 1
            if not last:                           # group header: merge equal prefixes
                while j < C and cols[j][:lvl + 1] == p[:lvl + 1]:
                    j += 1
            out.append({"text": p[lvl] if p else "", "r": lvl, "c": k + 1,
                        "rs": H - lvl if last else 1, "cs": j - k})
            k = j

    filled = set()
    for i, lab in enumerate(t["rows"]):
        out.append({"text": lab, "r": H + i, "c": 0, "rs": 1, "cs": 1})
    for cell in t["cells"]:
        ci = cell["col_index"]
        r = H + cell["row_index"]
        text = cell.get("repaired_raw") or cell.get("raw") or ""
        out.append({"text": text, "r": r, "c": ci[0] + 1,
                    "rs": 1, "cs": len(ci)})
        filled.update((r, k + 1) for k in ci)
    for i in range(len(t["rows"])):
        for k in range(1, C + 1):
            if (H + i, k) not in filled:
                out.append({"text": "", "r": H + i, "c": k, "rs": 1, "cs": 1})
    return sorted(out, key=lambda x: (x["r"], x["c"]))


def main(inp, out):
    inp, out = Path(inp), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    times = {}
    for img in sorted(inp.iterdir()):
        if img.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        t0 = time.monotonic()
        try:
            cells = to_cells(parse_table(str(img)))
        except Exception:
            traceback.print_exc()
            cells = [{"text": "", "r": 0, "c": 0}]   # valid file beats a missing one
        (out / f"{img.stem}.json").write_text(
            json.dumps(cells, ensure_ascii=False), encoding="utf-8")
        times[img.stem] = time.monotonic() - t0
        print(f"{img.stem} done {times[img.stem]:.1f}s", flush=True)
    if times:
        worst = max(times, key=times.get)
        print(f"{len(times)} tables, mean {sum(times.values()) / len(times):.1f}s, "
              f"max {times[worst]:.1f}s ({worst})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
