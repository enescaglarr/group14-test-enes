"""Entry point: image -> structured, LLM-ready table."""
import json
import sys
import time

from .assemble import assemble
from .preprocess import preprocess


def parse_table(path):
    t0 = time.time()
    pre = preprocess(path)
    bordered = len(pre["vlines"]) >= 3     # router (ch. 7-11): needs vertical lines
    if bordered:
        from .grid import bordered_cells
        cells, R, C = bordered_cells(pre)
    else:
        from .borderless import borderless_cells
        cells, R, C = borderless_cells(pre)
    meta = {"source": str(path), "path": "bordered" if bordered else "borderless",
            "rotation_deg": round(pre["angle"], 2), "shear_deg": round(pre["shear"], 2),
            "text_height_px": pre["th"]}
    table = assemble(cells, R, C, meta)
    table["meta"]["seconds"] = round(time.time() - t0, 1)
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
