#!/usr/bin/env python3
"""Run tableparser or the Tesseract floor on the OCRBench v2 "table parsing en" subset.

    python tools/ocrbench_run.py tableparser|tesseract <OCRBench_v2 root> <out_dir> [--timeout 60]

Reads <root>/OCRBench_v2.json, keeps type == "table parsing en" (400 samples),
and writes per sample id:
    <out_dir>/<id>.json   tableparser: submission cell list (tableparser_adapter.to_cells)
    <out_dir>/<id>.html   tesseract:   the baseline's HTML table
plus <out_dir>/_runs.json: {id: {"s": seconds, "status": ok|error|timeout, "err": ...}}.
Already finished ids are skipped, so an interrupted run can be resumed.
Measurement only: tableparser is called exactly as in the submission.
"""
import argparse
import importlib.util
import json
import signal
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=["tableparser", "tesseract"])
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--timeout", type=int, default=60)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = [x for x in json.loads((root / "OCRBench_v2.json").read_text(encoding="utf-8"))
             if x["type"] == "table parsing en"]

    if a.system == "tableparser":
        ad = load("ad", REPO / "submission_template" / "tableparser_adapter.py")

        def run(img, sid):
            cells = ad.to_cells(ad.parse_table(str(img)))
            (out / f"{sid}.json").write_text(json.dumps(cells, ensure_ascii=False), encoding="utf-8")
    else:
        bt = load("bt", REPO / "baselines" / "baseline_tesseract.py")

        def run(img, sid):
            (out / f"{sid}.html").write_text(bt.to_html(img), encoding="utf-8")

    runs_path = out / "_runs.json"
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else {}
    signal.signal(signal.SIGALRM, _alarm)
    for x in items:
        sid = str(x["id"])
        if sid in runs:
            continue
        t0 = time.perf_counter()
        rec = {"status": "ok"}
        signal.alarm(a.timeout)
        try:
            run(root / x["image_path"], sid)
        except Timeout:
            rec = {"status": "timeout"}
        except Exception as e:
            rec = {"status": "error", "err": f"{type(e).__name__}: {e}"[:300]}
            traceback.print_exc()
        finally:
            signal.alarm(0)
        rec["s"] = round(time.perf_counter() - t0, 3)
        runs[sid] = rec
        runs_path.write_text(json.dumps(runs, indent=1))
        print(sid, x["dataset_name"], rec["status"], rec["s"], flush=True)


if __name__ == "__main__":
    main()
