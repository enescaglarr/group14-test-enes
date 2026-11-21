#!/usr/bin/env python3
"""Ablation runs: Tesseract floor -> full tableparser, one part at a time.

    python tools/ablation_run.py V1|V2|V3|V4|V5|V6 <src> <out_dir> [--timeout 60]
                                 [--shard I/N] [--code DIR]

    V1  plain Tesseract floor (baselines/baseline_tesseract.py), original image
    V2  tableparser preprocessing (crop, upscale, deskew, shear, light normalization,
        despeckle) on the whole page, then exactly V1's Tesseract floor on that image
    V3  no preprocessing, but tableparser structure + cell reading + values + assemble
        (TABLEPARSER_ABLATE=preprocess,triage,third)
    V4  preprocessing + structure + post-processing; triage and third reader off
        (TABLEPARSER_ABLATE=triage,third)
    V5  full system as submitted (no flags)
    V6  full system, but PaddleOCR recognition (rec_onnx) is the cell reader; Tesseract only
        as fallback (Paddle confidence < 90 or empty) and second opinion on numbers
        (TABLEPARSER_ABLATE=third,paddle_primary: the third reader is that same model)

<src> is a folder of <id>.png|.jpg (a flattened dataset) or ocrbench:<OCRBench_v2 root>
(the 400 "table parsing en" samples, named by sample id). Writes <id>.html (V1, V2) or
<id>.json cell lists (V3-V6) and _runs.json {id: {s, status}}; resumes finished ids.

--shard I/N  process every N-th image starting at I (0-based), for N parallel processes
             writing into the same <out_dir>; the run log is then _runs.sI.json.
             Parallel runs are for scores only: their seconds are not clean timings.
--code DIR   use tableparser / baselines / adapter from DIR (e.g. a `git archive` of a
             commit) instead of this checkout.
"""
import argparse
import importlib.util
import json
import os
import signal
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FLAGS = {"V1": None, "V2": "", "V3": "preprocess,triage,third", "V4": "triage,third", "V5": "",
         "V6": "third,paddle_primary"}


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sources(src):
    if src.startswith("ocrbench:"):
        root = Path(src.split(":", 1)[1])
        d = json.loads((root / "OCRBench_v2.json").read_text(encoding="utf-8"))
        return [(str(x["id"]), root / x["image_path"]) for x in d if x["type"] == "table parsing en"]
    return [(p.stem, p) for p in sorted(Path(src).iterdir())
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version", choices=sorted(FLAGS))
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--code", default=None)
    a = ap.parse_args()
    global REPO
    if a.code:
        REPO = Path(a.code).resolve()
    # the flags must be set before tableparser is imported
    if FLAGS[a.version] is not None:
        os.environ["TABLEPARSER_ABLATE"] = FLAGS[a.version]
    sys.path.insert(0, str(REPO))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.version in ("V1", "V2"):
        bt = load("bt", REPO / "baselines" / "baseline_tesseract.py")
        if a.version == "V2":
            import cv2
            from tableparser import budget
            from tableparser.ocr import despeckle, normalize_light
            from tableparser.preprocess import preprocess
            tmp = Path(tempfile.mkdtemp())

        def run(img, sid):
            if a.version == "V2":
                budget.start()
                pre = preprocess(str(img))
                page = normalize_light(despeckle(pre["gray"], pre["th"]), pre["th"])
                img = tmp / "page.png"
                cv2.imwrite(str(img), page)
            (out / f"{sid}.html").write_text(bt.to_html(img), encoding="utf-8")
    else:
        ad = load("ad", REPO / "submission_template" / "tableparser_adapter.py")

        def run(img, sid):
            cells = ad.to_cells(ad.parse_table(str(img)))
            (out / f"{sid}.json").write_text(json.dumps(cells, ensure_ascii=False), encoding="utf-8")

    items = sources(a.src)
    runs_path = out / "_runs.json"
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        items = items[i::n]
        runs_path = out / f"_runs.s{i}.json"
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else {}
    signal.signal(signal.SIGALRM, _alarm)
    for sid, img in items:
        if sid in runs:
            continue
        t0 = time.perf_counter()
        rec = {"status": "ok"}
        signal.alarm(a.timeout)
        try:
            run(img, sid)
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
        print(sid, rec["status"], rec["s"], flush=True)


if __name__ == "__main__":
    main()
