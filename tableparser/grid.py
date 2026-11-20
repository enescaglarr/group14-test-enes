"""Bordered path: cells from the line grid, merges from missing separators,
OCR per cell with lines removed.

Challenges: 7, 8 (broken + drifting lines: searched in a window per row),
12 (lines erased before OCR), 16/20/21 (vertical + horizontal merges via
missing separators), 23/24 (blank cells exist as real cells, nothing shifts).
"""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .ocr import ocr_line


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _has_vsep(vmask, x, y0, y1, tol):
    h = y1 - y0
    m = int(0.2 * h)
    win = vmask[y0 + m:y1 - m, max(0, x - tol):x + tol + 1]
    if win.size == 0:
        return True
    return win.any(axis=1).mean() > 0.5


def _has_hsep(hmask, y, x0, x1, tol):
    w = x1 - x0
    m = int(0.15 * w)
    win = hmask[max(0, y - tol):y + tol + 1, x0 + m:x1 - m]
    if win.size == 0:
        return True
    return win.any(axis=0).mean() > 0.5


def bordered_cells(pre):
    gray, th = pre["gray"], pre["th"]
    hmask, vmask = pre["hmask"], pre["vmask"]
    xs, ys = sorted(pre["vlines"]), sorted(pre["hlines"])
    # keep only horizontal lines inside the vertical extent of the grid
    R, C = len(ys) - 1, len(xs) - 1
    tol = max(4, int(0.8 * th))
    uf = _UF(R * C)
    for r in range(R):
        for c in range(C - 1):
            if not _has_vsep(vmask, xs[c + 1], ys[r], ys[r + 1], tol):
                uf.union(r * C + c, r * C + c + 1)
    for r in range(R - 1):
        for c in range(C):
            if not _has_hsep(hmask, ys[r + 1], xs[c], xs[c + 1], max(3, int(0.4 * th))):
                uf.union(r * C + c, (r + 1) * C + c)

    groups = {}
    for r in range(R):
        for c in range(C):
            groups.setdefault(uf.find(r * C + c), []).append((r, c))

    # erase lines before OCR so borders are not read as 1 / l / |
    k = max(5, int(0.3 * th)) | 1
    lines = cv2.dilate(hmask | vmask, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    clean = gray.copy()
    clean[lines > 0] = 255

    inset = max(3, int(0.25 * th))
    jobs = []
    for slots in groups.values():
        r0, r1 = min(s[0] for s in slots), max(s[0] for s in slots)
        c0, c1 = min(s[1] for s in slots), max(s[1] for s in slots)
        x0, x1 = xs[c0] + inset, xs[c1 + 1] - inset
        y0, y1 = ys[r0] + inset, ys[r1 + 1] - inset
        jobs.append((r0, r1, c0, c1, x0, y0, x1, y1))

    def run(job):
        r0, r1, c0, c1, x0, y0, x1, y1 = job
        res = ocr_line(clean[y0:y1, x0:x1], th)
        return {"r0": r0, "r1": r1, "c0": c0, "c1": c1, "text": res["text"], "conf": res["conf"],
                "alt": res["alt"], "sign_doubt": res.get("sign_doubt", False), "bbox": [int(x0), int(y0), int(x1), int(y1)]}

    with ThreadPoolExecutor(max_workers=8) as ex:
        cells = list(ex.map(run, jobs))
    return cells, R, C
