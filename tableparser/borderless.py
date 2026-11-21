"""Borderless path: OCR words -> rows -> phrases -> columns -> merged cells.

Challenges: 9/10/11 (no vertical lines), 14/15 (group headers by nearest
column), 22 (merged vs single by position + dash length), 23/24 (blank cells
by column position, never by order), 27-31 (alignment, tight gaps, row drift),
43 (split numbers joined), 45 (units attached), 26 (speck-only words dropped).
"""
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import budget
from .assemble import is_real_number
from .ocr import despeckle, normalize_light, ocr_line, ocr_words
from .preprocess import remove_specks
from .values import parse_cell

DASHES = set("-–—_~")
UNIT_WORD = re.compile(r"^(%|€|\$|kg|ms|EUR|USD|t|g)$", re.I)
NA_WORDS = {"n/a", "na", "-", "–", "—"}
COMPLETE_NUM_END = re.compile(r"\d[.,:]\d{2}\)?$|\)$")
NUM_START = re.compile(r"^[\d(\-−]")


def _words(pre):
    th = pre["th"]
    page = pre["gray"] if budget.fast() else despeckle(pre["gray"], th)   # fax specks out first (ch. 4)
    gray = normalize_light(page, th)
    words = ocr_words(gray, th)
    out = []
    for w in words:
        t = w["text"]
        if all(ch in DASHES for ch in t):
            continue                      # lone dashes come from the blob detector
        if not re.search(r"[0-9A-Za-zÄÖÜäöüß%€$]", t):
            continue                      # specks / punctuation only
        if w["conf"] < 20 and len(t) <= 2 and not t.isdigit():
            continue                      # fax noise read as letters
        out.append(w)
    return out


def _dash_blobs(pre, words):
    """Find lone dashes from the ink itself; measure long vs short (ch. 22)."""
    th = pre["th"]
    bw = remove_specks(pre["bw"], max(8, int(0.02 * th * th)))
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    dashes = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (w >= 2.5 * h and h <= 0.3 * th and 0.25 * th <= w <= 2.0 * th):
            continue
        cx, cy = x + w / 2, y + h / 2
        if any(wd["x0"] - 2 <= cx <= wd["x1"] + 2 and wd["y0"] - 2 <= cy <= wd["y1"] + 2 for wd in words):
            continue                      # part of a word (minus sign, underline in text)
        # a bar right before a number on the same line is a minus sign (ch. 39)
        nxt = [wd for wd in words if 0 <= wd["x0"] - (x + w) <= 0.6 * th
               and abs((wd["y0"] + wd["y1"]) / 2 - cy) < 0.5 * th and NUM_START.match(wd["text"])]
        if nxt:
            nxt[0]["text"] = "-" + nxt[0]["text"].lstrip("-")
            continue
        dashes.append({"text": "—" if w >= 0.8 * th else "–", "conf": 90.0,
                       "x0": float(x), "y0": float(cy - 0.35 * th), "x1": float(x + w),
                       "y1": float(cy + 0.35 * th)})
    return dashes


def _cluster_rows(words, th, key, tol):
    """Sort by y and start a new row at every gap > tol. More tolerant to
    per-word jitter (curved paper, keystone) than distance-to-row-mean."""
    words = sorted(words, key=key)
    rows, cur, prev = [], [], None
    for w in words:
        y = key(w)
        if cur and y - prev > tol:
            rows.append(cur)
            cur = []
        cur.append(w)
        prev = y
    if cur:
        rows.append(cur)
    # safety: a "row" taller than ~1.6 text heights is two rows glued together
    out = []
    for r in rows:
        ys = sorted(key(w) for w in r)
        if len(r) > 1 and ys[-1] - ys[0] > 1.6 * th:
            gaps = np.diff(ys)
            cut = ys[int(np.argmax(gaps))]
            out.append([w for w in r if key(w) <= cut])
            out.append([w for w in r if key(w) > cut])
        else:
            out.append(r)
    return out


def _rows(words, th):
    """Group words into text rows (ch. 31). Residual tilt / keystone makes a row's
    words drift in y, so: cluster once, measure the drift (median slope of rows
    with 3+ words), remove it, cluster again."""
    if not words:
        return []
    yc = lambda w: (w["y0"] + w["y1"]) / 2
    first = _cluster_rows(words, th, yc, 0.55 * th)
    slopes = []
    for r in first:
        if len(r) >= 3:
            xs = np.array([(w["x0"] + w["x1"]) / 2 for w in r])
            ys = np.array([yc(w) for w in r])
            if np.ptp(xs) > 5 * th:
                slopes.append(np.polyfit(xs, ys, 1)[0])
    m = float(np.median(slopes)) if slopes else 0.0
    x_mid = np.median([(w["x0"] + w["x1"]) / 2 for w in words])
    for w in words:
        w["yk"] = yc(w) - m * ((w["x0"] + w["x1"]) / 2 - x_mid)
    return _cluster_rows(words, th, lambda w: w["yk"], 0.75 * th)


def _phrases(row, th):
    """Join words into phrases (ch. 43, 48). Two safety checks (ch. 30):
    gap must be small relative to text height, and two complete numbers never join."""
    row = sorted(row, key=lambda w: w["x0"])
    out = []
    for w in row:
        if out:
            p = out[-1]
            gap = w["x0"] - p["x1"]
            unit = UNIT_WORD.match(w["text"]) and gap < 1.0 * th
            small = gap < 0.85 * th      # measured: inside phrases <=0.7, columns >=1.07
            two_numbers = COMPLETE_NUM_END.search(p["text"]) and NUM_START.match(w["text"])
            if (small and not two_numbers) or unit:
                p["text"] += " " + w["text"]
                p["x1"] = max(p["x1"], w["x1"])
                p["y0"], p["y1"] = min(p["y0"], w["y0"]), max(p["y1"], w["y1"])
                p["conf"] = min(p["conf"], w["conf"])
                continue
        out.append(dict(w))
    return out


def _split_k(values, k):
    """1D clustering into k groups by cutting at the k-1 largest gaps."""
    order = np.argsort(values)
    v = np.asarray(values)[order]
    if k <= 1 or len(v) < k:
        return [list(order)]
    gaps = np.diff(v)
    cuts = sorted(np.argsort(gaps)[-(k - 1):])
    groups, start = [], 0
    for c in cuts:
        groups.append(list(order[start:c + 1]))
        start = c + 1
    groups.append(list(order[start:]))
    return groups


def _split_gap(values, min_gap):
    order = np.argsort(values)
    v = np.asarray(values)[order]
    groups, cur = [], [order[0]]
    for a, b, idx in zip(v[:-1], v[1:], order[1:]):
        if b - a > min_gap:
            groups.append(cur)
            cur = []
        cur.append(idx)
    groups.append(cur)
    return groups


def borderless_cells(pre):
    th = pre["th"]
    words = _words(pre)
    dashes = _dash_blobs(pre, words)
    raw_rows = _rows(words, th)
    # dashes join the nearest text row (they sit higher/lower than the text centre)
    for d in dashes:
        cy = (d["y0"] + d["y1"]) / 2
        cx = (d["x0"] + d["x1"]) / 2
        ycs, dists = [], []
        for r in raw_rows:
            xs = np.array([(w["x0"] + w["x1"]) / 2 for w in r])
            ys = np.array([(w["y0"] + w["y1"]) / 2 for w in r])
            ycs.append(float(np.mean(ys)))
            if len(r) >= 2 and np.ptp(xs) > 3 * th:
                a, c0 = np.polyfit(xs, ys, 1)        # follow the row's own tilt
                dists.append(abs(cy - (a * cx + c0)))
            else:
                dists.append(abs(cy - ycs[-1]))
        spacing = np.median(np.diff(sorted(ycs))) if len(ycs) > 2 else 2 * th
        if dists and min(dists) < 0.5 * spacing:
            raw_rows[int(np.argmin(dists))].append(d)
    rows = [_phrases(r, th) for r in raw_rows]
    rows = [r for r in rows if r]

    # body = rows holding a real number; header = text rows directly above
    is_body = [any(is_real_number(p["text"]) for p in r) for r in rows]
    if not any(is_body):
        return [], 0, 0
    first = is_body.index(True)
    last = len(is_body) - 1 - is_body[::-1].index(True)
    body = rows[first:last + 1]
    header = rows[:first][-3:]                      # at most 3 header rows

    # label zone: left of every number in the body
    nums = [p for r in body for p in r if is_real_number(p["text"])]
    min_num_x0 = min(p["x0"] for p in nums)
    def is_label(p):
        return p["x1"] <= min_num_x0 + 0.2 * th and re.match(r"[A-Za-zÄÖÜäöü(]", p["text"])
    labels = {i: next((p for p in r if is_label(p)), None) for i, r in enumerate(body)}
    label_right = max([p["x1"] for p in labels.values() if p] + [min_num_x0 - th])
    label_x0 = min([p["x0"] for p in labels.values() if p] + [label_right - 8 * th])

    # columns from right edges of numbers (whole phrase incl. unit, ch. 27)
    rights = [p["x1"] for p in nums]
    hv_rows = [[p for p in r if p["x0"] > label_right - 0.2 * th] for r in header]
    header_val = max(hv_rows, key=len) if hv_rows else []
    K = len(header_val)
    groups = _split_k(rights, K) if K >= 1 and K <= len(set(np.round(rights))) else _split_gap(rights, 1.0 * th)
    # sanity: if header count gives clusters with internal spread > a column gap, fall back
    if K >= 1:
        spread = max(np.ptp([rights[i] for i in g]) for g in groups)
        if spread > 3.0 * th:
            groups = _split_gap(rights, 1.0 * th)
    cols = []
    for g in groups:
        ps = [nums[i] for i in g]
        cols.append({"right": float(np.median([p["x1"] for p in ps])),
                     "left": float(np.median([p["x0"] for p in ps])),
                     "lo": float(np.percentile([p["x0"] for p in ps], 10)),
                     "hi": float(np.percentile([p["x1"] for p in ps], 90))})
    cols.sort(key=lambda c: c["right"])
    K = len(cols)
    # boundaries between cells: b[0] label|col0, b[i] col(i-1)|col(i), b[K] table end
    b = [(label_right + cols[0]["lo"]) / 2]
    for i in range(K - 1):
        b.append((cols[i]["hi"] + cols[i + 1]["lo"]) / 2)
    b.append(cols[-1]["hi"] + 0.5 * th)

    def place_number(p):
        return int(np.argmin([abs(p["x1"] - c["right"]) for c in cols]))

    def place_marker(p):
        """Single cell vs merged pair: whichever centre is closest (ch. 22).
        Cell centres come from the header (headers are centred in their cells)."""
        c = (p["x0"] + p["x1"]) / 2
        best, best_d = (0, 0), 1e9
        for i in range(K):
            d = abs(c - centers[i])
            if p["text"] == "–":
                d *= 0.85            # dash length is only a weak hint (ex. 5 breaks it)
            if d < best_d:
                best, best_d = (i, i), d
            if i + 1 < K:
                d2 = abs(c - (centers[i] + centers[i + 1]) / 2)
                if p["text"] == "—":
                    d2 *= 0.85
                if d2 < best_d:
                    best, best_d = (i, i + 1), d2
        return best

    def place_header(p):
        c = (p["x0"] + p["x1"]) / 2
        return int(np.argmin([abs(c - (b[i] + b[i + 1]) / 2) for i in range(K)]))

    # cell centres: header centre when the bottom header row has one, else boundaries
    centers = [(b[i] + b[i + 1]) / 2 for i in range(K)]
    if header_val:
        hc = {}
        for p in header_val:
            hc.setdefault(place_header(p), []).append((p["x0"] + p["x1"]) / 2)
        for k, cs in hc.items():
            centers[k] = float(np.mean(cs))

    cells = []
    # ---- header rows
    label_head = [p["text"] for r in header for p in r if p["x1"] <= label_right + 0.2 * th]
    hrows = [[p for p in r if p["x1"] > label_right + 0.2 * th] for r in header]
    hrows = [r for r in hrows if r]
    if hrows:
        # the bottom header row is the fullest one; rows below it are noise (ch. 19)
        best = max(range(len(hrows)), key=lambda i: (len(hrows[i]), i))
        hrows = [r for r in hrows[:best] if any(len(p["text"]) >= 2 for p in r)] + [hrows[best]]
    H = len(hrows)
    for hi, r in enumerate(hrows):
        if hi == H - 1:                      # bottom header row: one phrase per column
            slots = {}
            for p in r:
                slots.setdefault(place_header(p), []).append(p["text"])
            for k, ts in slots.items():
                ps = [p for p in r if place_header(p) == k]
                cells.append({"r0": hi, "r1": hi, "c0": k + 1, "c1": k + 1, "text": " ".join(ts), "conf": 90,
                              "bbox": _union(ps)})
        else:                                # group row: each column -> nearest group (ch. 14/15)
            gcent = [(p["x0"] + p["x1"]) / 2 for p in r]
            owner = [int(np.argmin([abs(centers[k] - c) for c in gcent])) for k in range(K)]
            for gi, p in enumerate(r):
                ks = [k for k in range(K) if owner[k] == gi]
                if ks:
                    cells.append({"r0": hi, "r1": hi, "c0": min(ks) + 1, "c1": max(ks) + 1,
                                  "text": p["text"], "conf": 90, "bbox": _union([p])})
    if H:
        cells.append({"r0": 0, "r1": H - 1, "c0": 0, "c1": 0, "text": " ".join(label_head), "conf": 90})

    # ---- body rows
    for bi, r in enumerate(body):
        R = H + bi
        lab = labels[bi]
        if lab:
            cells.append({"r0": R, "r1": R, "c0": 0, "c1": 0, "text": lab["text"], "conf": lab["conf"],
                          "bbox": _union([lab])})
        else:
            # page OCR skipped the label: read the label zone of this row on its own
            y0 = min(p["y0"] for p in r)
            y1 = max(p["y1"] for p in r)
            cells.append({"r0": R, "r1": R, "c0": 0, "c1": 0, "text": "", "conf": 0, "force": True,
                          "bbox": [label_x0, y0, label_right, y1]})
        taken = {}
        for p in r:
            if p is lab:
                continue
            if is_real_number(p["text"]) or (parse_cell(p["text"])["kind"] == "number"
                                             and p["text"].lower() not in NA_WORDS):
                c0 = c1 = place_number(p)
            else:
                c0, c1 = place_marker(p)
            key = (c0, c1)
            if key in taken:                       # two pieces of one cell
                taken[key]["text"] += " " + p["text"]
                taken[key]["bbox"] = _union([p], taken[key]["bbox"])
            else:
                taken[key] = {"r0": R, "r1": R, "c0": c0 + 1, "c1": c1 + 1, "text": p["text"],
                              "conf": p["conf"], "bbox": _union([p]),
                              "dash": p["text"] in ("—", "–")}
        covered = set()
        for cell in taken.values():
            cells.append(cell)
            covered.update(range(cell["c0"], cell["c1"] + 1))
        for k in range(1, K + 1):                  # blank cells (ch. 23/24)
            if k not in covered:
                cells.append({"r0": R, "r1": R, "c0": k, "c1": k, "text": "", "conf": 100})
    _reread(cells, pre)
    return cells, H + len(body), K + 1


def _union(ps, bbox=None):
    xs0 = [p["x0"] for p in ps] + ([bbox[0]] if bbox else [])
    ys0 = [p["y0"] for p in ps] + ([bbox[1]] if bbox else [])
    xs1 = [p["x1"] for p in ps] + ([bbox[2]] if bbox else [])
    ys1 = [p["y1"] for p in ps] + ([bbox[3]] if bbox else [])
    return [min(xs0), min(ys0), max(xs1), max(ys1)]


def _reread(cells, pre):
    """Re-OCR every phrase as its own crop with the cell reader (better than the
    page pass, and gives the second-reader cross-check for borderless tables too)."""
    th, gray = pre["th"], pre["gray"]
    H, W = gray.shape
    m = 0.3 * th

    def run(c):
        if (not c.get("text") and not c.get("force")) or c.get("dash") or "bbox" not in c:
            return
        if budget.expired():               # over the table budget: keep the page read
            budget.skip()
            return
        x0, y0, x1, y1 = c["bbox"]
        crop = gray[max(0, int(y0 - m)):min(H, int(y1 + m)), max(0, int(x0 - m)):min(W, int(x1 + m))]
        res = ocr_line(crop, th)
        if res["text"]:
            c["page_text"] = c["text"]
            c["text"], c["conf"], c["alt"] = res["text"], res["conf"], res["alt"]
            c["sign_doubt"] = res.get("sign_doubt", False)
            # keep the page read as an extra opinion when the two differ on a number
            if c["page_text"].replace(" ", "") != res["text"].replace(" ", "") and not res["alt"]:
                c["alt"] = {"text": c["page_text"], "conf": 50.0}

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(run, cells))
