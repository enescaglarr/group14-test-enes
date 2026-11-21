"""Image preprocessing: crop dark frame, upscale, deskew, binarize, line masks.

Challenges handled: 1 (rotation), 3 (uneven light), 4 (fax specks, for angle
estimation only), 5 (blur, via upscaling), 7-11 (line detection + router).
"""
import cv2
import numpy as np

from .ablation import off


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def crop_dark_frame(gray, dark=70):
    """Screenshots / photos often have a dark frame around the page. Cut it off
    so its edges are not mistaken for table borders."""
    h, w = gray.shape
    rows = np.where(np.median(gray, axis=1) > dark)[0]
    cols = np.where(np.median(gray, axis=0) > dark)[0]
    if len(rows) == 0 or len(cols) == 0:
        return gray
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    pad = 6  # also drop the anti-aliased edge of the frame -- only on sides where a
    #          frame was actually cut; trimming a frameless edge eats glyphs that
    #          touch it (tight crops lost the first letter of every label)
    y0 = min(y0 + pad, h - 1) if y0 > 0 else 0
    x0 = min(x0 + pad, w - 1) if x0 > 0 else 0
    y1 = max(y1 - pad, 1) if y1 < h else h
    x1 = max(x1 - pad, 1) if x1 < w else w
    return gray[y0:y1, x0:x1]


def binarize(gray):
    """Ink = 255. Adaptive threshold copes with gradients and gray stripes."""
    block = max(15, (min(gray.shape) // 30) | 1)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, 15)


def remove_specks(bw, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


TH_MIN, TH_MAX = 12.0, 80.0


def text_height(bw, clamp=True):
    """Typical text height in px from ink blobs.

    Single characters AND merged letter blobs (blur / bold fuse letters into
    word blobs) are used: a fused word still has the text's height, while
    dropping it leaves mostly specks and dots, which made blurred scans read
    as ~9 px text. Clamped to [TH_MIN, TH_MAX]."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    ws = stats[1:, cv2.CC_STAT_WIDTH]
    areas = stats[1:, cv2.CC_STAT_AREA]
    # ws <= 15*hs keeps fused words but drops table lines / underlines
    ok = (hs >= 6) & (hs <= 120) & (ws <= 15 * hs) & (ws >= 2) & (areas >= 15)
    if ok.sum() < 10:
        th = 20.0
    else:
        h = hs[ok]
        # fax specks are many and small: characters are the upper half of the blobs.
        # Digits/capitals share one height, so take the mode of the upper half.
        upper = h[h >= np.percentile(h, 50)]
        vals, counts = np.unique(upper, return_counts=True)
        # smooth the histogram a little (heights jitter by +-1 px)
        sm = np.convolve(counts, [1, 1, 1], mode="same")
        th = float(vals[int(np.argmax(sm))])
    return float(np.clip(th, TH_MIN, TH_MAX)) if clamp else th


def tesseract_word_height(gray, with_conf=False):
    """Median height of confident Tesseract word boxes (None if too few).
    Independent of specks/dots: those are not returned as words.
    with_conf=True also returns the mean confidence of all words (triage)."""
    import pytesseract
    h, w = gray.shape
    s = 1.0
    if max(h, w) > 1600:                      # keep the cross-check cheap
        s = 1600.0 / max(h, w)
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    pad = max(8, int(0.02 * max(gray.shape)))      # white margin for edge-touching text
    gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    d = pytesseract.image_to_data(gray, lang="eng", config="--psm 11",
                                  output_type=pytesseract.Output.DICT)
    hs = [d["height"][i] / s for i, t in enumerate(d["text"])
          if t.strip() and float(d["conf"][i]) >= 60 and sum(ch.isalnum() for ch in t) >= 2]
    med = float(np.median(hs)) if len(hs) >= 5 else None
    if not with_conf:
        return med
    confs = [float(c) for t, c in zip(d["text"], d["conf"]) if t.strip() and float(c) >= 0]
    return med, (float(np.mean(confs)) if confs else 0.0)


def cross_checked_height(gray, bw, info=None):
    """Blob estimate, overruled by Tesseract's word-box median when the two
    disagree by more than 1.5x (blob statistics are fooled by specks / dots).
    `info` (dict) receives the word confidence for triage."""
    th = text_height(bw)
    tw, conf = tesseract_word_height(gray, with_conf=True)
    if info is not None:
        info["ocr_conf"] = conf
    if tw is not None and not (th / 1.5 <= tw <= th * 1.5):
        th = float(np.clip(tw, TH_MIN, TH_MAX))
    return th


def estimate_skew(bw, th):
    """Projection-profile deskew: rotate the (speck-free) ink mask and pick the
    angle where the horizontal projection is sharpest. Works with and without
    table lines, and for both rotation directions."""
    clean = remove_specks(bw, max(12, int(0.15 * th * th)))
    small = cv2.resize(clean, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    c = (w / 2, h / 2)

    def score(a):
        M = cv2.getRotationMatrix2D(c, a, 1.0)
        r = cv2.warpAffine(small, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        p = r.sum(axis=1, dtype=np.float64)
        return float(np.var(p))

    coarse = np.arange(-12, 12.01, 0.5)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.5, best + 0.501, 0.05)
    return float(max(fine, key=score))


def rotate(gray, angle):
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    border = int(np.median(gray))
    return cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=border)


def _shear_matrix(shape, deg):
    h, w = shape
    t = np.tan(np.radians(deg))
    return np.float32([[1, -t, t * h / 2], [0, 1, 0]])


def estimate_shear(bw, th):
    """Perspective photos keep leaning columns after rotation (challenge 2).
    Find the horizontal shear that makes the vertical projection sharpest.
    Uses vertical strokes (table lines + digit stems), not specks."""
    clean = remove_specks(bw, max(12, int(0.15 * th * th)))
    vert = cv2.morphologyEx(clean, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(0.6 * th)))))
    small = cv2.resize(vert, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    def score(d):
        r = cv2.warpAffine(small, _shear_matrix(small.shape, d), small.shape[::-1],
                           flags=cv2.INTER_NEAREST, borderValue=0)
        return float(np.var(r.sum(axis=0, dtype=np.float64)))

    coarse = np.arange(-6, 6.01, 0.5)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.5, best + 0.501, 0.1)
    best = float(max(fine, key=score))
    return best if score(best) > 1.05 * score(0.0) else 0.0


def shear(gray, deg):
    border = int(np.median(gray))
    return cv2.warpAffine(gray, _shear_matrix(gray.shape, deg), gray.shape[::-1],
                          flags=cv2.INTER_CUBIC, borderValue=border)


def line_masks(bw, th):
    """Long horizontal / vertical strokes only. Text and specks drop out.
    Must run AFTER deskew (challenge: rotated lines vanish under the opening)."""
    hk = max(25, int(4 * th))
    vk = max(15, int(1.6 * th))
    h = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    # close small gaps in broken (fax) lines
    h = cv2.morphologyEx(h, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (int(th), 1)))
    v = cv2.morphologyEx(v, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(th))))
    return h, v


def _peaks(profile, min_val, merge_dist):
    idx = np.where(profile >= min_val)[0]
    if len(idx) == 0:
        return []
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= merge_dist:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return [int(round(np.average(g, weights=profile[g]))) for g in groups]


def find_lines(hmask, vmask, th):
    """Positions of horizontal (y) and vertical (x) table lines."""
    H, W = hmask.shape
    # tolerate slightly tilted lines: count a column as covered if the line is
    # within +-k rows (a tilted line spreads over several rows otherwise)
    k = max(2, int(0.3 * th))
    hm = cv2.dilate(hmask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2 * k + 1)))
    vm = cv2.dilate(vmask, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 1)))
    ys = _peaks(hm.sum(axis=1) / 255.0, 0.25 * W, th * 0.5)
    vprof = vm.sum(axis=0) / 255.0
    xs = _peaks(vprof, max(2.0 * th, 0.12 * H), th * 0.5)
    # ignore leftovers of the image frame at the very edge
    edge = max(10, int(0.015 * W))
    xs = [x for x in xs if edge < x < W - edge]
    ys = [y for y in ys if max(10, int(0.015 * H)) < y < H - max(10, int(0.015 * H))]
    return ys, xs, vprof


def table_height(bw, ys, th):
    """Vertical extent of the table: outer horizontal rules if there are two,
    otherwise the ink extent (rows with more than speck-level ink)."""
    if len(ys) >= 2:
        return float(ys[-1] - ys[0])
    clean = remove_specks(bw, max(12, int(0.15 * th * th)))
    rows = np.where(clean.sum(axis=1) / 255.0 >= 0.5 * th)[0]
    return float(rows[-1] - rows[0]) if len(rows) else float(bw.shape[0])


MIN_VLINE_SHARE = 0.4


def long_vlines(xs, vprof, height):
    """Router guard: a table border runs (almost) the whole table height.
    Glyph stems lined up in a column of numbers can pass the morphology but
    cover only a small part of it."""
    return [x for x in xs if vprof[x] >= MIN_VLINE_SHARE * height]


def preprocess(path):
    nopre = off("preprocess")             # ablation: no crop / upscale / deskew / shear
    gray = load_gray(path) if nopre else crop_dark_frame(load_gray(path))
    # upscale small text: Tesseract works best around 25-35 px text height
    bw0 = binarize(gray)
    info = {}
    th0 = cross_checked_height(gray, bw0, info)
    # triage (tier guess) from cheap signals of the original image
    from . import triage
    if off("triage"):
        tri, tier = None, None            # ablation: no triage -> full path for every table
    else:
        tri = triage.features(gray, bw0, th0, info.get("ocr_conf"), estimate_skew)
        tier = triage.predict(tri)
    scale = 1.0 if nopre else float(np.clip(30.0 / th0, 1.0, 3.0))
    if scale > 1.05:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    bw = binarize(gray)
    th = text_height(bw)
    expected = float(np.clip(th0 * (scale if scale > 1.05 else 1.0), TH_MIN, TH_MAX))
    if not (expected / 1.5 <= th <= expected * 1.5):
        th = expected          # re-estimate after upscaling disagrees: trust the cross-checked one
    angle = 0.0 if nopre else estimate_skew(bw, th)
    if abs(angle) > 0.1:
        gray = rotate(gray, angle)
        bw = binarize(gray)
    sh = 0.0 if nopre else estimate_shear(bw, th)
    if abs(sh) > 0.15:
        gray = shear(gray, sh)
        bw = binarize(gray)
    hmask, vmask = line_masks(bw, th)
    ys, xs, vprof = find_lines(hmask, vmask, th)
    xs = long_vlines(xs, vprof, table_height(bw, ys, th))
    return {"gray": gray, "bw": bw, "th": th, "angle": angle, "shear": sh, "scale": scale,
            "hmask": hmask, "vmask": vmask, "hlines": ys, "vlines": xs,
            "triage": tri, "tier": tier}
