"""Tesseract wrappers + dash detection.

Challenges: 12 (lines removed before OCR, inset crops), 22/23 (em vs en dash
measured from the ink, since OCR often drops or confuses lone dashes).
"""
import cv2
import numpy as np
import pytesseract

from . import budget
from .ablation import off

LANG = "deu+eng"


def normalize_light(gray, th):
    """Divide by the estimated background so gray stripes, shadows and erased
    lines all become the same white (challenge 3)."""
    if off("preprocess"):
        return gray                       # ablation: no light normalization
    k = max(15, int(1.5 * th)) | 1
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    norm = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    return norm


def _prep(crop, th, target=32):
    """Rescale so text is ~target px tall, normalize light (no hard threshold:
    bold fonts fuse when binarized), pad with white."""
    s = target / max(th, 1)
    if abs(s - 1) > 0.05:
        crop = cv2.resize(crop, None, fx=s, fy=s,
                          interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    norm = normalize_light(crop, target)
    pad = max(15, int(np.ceil(0.5 * target)))      # white margin >= 0.5 x text height
    return cv2.copyMakeBorder(norm, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def ink_mask(gray):
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def _blobs(crop_gray, th):
    bw = cv2.adaptiveThreshold(crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 20)
    H, W = bw.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < max(6, 0.015 * th * th):
            continue  # fax speck
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue  # leftover piece of an erased table line
        out.append((x, y, w, h, a))
    return out


def detect_dash(crop_gray, th):
    """If the only real ink in the crop is one flat bar, return '—' (long) or '–'
    (short). Width relative to text height decides: long dash ~1 em, short ~0.5."""
    blobs = _blobs(crop_gray, th)
    if len(blobs) != 1:
        return None
    x, y, w, h, a = blobs[0]
    if w >= 2.5 * h and h <= 0.35 * th and w >= 0.25 * th:
        return "—" if w >= 0.8 * th else "–"
    return None


def has_ink(crop_gray, th):
    # need at least one blob of character size (fax specks alone = blank)
    return any(h >= 0.35 * th or w >= 0.35 * th for x, y, w, h, a in _blobs(crop_gray, th))


_EDGE_JUNK = "|[]{}\\\"'`‘’“”_"


def clean_text(t):
    t = " ".join(w for w in t.split() if w.strip(_EDGE_JUNK))
    return t.strip(_EDGE_JUNK + " ")


def _tess(img, psm, whitelist=None):
    cfg = f"--psm {psm}"
    if whitelist:
        cfg += f" -c tessedit_char_whitelist={whitelist}"
    d = pytesseract.image_to_data(img, lang=LANG, config=cfg, output_type=pytesseract.Output.DICT)
    words, confs = [], []
    for t, c in zip(d["text"], d["conf"]):
        if t.strip():
            words.append(t.strip())
            confs.append(float(c))
    return clean_text(" ".join(words)), (float(np.mean(confs)) if confs else 0.0)


NUM_WHITELIST = "0123456789.,()-%€kgmsEURt "


def _looks_numeric(t):
    digits = sum(ch.isdigit() for ch in t)
    return digits >= 2 and digits >= 0.5 * len(t.replace(" ", ""))


def despeckle(crop_gray, th):
    """Whiten fax specks that are far from any character (challenge 4, 26).
    Specks next to characters are kept: they could be real commas/periods;
    the value parser + table vote deal with those."""
    if off("preprocess"):
        return crop_gray                  # ablation: no despeckle
    bw = cv2.adaptiveThreshold(crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 20)
    n, labels, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 1:
        return crop_gray
    big = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        big[i] = (h >= 0.35 * th or w >= 0.35 * th) and a >= 0.02 * th * th
    near = cv2.dilate(np.where(big[labels], 255, 0).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.6 * th) | 1, int(0.6 * th) | 1)))
    out = crop_gray.copy()
    for i in range(1, n):
        if big[i]:
            continue
        x, y, w, h, a = st[i]
        blob = labels[y:y + h, x:x + w] == i
        if not (near[y:y + h, x:x + w][blob] > 0).any():
            out[y:y + h, x:x + w][blob] = 255
    return out


def ocr_line(crop_gray, th):
    """OCR one cell. Returns dict(text, conf, alt) where alt is the second
    reader's answer when the two readers disagree (challenge 41-44 cross-check)."""
    if crop_gray.size == 0:
        return {"text": "", "conf": 100.0, "alt": None}
    if not budget.fast():                   # fast path (T0/T1 triage): no fax specks to remove
        crop_gray = despeckle(crop_gray, th)
    if not has_ink(crop_gray, th):
        return {"text": "", "conf": 100.0, "alt": None}
    dash = detect_dash(crop_gray, th)
    if dash:
        return {"text": dash, "conf": 90.0, "alt": None}
    img = _prep(crop_gray, th, 32)
    text, conf = _tess(img, 7)
    late = budget.expired()                 # over the table budget: first read only
    if late:
        budget.skip()
    if (not text or conf < 60) and not late:
        t2, c2 = _tess(img, 13)
        if t2 and (not text or c2 > conf):
            text, conf = t2, c2
    if not late and not _looks_numeric(text) and (conf < 75 or len(text.replace(" ", "")) <= 4):
        # short words / markers (n/a, Apr, Min) are fragile: retry at another scale
        for target, psm in ((42, 7), (42, 8), (26, 7)):
            t4, c4 = _tess(_prep(crop_gray, th, target), psm)
            if t4 and (c4 > conf + 5 or t4.lower() in ("n/a", "na")):
                text, conf = t4, c4
    sign_doubt = False

    def _is_bar(b):
        return b is not None and b[2] >= 0.22 * th and b[2] >= 1.8 * b[3] and b[3] <= 0.25 * th

    if _looks_numeric(text) and (text[:1] in "-−" or text[-1:] in "-−"):
        # challenge 40: a fax speck before/after a number is read as a minus
        # ("-92,004.84" or German-style "92,004.84-"). Accept the minus only if
        # the blob at that end really is a flat bar.
        bl = sorted(_blobs(crop_gray, th), key=lambda b: b[0])
        if text[:1] in "-−" and not _is_bar(bl[0] if bl else None):
            text, sign_doubt = text[1:].lstrip(), True
        if text[-1:] in "-−" and not _is_bar(bl[-1] if bl else None):
            text, sign_doubt = text[:-1].rstrip(), True
    alt = None
    second = not budget.fast() or conf < 60  # fast path: second reader only when unsure
    if _looks_numeric(text) and second and not late and not budget.expired():
        # second, independent read restricted to number characters
        # different scale + restricted alphabet = different failure modes
        t3, c3 = _tess(_prep(crop_gray, th, 42), 7, NUM_WHITELIST.replace(" ", ""))
        if t3 and t3.replace(" ", "") != text.replace(" ", ""):
            alt = {"text": t3, "conf": c3}
    return {"text": text, "conf": conf, "alt": alt, "sign_doubt": sign_doubt}


def ocr_words(gray, th):
    """OCR a whole (deskewed) table image into word boxes (borderless path).
    Returns list of dicts with text, conf, x0, y0, x1, y1 in `gray` coordinates."""
    s = 32 / max(th, 1)
    img = gray
    if abs(s - 1) > 0.1:
        img = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    else:
        s = 1.0                      # not resized -> coordinates must not be rescaled
    # white margin >= 0.5 x text height: glyphs touching the image edge (tight
    # crops) otherwise lose their first/last character
    pad = int(np.ceil(0.5 * th * s))
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    words = []
    seen = set()
    for psm in ("11", "6"):
        d = pytesseract.image_to_data(img, lang=LANG, config=f"--psm {psm}",
                                      output_type=pytesseract.Output.DICT)
        for i, t in enumerate(d["text"]):
            t = t.strip()
            if not t:
                continue
            x0, y0 = (d["left"][i] - pad) / s, (d["top"][i] - pad) / s
            x1, y1 = x0 + d["width"][i] / s, y0 + d["height"][i] / s
            key = (round(x0 / 6), round(y0 / 6), t)
            if key in seen:
                continue
            seen.add(key)
            words.append({"text": t, "conf": float(d["conf"][i]), "x0": x0, "y0": y0,
                          "x1": x1, "y1": y1, "psm": psm})
        if psm == "11":
            break  # psm 6 kept as an option; psm 11 (sparse text) suits tables best
    return words
