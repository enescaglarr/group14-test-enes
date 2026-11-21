"""Triage: guess the degradation tier (T0-T3) from cheap image signals, so clean
inputs take a fast path and damaged ones the full (slow) path.

Signals (all on the original image, before upscaling; < 200 ms together, the
OCR confidence comes from the word-height cross-check that preprocess already
runs):
  specks        small ink blobs per 10k px (fax noise, bleed-through)
  lap_var       Laplacian variance (sharpness; blur lowers it)
  skew          |deskew angle| on a 0.5x copy
  bg_spread     p95 - p5 of the estimated background (shadows, gradients)
  text_h        blob text height
  ocr_conf      mean Tesseract word confidence of the cross-check pass

The classifier is a depth-4 decision tree stored as plain JSON
(models/triage_tree.json), evaluated without scikit-learn.
"""
import json
from pathlib import Path

import cv2
import numpy as np

FEATURES = ["specks", "lap_var", "skew", "bg_spread", "text_h", "ocr_conf"]
TREE_PATH = Path(__file__).resolve().parent.parent / "models" / "triage_tree.json"
_tree = None


def features(gray, bw, th, ocr_conf, skew_fn):
    h, w = gray.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    areas = st[1:, cv2.CC_STAT_AREA]
    small_area = max(4, int(0.02 * th * th))
    specks = float((areas < small_area).sum()) / (h * w / 1e4)
    s = min(1.0, 800.0 / max(h, w))
    g = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    lap_var = float(cv2.Laplacian(g, cv2.CV_64F).var())
    k = max(15, int(1.5 * th * s)) | 1
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg_spread = float(np.percentile(bg, 95) - np.percentile(bg, 5))
    small_bw = cv2.resize(bw, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    skew = abs(float(skew_fn(small_bw, max(6.0, th * 0.5))))
    return {"specks": specks, "lap_var": lap_var, "skew": skew, "bg_spread": bg_spread,
            "text_h": float(th), "ocr_conf": float(ocr_conf if ocr_conf is not None else 0.0)}


def _load():
    global _tree
    if _tree is None and TREE_PATH.exists():
        _tree = json.loads(TREE_PATH.read_text())
    return _tree


def predict(feat):
    """Tier string, or None when no trained tree is available."""
    t = _load()
    if t is None:
        return None
    node = 0
    while t["left"][node] != -1:
        f = t["feature"][node]
        node = t["left"][node] if feat[f] <= t["threshold"][node] else t["right"][node]
    return t["classes"][int(np.argmax(t["value"][node]))]
