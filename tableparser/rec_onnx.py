"""Third reader: PaddleOCR PP-OCRv5 latin text RECOGNITION model (ONNX, CPU).

Recognition only (no detection): it reads one cell crop. Weights live in
models/paddleocr_rec_latin/ (Apache-2.0, see SOURCE_README.md there); no
network at run time. Loaded lazily on first use; if onnxruntime or the model
is missing, available() is False and the pipeline runs without it.
"""
import os
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(os.environ.get("TABLEPARSER_MODELS",
                                Path(__file__).resolve().parent.parent / "models")) / "paddleocr_rec_latin"
REC_H = 48            # the exported graph expects height 48
MAX_W = 960

_sess = None
_chars = None
_failed = False


def _load():
    global _sess, _chars, _failed
    if _sess is not None or _failed:
        return _sess is not None
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.log_severity_level = 3
        _sess = ort.InferenceSession(str(MODEL_DIR / "rec.onnx"), sess_options=so,
                                     providers=["CPUExecutionProvider"])
        lines = (MODEL_DIR / "dict.txt").read_text(encoding="utf-8").splitlines()
        _chars = ["<blank>"] + [l for l in lines] + [" "]
        return True
    except Exception:
        _failed = True
        return False


def available():
    return _load()


def _prep(gray):
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return None
    nw = int(min(MAX_W, max(8, round(w * REC_H / h))))
    img = cv2.resize(gray, (nw, REC_H), interpolation=cv2.INTER_CUBIC if h < REC_H else cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return img.transpose(2, 0, 1)[None]


def read(gray):
    """Text and mean character confidence (0-100) of one cell crop."""
    if not _load():
        return "", 0.0
    x = _prep(gray)
    if x is None:
        return "", 0.0
    probs = _sess.run(None, {_sess.get_inputs()[0].name: x})[0][0]   # (T, C)
    idx = probs.argmax(axis=1)
    out, confs, prev = [], [], -1
    for t, k in enumerate(idx):
        if k != prev and k != 0 and k < len(_chars):
            out.append(_chars[k])
            confs.append(float(probs[t, k]))
        prev = k
    text = "".join(out).strip()
    return text, (100.0 * float(np.mean(confs)) if confs else 0.0)
