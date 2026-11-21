"""Ablation switches (measurement only; the default is the full system).

    TABLEPARSER_ABLATE=preprocess,triage,third python ...

    preprocess  no image preprocessing: no dark-frame crop, no upscaling, no deskew,
                no shear correction, no light normalization, no despeckle
                (text height and line masks are still measured: the structure needs them)
    triage      no tier triage: every table takes the full path (12 s budget)
    third       no third reader (ONNX recognizer) on suspicious number cells

Unset or empty -> nothing is switched off, behaviour is unchanged.
"""
import os

FLAGS = frozenset(f.strip() for f in os.environ.get("TABLEPARSER_ABLATE", "").split(",") if f.strip())
_KNOWN = {"preprocess", "triage", "third"}
if FLAGS - _KNOWN:
    raise ValueError(f"unknown TABLEPARSER_ABLATE flags: {sorted(FLAGS - _KNOWN)}")


def off(name: str) -> bool:
    return name in FLAGS
