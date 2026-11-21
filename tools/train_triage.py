#!/usr/bin/env python3
"""Train the tier triage classifier (tableparser/triage.py).

    python tools/train_triage.py --train data/triage --test data/gen data/holdout \
        --out models/triage_tree.json --report triage_report.json

Features come from exactly the runtime code path (preprocess helpers +
triage.features). Compares a depth-4 decision tree with a logistic regression
(5-fold CV on --train), saves the decision tree as JSON (no scikit-learn at
run time), and reports accuracy, the confusion matrix and the share of T3
images predicted as T0 (the costly mistake: a fax would take the fast path).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tableparser import preprocess as P  # noqa: E402
from tableparser import triage  # noqa: E402

TIERS = ["T0", "T1", "T2", "T3"]


def extract(ds):
    man = json.loads((Path(ds) / "manifest.json").read_text())
    X, y, ms = [], [], []
    for e in man:
        gray = P.crop_dark_frame(P.load_gray(Path(ds) / e["image"]))
        bw0 = P.binarize(gray)
        t0 = time.perf_counter()
        info = {}
        th0 = P.cross_checked_height(gray, bw0, info)
        t1 = time.perf_counter()
        f = triage.features(gray, bw0, th0, info.get("ocr_conf"), P.estimate_skew)
        t2 = time.perf_counter()
        X.append([f[k] for k in triage.FEATURES])
        y.append(e["tier"])
        ms.append(((t1 - t0) * 1000, (t2 - t1) * 1000))
    return np.array(X), np.array(y), np.array(ms)


def confusion(y, p):
    m = {a: {b: 0 for b in TIERS} for a in TIERS}
    for a, b in zip(y, p):
        m[a][b] += 1
    return m


def summary(y, p):
    m = confusion(y, p)
    n3 = sum(m["T3"].values())
    return {"accuracy": round(float(np.mean(np.array(y) == np.array(p))), 4),
            "t3_as_t0": round(m["T3"]["T0"] / n3, 4) if n3 else None,
            "t3_or_t2_as_fast": round(sum(m[a][b] for a in ("T2", "T3") for b in ("T0", "T1"))
                                      / max(1, sum(sum(m[a].values()) for a in ("T2", "T3"))), 4),
            "confusion(rows=true)": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--out", default=str(ROOT / "models" / "triage_tree.json"))
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    X, y, ms = extract(args.train)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    tree = DecisionTreeClassifier(max_depth=4, random_state=0, class_weight="balanced")
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    rep = {"n_train": len(y), "features": triage.FEATURES,
           "ms_per_image": {"ocr_crosscheck_mean": round(float(ms[:, 0].mean()), 1),
                            "ocr_crosscheck_max": round(float(ms[:, 0].max()), 1),
                            "other_signals_mean": round(float(ms[:, 1].mean()), 1),
                            "other_signals_max": round(float(ms[:, 1].max()), 1)},
           "cv": {"tree_depth4": summary(y, cross_val_predict(tree, X, y, cv=cv)),
                  "logreg": summary(y, cross_val_predict(logreg, X, y, cv=cv))}}
    tree.fit(X, y)
    t = tree.tree_
    model = {"type": "decision_tree", "max_depth": 4, "features": triage.FEATURES,
             "classes": [str(c) for c in tree.classes_],
             "feature": [triage.FEATURES[i] if i >= 0 else None for i in t.feature],
             "threshold": [float(v) for v in t.threshold],
             "left": [int(v) for v in t.children_left], "right": [int(v) for v in t.children_right],
             "value": [[float(x) for x in v[0]] for v in t.value],
             "trained_on": f"{args.train} (n={len(y)})"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(model, indent=1))
    triage._tree = model
    for ds in args.test:
        Xt, yt, _ = extract(ds)
        pt = [triage.predict(dict(zip(triage.FEATURES, row))) for row in Xt]
        rep[f"test:{ds}"] = summary(yt, pt)
    print(json.dumps({k: v for k, v in rep.items()}, indent=1))
    if args.report:
        Path(args.report).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
