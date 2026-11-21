#!/usr/bin/env python3
"""Score the ablation runs (tools/ablation_run.py) per version x set.

    python tools/ablation_score.py <abl_dir> <out.json> [--ocb-res <dir>]

<abl_dir>/<V>/<set>/ holds predictions + _runs.json for V in V1..V6 and set in
dev, final (TableSalt: FINAL via score_all.score_team, tier-weighted TEDS, QA),
ftn, ftnd (FinTabNet clean / degraded, data/ext: mean TEDS, TEDS-S of evaluate.py;
no tiers, so no FINAL) and ocb (OCRBench v2 table parsing en: official eval.py TEDS,
read from <ocb-res>/res_<V>.json, and TEDS-S from <ocb-res>/teds_s_<V>.json).
Seconds per table come from _runs.json (wall time of one table, sequential runs).
"""
import argparse
import importlib.util
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GT = {"dev": REPO / "data/dev", "final": REPO / "data/final",
      "ftn": REPO / "data/ext/fintabnet", "ftnd": REPO / "data/ext/fintabnet_degraded"}


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def times(d):
    runs = {}
    for f in sorted(d.glob("_runs*.json")):            # _runs.json or parallel shards _runs.sI.json
        runs.update(json.loads(f.read_text()))
    s = [r["s"] for r in runs.values()]
    bad = sum(r["status"] != "ok" for r in runs.values())
    return {"n": len(runs), "mean_s": round(st.mean(s), 2), "median_s": round(st.median(s), 2),
            "max_s": round(max(s), 1), "over_15s": sum(x > 15 for x in s), "failed": bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("abl")
    ap.add_argument("out")
    ap.add_argument("--ocb-res", default=None)
    a = ap.parse_args()
    sa, ev = load("score_all"), load("evaluate")
    abl = Path(a.abl)
    res = {}
    for vd in sorted(p for p in abl.iterdir() if p.is_dir() and p.name[:1] == "V" and p.name[1:].isdigit()):
        v = vd.name
        for s in ("dev", "final", "ftn", "ftnd", "ocb"):
            d = vd / s
            if not any(d.glob("_runs*.json")):
                continue
            r = times(d)
            if s in ("dev", "final"):
                t = sa.score_team(v, GT[s], d)
                r.update(final=t["final"], teds_w=t["teds_weighted"], qa=t["qa_accuracy"],
                         cell_f1=t["cell_f1"], per_tier=t["per_tier"])
            elif s in ("ftn", "ftnd"):
                man = json.loads((GT[s] / "manifest.json").read_text(encoding="utf-8"))
                tt, ts = [], []
                for e in man:
                    g = (GT[s] / e["id"] / "gt" / "table.html").read_text(encoding="utf-8")
                    p = ev.load_prediction(d, e["id"])
                    tt.append(ev.teds(g, p) if p else 0.0)
                    ts.append(ev.teds(g, p, True) if p else 0.0)
                r.update(teds=round(st.mean(tt), 4), teds_s=round(st.mean(ts), 4))
            elif a.ocb_res:
                rf = Path(a.ocb_res) / f"res_{v}.json"
                sf = Path(a.ocb_res) / f"teds_s_{v}.json"
                if rf.exists():
                    items = [x for x in json.loads(rf.read_text(encoding="utf-8"))
                             if x["type"] == "table parsing en"]
                    r["teds"] = round(st.mean(x["score"] for x in items), 4)
                if sf.exists():
                    r["teds_s"] = round(st.mean(json.loads(sf.read_text()).values()), 4)
            res.setdefault(v, {})[s] = r
            print(v, s, {k: x for k, x in r.items() if k != "per_tier"}, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
