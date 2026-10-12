"""Daylight-only and regime-resolved scores for every benchmark run.

Predictions and targets are shifted by the night level of the target. A test hour is
night when the target is below 5% of its range. Daylight nMAE is the daylight MAE divided by
the mean daylight output. Stable / variable windows are the lowest / highest tercile of the
daylight standard deviation of the ground truth; ramp windows are the top 10% by largest
hourly change. The definitions are those of analysis/evaluate_daylight_regime.py.

usage: python analysis/daylight_regime.py [--roots results] [--horizons 24 48 72 96] [--out analysis_out]
"""
import argparse
import os
import numpy as np
import pandas as pd
from common import collect, reference_archs
from evaluate_daylight_regime import evaluate_run

ap = argparse.ArgumentParser()
ap.add_argument("--roots", nargs="+", default=["results"])
ap.add_argument("--horizons", nargs="+", type=int, default=[24, 48, 72, 96])
ap.add_argument("--out", default="analysis_out")
ap.add_argument("--reference", default="results/paper_metrics_summary.csv",
                help="CSV fixing the backbone setting of each configuration (use none to disable)")
a = ap.parse_args()
archs = None if a.reference == "none" else reference_archs(a.reference)
rows = []
for (data, model, feat, pl, arch), seeds in sorted(collect(a.roots, required=("pred.npy", "true.npy"), horizons=a.horizons, archs=archs).items()):
    if model == "ARIMA":
        continue
    res = []
    for s, path in sorted(seeds.items()):
        p, t = np.load(os.path.join(path, "pred.npy")), np.load(os.path.join(path, "true.npy"))
        lo = float(np.nanmin(t[:, :, 0]))
        r = evaluate_run(p - lo, t - lo, 0, 0.05, 0.05)
        r["all_MAE"] = float(np.mean(np.abs(p - t)))
        res.append(r)
    row = dict(data=data, model=model, features=feat, pred_len=pl, n=len(res))
    for k in res[0]:
        v = np.array([x[k] for x in res], float)
        row[k + "_mean"] = np.nanmean(v)
        row[k + "_std"] = np.nanstd(v, ddof=1) if len(v) > 1 else 0.0
    rows.append(row)
os.makedirs(a.out, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(a.out, "daylight_regime.csv"), index=False)
print(f"{len(rows)} configurations -> {a.out}/daylight_regime.csv")
