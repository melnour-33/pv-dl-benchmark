"""Diebold-Mariano test of every model against the best model of each cell.

Loss differential: per-window MAE averaged over the five seeds. Long-run variance:
Bartlett kernel with H-1 lags. Small-sample correction: Harvey, Leybourne and Newbold (1997).
p-values are Holm-adjusted over the comparisons within each cell.

usage: python analysis/dm_tests.py [--roots results] [--horizons 24] [--out analysis_out]
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats
from common import collect, reference_archs


def dm_test(d, h):
    d = np.asarray(d, float); n = d.size
    dc = d - d.mean()
    lr = np.dot(dc, dc) / n
    for k in range(1, min(h, n - 1)):
        lr += 2.0 * (1.0 - k / h) * np.dot(dc[k:], dc[:-k]) / n
    stat = d.mean() / np.sqrt(lr / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2.0 * stats.t.sf(abs(stat), df=n - 1))


def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j])); adj[j] = run
    return adj


ap = argparse.ArgumentParser()
ap.add_argument("--roots", nargs="+", default=["results"])
ap.add_argument("--horizons", nargs="+", type=int, default=[24])
ap.add_argument("--out", default="analysis_out")
ap.add_argument("--reference", default="results/paper_metrics_summary.csv",
                help="CSV fixing the backbone setting of each configuration (use none to disable)")
a = ap.parse_args()
archs = None if a.reference == "none" else reference_archs(a.reference)
runs = collect(a.roots, required=("pred.npy", "true.npy"), horizons=a.horizons, archs=archs)
perwin, cells = {}, {}
for (data, model, feat, pl, arch), seeds in runs.items():
    if model == "ARIMA":
        continue
    errs = []
    for s, path in sorted(seeds.items()):
        p, t = np.load(os.path.join(path, "pred.npy")), np.load(os.path.join(path, "true.npy"))
        e = np.abs(p - t); errs.append(e.reshape(len(e), -1).mean(1))
    perwin[(data, model, feat, pl)] = np.mean(errs, 0)
    cells.setdefault((data, feat, pl), []).append(model)
rows = []
for (data, feat, pl), models in sorted(cells.items()):
    best = min(models, key=lambda m: perwin[(data, m, feat, pl)].mean())
    res = []
    for m in models:
        if m == best:
            continue
        d = perwin[(data, m, feat, pl)] - perwin[(data, best, feat, pl)]
        st, pv = dm_test(d, pl)
        res.append(dict(data=data, features=feat, pred_len=pl, best=best, model=m,
                        mae_best=perwin[(data, best, feat, pl)].mean(), mae=perwin[(data, m, feat, pl)].mean(),
                        gap_pct=100 * d.mean() / perwin[(data, best, feat, pl)].mean(), dm=st, p=pv))
    for r, h in zip(res, holm([r["p"] for r in res])):
        r["p_holm"] = h
    rows += res
os.makedirs(a.out, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(a.out, "dm_vs_best.csv"), index=False)
print(f"{len(rows)} comparisons -> {a.out}/dm_vs_best.csv")
