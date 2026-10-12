"""Collect the per-seed test metrics of every benchmark run.

usage: python analysis/collect_metrics.py [--roots results] [--out analysis_out]
Writes metrics_per_seed.csv and metrics_summary.csv (mean, std and raw seed values).
Metrics are computed by run.py on the z-normalized series: MAE, MSE, RMSE, MAPE, MSPE.
"""
import argparse
import os
import numpy as np
import pandas as pd
from common import collect, reference_archs

ap = argparse.ArgumentParser()
ap.add_argument("--roots", nargs="+", default=["results"])
ap.add_argument("--out", default="analysis_out")
ap.add_argument("--reference", default="results/paper_metrics_summary.csv",
                help="CSV fixing the backbone setting of each configuration (use none to disable)")
a = ap.parse_args()
archs = None if a.reference == "none" else reference_archs(a.reference)
os.makedirs(a.out, exist_ok=True)

rows = []
for (data, model, feat, pl, arch), seeds in collect(a.roots, archs=archs).items():
    for s, path in sorted(seeds.items()):
        m = np.load(os.path.join(path, "metrics.npy"))
        rows.append(dict(data=data, model=model, features=feat, pred_len=pl, arch=arch, seed=s,
                         MAE=float(m[0]), MSE=float(m[1]), RMSE=float(m[2]), MAPE=float(m[3]), MSPE=float(m[4])))
seed = pd.DataFrame(rows).sort_values(["data", "model", "features", "pred_len", "seed"])
seed.to_csv(os.path.join(a.out, "metrics_per_seed.csv"), index=False)

out = []
for key, g in seed.groupby(["data", "model", "features", "pred_len", "arch"]):
    r = dict(zip(["data", "model", "features", "pred_len", "arch"], key), n_seeds=len(g))
    for k in ["MAE", "MSE", "RMSE", "MAPE", "MSPE"]:
        v = g[k].values
        r[k + "_mean"] = v.mean()
        r[k + "_std"] = v.std(ddof=1) if len(v) > 1 else 0.0
        r[k + "_raw"] = "|".join(f"{x:.6f}" for x in v)
    out.append(r)
pd.DataFrame(out).to_csv(os.path.join(a.out, "metrics_summary.csv"), index=False)
print(f"{len(seed)} runs in {len(out)} configurations -> {a.out}/metrics_summary.csv")
