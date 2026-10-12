"""Rank, win, seed-stability, univariate-vs-multivariate, horizon and gap summaries.

usage: python analysis/summary_tables.py [--summary analysis_out/metrics_summary.csv] [--out analysis_out]
"""
import argparse
import os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--summary", default="analysis_out/metrics_summary.csv")
ap.add_argument("--out", default="analysis_out")
a = ap.parse_args()
b = pd.read_csv(a.summary)
D = ["UK", "Miryang", "Germany"]
cell = ["data", "features", "pred_len"]

b["rank"] = b.groupby(cell).MAE_mean.rank()
ranks = b.pivot_table(index="model", columns="data", values="rank", aggfunc="mean")[D]
ranks["all"] = ranks.mean(axis=1)
wins = b[b["rank"] == 1].groupby(["model", "data"]).size().unstack(fill_value=0).reindex(index=ranks.index, columns=D, fill_value=0)
b["cv"] = 100 * b.MAE_std / b.MAE_mean
cv = b[b.model != "ARIMA"].pivot_table(index="model", columns="data", values="cv", aggfunc="mean")[D]
p = b.pivot_table(index=["model", "data", "pred_len"], columns="features", values="MAE_mean")
sms = (100 * (p.MS / p.S - 1)).groupby(["model", "data"]).mean().unstack()[D]
q = b.pivot_table(index=["model", "data", "features"], columns="pred_len", values="MAE_mean")
hz = (100 * (q[96] / q[24] - 1)).groupby(["model", "data"]).mean().unstack()[D]
b["gap"] = 100 * (b.MAE_mean / b.groupby(cell).MAE_mean.transform("min") - 1)
gap = b.pivot_table(index="model", columns="data", values="gap", aggfunc="mean")[D]

os.makedirs(a.out, exist_ok=True)
for name, t in [("avg_rank", ranks), ("wins", wins), ("seed_cv_pct", cv), ("ms_vs_s_pct", sms),
                ("h96_vs_h24_pct", hz), ("gap_to_best_pct", gap)]:
    t.round(4).to_csv(os.path.join(a.out, name + ".csv"))
    print(f"\n== {name}\n{t.round(2)}")
