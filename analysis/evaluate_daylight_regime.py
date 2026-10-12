#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_daylight_regime.py
====================================================================
Re-score saved predictions with DAYLIGHT-ONLY metrics and split them by
weather / variability REGIME. No retraining required -- it reads the
pred.npy / true.npy that run.py already saved in ./results/<setting>/.

Addresses:
  * Reviewer 3 -- "Solar power is zero at night, which makes MAPE/MSPE
    undefined or unstable ... daylight-only metrics should be reported ...
    small-denominator clipping." Also "splitting metrics by clear-sky /
    variable-sky regimes would better isolate where the method helps."
  * Reviewer 2 -- "the analysis does not directly examine ramp events,
    cloudy/clear regimes, or high-variability periods."

WHAT IT COMPUTES (per run, then averaged across the 5 seeds)
  * night_fraction          : share of target points treated as night.
  * overall_RMSE            : RMSE on all points (matches the paper tables).
  * day_MAE / day_RMSE      : daylight-only errors.
  * day_nMAE% / day_nRMSE%  : daylight errors normalised by the daylight mean
                              of the ground truth (a stable, well-defined
                              stand-in for MAPE that does not blow up at night).
  * day_MAPE_clip%          : daylight MAPE with the denominator floored at
                              clip_frac * max(daylight truth)  (small-denominator
                              clipping, exactly as the reviewer requests).
  * clear_RMSE / var_RMSE   : daylight RMSE on the least- vs most-variable
                              forecast windows (clear-sky vs cloudy/variable
                              regime, split by per-window ground-truth std).
  * ramp_RMSE               : daylight RMSE on the top-10% ramp-event windows
                              (largest max |Δ power| within the window).

NOTES
  * Predictions are stored on the training scale (StandardScaler) unless the
    run used --inverse. Absolute (MAE/RMSE) and normalised (%-style) numbers are
    still directly comparable BETWEEN baseline and dBG for the same dataset,
    which is what the reviewers care about. If you re-run with --inverse the raw
    numbers become physical kW.
  * Night is detected from the target itself (PV -> its per-dataset minimum at
    night). Tune with --night_frac if a dataset has a noisy floor.

USAGE
  python scripts/revision/evaluate_daylight_regime.py --results_dir ./results
  python scripts/revision/evaluate_daylight_regime.py --results_dir ./results \
         --tag Rev --pv_channel 0 --night_frac 0.05
====================================================================
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np

RE_CORE = re.compile(
    r".*_([A-Za-z0-9]+)_([A-Za-z0-9]+)_ft(S|M|MS)_sl(\d+)_ll(\d+)_pl(\d+)_"
)
RE_DBGFLAG = re.compile(r"dBG_(True|False)_")
RE_DES_II = re.compile(r"_dt(?:True|False)_(.+)_(\d+)$")


def parse_setting(name):
    core = RE_CORE.search(name)
    if not core:
        return None
    model, data, feats, sl, ll, pl = core.groups()
    dbgflag = RE_DBGFLAG.search(name)
    use_dbg = (dbgflag.group(1) == "True") if dbgflag else False
    des_ii = RE_DES_II.search(name)
    des, ii = (des_ii.group(1), int(des_ii.group(2))) if des_ii else ("NA", 0)
    return dict(model=model, data=data, features=feats, seq_len=int(sl),
                pred_len=int(pl), dBG=use_dbg, des=des, seed_idx=ii)


def group_key(name):
    return re.sub(r"_(\d+)$", "", name)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else float("nan")


def mae(a, b):
    return float(np.mean(np.abs(a - b))) if a.size else float("nan")


def evaluate_run(pred, true, pv_channel, night_frac, clip_frac,
                 ramp_top=0.10):
    """pred/true: [N, P, D]. Returns dict of metrics for the PV channel."""
    c = pv_channel if pv_channel < pred.shape[-1] else 0
    p = pred[:, :, c].astype(float)      # [N, P]
    t = true[:, :, c].astype(float)      # [N, P]

    # ---- night / day mask from the target floor -----------------------------
    lo, hi = np.nanmin(t), np.nanmax(t)
    rng = (hi - lo) if hi > lo else 1.0
    thr = lo + night_frac * rng
    day = t > thr                        # boolean [N, P]
    night_fraction = float(1.0 - day.mean())

    out = dict(night_fraction=night_fraction, overall_RMSE=rmse(p, t))

    pd_day, td_day = p[day], t[day]
    out["day_MAE"] = mae(pd_day, td_day)
    out["day_RMSE"] = rmse(pd_day, td_day)

    day_mean = td_day.mean() if td_day.size else float("nan")
    out["day_nMAE_pct"] = 100.0 * out["day_MAE"] / day_mean if day_mean else float("nan")
    out["day_nRMSE_pct"] = 100.0 * out["day_RMSE"] / day_mean if day_mean else float("nan")

    # daylight MAPE with small-denominator clipping
    floor = max(clip_frac * hi, 1e-6)
    denom = np.maximum(td_day, floor)
    out["day_MAPE_clip_pct"] = float(100.0 * np.mean(np.abs((td_day - pd_day) / denom))) \
        if td_day.size else float("nan")

    # ---- regime split by per-window variability (daylight only) -------------
    # per-window std and max-ramp of the ground truth over daylight steps
    win_std = np.full(t.shape[0], np.nan)
    win_ramp = np.full(t.shape[0], np.nan)
    for i in range(t.shape[0]):
        m = day[i]
        if m.sum() >= 2:
            tv = t[i, m]
            win_std[i] = tv.std()
            win_ramp[i] = np.max(np.abs(np.diff(t[i])))  # ramp over full window
    valid = ~np.isnan(win_std)
    if valid.sum() >= 3:
        s = win_std[valid]
        idx = np.where(valid)[0]
        q1, q2 = np.quantile(s, [1 / 3, 2 / 3])
        clear = idx[s <= q1]
        var = idx[s >= q2]
        out["clear_RMSE"] = rmse(p[clear][day[clear]], t[clear][day[clear]])
        out["var_RMSE"] = rmse(p[var][day[var]], t[var][day[var]])
    else:
        out["clear_RMSE"] = float("nan")
        out["var_RMSE"] = float("nan")

    validr = ~np.isnan(win_ramp)
    if validr.sum() >= 3:
        r = win_ramp[validr]
        idxr = np.where(validr)[0]
        thr_r = np.quantile(r, 1 - ramp_top)
        ramp_idx = idxr[r >= thr_r]
        out["ramp_RMSE"] = rmse(p[ramp_idx][day[ramp_idx]], t[ramp_idx][day[ramp_idx]])
    else:
        out["ramp_RMSE"] = float("nan")
    return out


METRIC_ORDER = ["night_fraction", "overall_RMSE", "day_MAE", "day_RMSE",
                "day_nMAE_pct", "day_nRMSE_pct", "day_MAPE_clip_pct",
                "clear_RMSE", "var_RMSE", "ramp_RMSE"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="./results")
    ap.add_argument("--out_dir", default="./revision_outputs")
    ap.add_argument("--tag", default=None,
                    help="only runs whose --des starts with this (e.g. 'Rev')")
    ap.add_argument("--pv_channel", type=int, default=0,
                    help="index of the PV-power channel in pred/true "
                         "(0 for univariate S runs; verify for multivariate)")
    ap.add_argument("--night_frac", type=float, default=0.05,
                    help="fraction of the target range used as the night floor")
    ap.add_argument("--clip_frac", type=float, default=0.05,
                    help="denominator floor for clipped MAPE = clip_frac*max")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    groups = defaultdict(lambda: dict(info=None, runs=[]))
    n_found = 0
    for pred_path in glob.glob(os.path.join(args.results_dir, "*", "pred.npy")):
        folder = os.path.basename(os.path.dirname(pred_path))
        info = parse_setting(folder)
        if info is None:
            continue
        if args.tag is not None and not info["des"].startswith(args.tag):
            continue
        true_path = os.path.join(os.path.dirname(pred_path), "true.npy")
        if not os.path.exists(true_path):
            continue
        try:
            pred = np.load(pred_path); true = np.load(true_path)
        except Exception as e:
            print(f"  ! could not load {folder}: {e}")
            continue
        res = evaluate_run(pred, true, args.pv_channel, args.night_frac,
                           args.clip_frac)
        gkey = group_key(folder)
        g = groups[gkey]
        g["info"] = info
        g["runs"].append(res)
        n_found += 1

    if not groups:
        print("No pred.npy/true.npy found. Check --results_dir / --tag.")
        return
    print(f"Scored {n_found} runs in {len(groups)} configurations.")

    # average across seeds
    rows = []
    for gkey, g in sorted(groups.items()):
        info = g["info"]
        agg = {}
        for mname in METRIC_ORDER:
            vals = np.array([r[mname] for r in g["runs"]], float)
            vals = vals[~np.isnan(vals)]
            agg[mname] = (float(vals.mean()), float(vals.std(ddof=1)) if vals.size > 1 else 0.0)
        rows.append((info, agg, len(g["runs"])))

    cols = ["data", "model", "features", "pred_len", "dBG", "des", "n"] + METRIC_ORDER
    csv_path = os.path.join(args.out_dir, "daylight_regime_metrics.csv")
    md_path = os.path.join(args.out_dir, "daylight_regime_metrics.md")
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for info, agg, n in rows:
            vals = [info["data"], info["model"], info["features"], info["pred_len"],
                    info["dBG"], info["des"], n]
            vals += [f"{agg[m][0]:.4f}" for m in METRIC_ORDER]
            f.write(",".join(str(v) for v in vals) + "\n")
    with open(md_path, "w") as f:
        f.write("# Daylight-only and regime-split metrics (mean over seeds)\n\n")
        f.write("`day_*` = nighttime excluded. `*_nMAE/nRMSE_pct` normalise by the "
                "daylight mean of the ground truth. `day_MAPE_clip_pct` uses "
                "small-denominator clipping. `clear/var_RMSE` = least/most variable "
                "windows; `ramp_RMSE` = top-10% ramp-event windows.\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for info, agg, n in rows:
            vals = [info["data"], info["model"], info["features"], info["pred_len"],
                    info["dBG"], info["des"], n]
            vals += [f"{agg[m][0]:.4f}" for m in METRIC_ORDER]
            f.write("| " + " | ".join(str(v) for v in vals) + " |\n")
    print(f"  wrote {csv_path}")
    print(f"  wrote {md_path}")
    print("Done.")


if __name__ == "__main__":
    main()
