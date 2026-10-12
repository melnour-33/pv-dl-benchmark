"""Shared helpers: parse TSLib run-folder names and collect benchmark runs.

run.py writes one folder per (setting, seed) under ./results, named e.g.
dBG_False_long_term_forecast_UK_120_24_PatchTST_UK_ftS_sl120_ll6_pl24_dm512_nh2_el1_dl1_df2048_exd2_dc4_fc3_ebtimeF_dtTrue_Rev_0
"""
import os
import re

MODELS = ["ARIMA", "iTransformer", "Transformer", "TimesNet", "TiDE", "PatchTST",
          "Nonstationary_Transformer", "LightTS", "FiLM", "FEDformer", "DLinear", "Crossformer"]
DATASETS = ["UK", "Miryang", "Germany"]
HORIZONS = [24, 48, 72, 96]

PAT = re.compile(
    r"^dBG_False_long_term_forecast_(?P<data>[A-Za-z]+)_(?P<sl>\d+)_(?P<pl>\d+)_(?P<model>.+?)_(?P=data)"
    r"_ft(?P<feat>MS|S)_sl\d+_ll\d+_pl\d+_(?P<arch>dm\d+_nh\d+_el\d+_dl\d+_df\d+)_.*_(?P<des>[A-Za-z]+)_(?P<seed>\d+)$")


def parse(name):
    m = PAT.match(name)
    if not m:
        return None
    d = m.groupdict()
    return dict(data=d["data"], model=d["model"], features=d["feat"], seq_len=int(d["sl"]),
                pred_len=int(d["pl"]), arch=d["arch"], des=d["des"], seed=int(d["seed"]))


def reference_archs(path):
    """Backbone setting (dm/nh/el/dl/df) of every configuration in a summary CSV, used to
    pick the paper configuration when a results tree holds more than one per model."""
    import csv
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return {(r["data"], r["model"], r["features"], int(r["pred_len"])): r["arch"] for r in csv.DictReader(f)}


def collect(roots, required=("metrics.npy",), des="Rev", seq_len=120, horizons=HORIZONS, archs=None):
    """Return {(data, model, features, pred_len, arch): {seed: folder}}.
    Roots are searched in the given order; the first copy of a run wins.
    archs: optional {(data, model, features, pred_len): arch} restricting each configuration."""
    runs = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            info = parse(name)
            if (info is None or info["des"] != des or info["seq_len"] != seq_len
                    or info["pred_len"] not in horizons or info["model"] not in MODELS):
                continue
            # The job file launches ARIMA with --factor 1 (folder tag _fc1_); an earlier
            # _fc3_ copy from an unpatched job is not part of the benchmark.
            if info["model"] == "ARIMA" and "_fc3_" in name:
                continue
            if archs is not None and archs.get((info["data"], info["model"], info["features"], info["pred_len"])) != info["arch"]:
                continue
            path = os.path.join(root, name)
            if not all(os.path.exists(os.path.join(path, f)) for f in required):
                continue
            key = (info["data"], info["model"], info["features"], info["pred_len"], info["arch"])
            runs.setdefault(key, {}).setdefault(info["seed"], path)
    return runs
