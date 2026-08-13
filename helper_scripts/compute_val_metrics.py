"""
Computes best_val_r and best_val_mae for each trial from saved .npy predictions
and writes them back into each LOO split's results_table.csv.

Run from the project root:
    python helper_scripts/compute_val_metrics.py
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

dir_path = os.environ.get("RBC_DIR", r"C:\Users\Faruk\Code\rbc-brain-age-ood")

LOO_SPLITS = [
    "LOO_BHRC",
    "LOO_HBN",
    "LOO_CCNP",
    "LOO_NKI",
    "LOO_PNC",
]

for experiment in LOO_SPLITS:
    exp_results_path = os.path.join(dir_path, experiment, "results_table.csv")
    if not os.path.exists(exp_results_path):
        print(f"Skipping {experiment} — no results_table.csv found")
        continue

    df = pd.read_csv(exp_results_path)

    if all(x in df.columns for x in ["best_val_r", "best_val_mae"]):
        print(f"{experiment}: best_val_r/mae already present — skipping")
        continue

    rs, maes = [], []
    for trial_id in df["trial_id"]:
        trial_dir  = os.path.join(dir_path, experiment, trial_id)
        preds_file = os.path.join(trial_dir, "best_val_preds.npy")
        true_file  = os.path.join(trial_dir, "best_val_true.npy")
        cfg_file   = os.path.join(trial_dir, "config.json")

        if os.path.exists(preds_file) and os.path.exists(true_file) and os.path.exists(cfg_file):
            preds = np.load(preds_file)
            true  = np.load(true_file)
            with open(cfg_file, encoding="utf-8-sig") as f:
                cfg = json.load(f)
            if cfg.get("age_norm") == "minmax_0_1":
                max_age = 22
                preds, true = preds * max_age, true * max_age
            rs.append(round(float(pearsonr(true, preds)[0]), 3))
            maes.append(round(float(np.mean(np.abs(preds - true))), 3))
        else:
            rs.append(None)
            maes.append(None)

    df["best_val_r"]   = rs
    df["best_val_mae"] = maes
    df.to_csv(exp_results_path, index=False)

    n_ok = sum(r is not None for r in rs)
    print(f"{experiment}: wrote best_val_r/mae for {n_ok}/{len(df)} trials -> {exp_results_path}")
