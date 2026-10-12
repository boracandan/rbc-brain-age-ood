"""
Fills best_val_r and best_val_mae (validation Pearson r and MAE in years of the best-loss checkpoint) in each
LOO split's results_table.csv, from the saved best_val_preds.npy / best_val_true.npy of every trial.
Only rows with a missing value are filled; existing values are never overwritten.

Run it once all training runs have finished (it rewrites the results tables):
    python helper_scripts/compute_val_metrics.py
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

dir_path = os.environ.get("RBC_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOO_SPLITS = [
    "LOO_BHRC",
    "LOO_HBN",
    "LOO_CCNP",
    "LOO_NKI",
    "LOO_PNC",
]
MAX_AGE = 22  # ages were divided by this for age_norm == "minmax_0_1"


def val_metrics(experiment, trial_id):
    """(val r, val MAE in years) of a trial, or None if its saved files are not available."""
    trial_dir = os.path.join(dir_path, experiment, trial_id)
    files = [os.path.join(trial_dir, f) for f in ("best_val_preds.npy", "best_val_true.npy", "config.json")]
    if not all(os.path.exists(f) for f in files):
        return None
    preds, true = np.load(files[0]), np.load(files[1])
    with open(files[2], encoding="utf-8-sig") as f:
        cfg = json.load(f)
    if cfg.get("age_norm") == "minmax_0_1":
        preds, true = preds * MAX_AGE, true * MAX_AGE
    return round(float(pearsonr(true, preds)[0]), 3), round(float(np.mean(np.abs(preds - true))), 3)


def main():
    for experiment in LOO_SPLITS:
        path = os.path.join(dir_path, experiment, "results_table.csv")
        if not os.path.exists(path):
            print(f"Skipping {experiment} — no results_table.csv found")
            continue
        df = pd.read_csv(path)
        for col in ("best_val_r", "best_val_mae"):
            if col not in df.columns:
                df[col] = np.nan

        missing = df["best_val_r"].isna() | df["best_val_mae"].isna()
        if not missing.any():
            print(f"{experiment}: no missing best_val_r/mae — nothing to do")
            continue
        filled = 0
        for i in df.index[missing]:
            m = val_metrics(experiment, df.at[i, "trial_id"])
            if m is not None:
                df.at[i, "best_val_r"], df.at[i, "best_val_mae"] = m
                filled += 1
        df.to_csv(path, index=False)
        print(f"{experiment}: filled {filled} of {int(missing.sum())} missing rows -> {path}")


if __name__ == "__main__":
    main()
