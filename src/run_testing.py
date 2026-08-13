"""
Runs Testing_multimodality.py for the top-5 trials per architecture group
per LOO split, ranked by best_state_val_loss.

Usage:
    python run_testing.py
"""

import pandas as pd
import subprocess
import os
import re
import sys

# ── Config ────────────────────────────────────────────────────────────────────
_SRC       = os.path.dirname(os.path.abspath(__file__))
DIR        = os.path.dirname(_SRC)
TESTING_PY = os.path.join(_SRC, "Testing_multimodality.py")
PYTHON     = sys.executable
TOP_K      = 5

LOO_SPLITS = {
    "LOO_BHRC": "BHRC",
    "LOO_CCNP": "CCNP",
    "LOO_HBN":  "HBN",
    "LOO_NKI":  "NKI",
    "LOO_PNC":  "PNC",
}

GROUPS = [
    "GCN_fMRI",
    "GCN_fMRI_nodeg",
    "GCN_fMRI&sMRI",
    "GCN_fMRI&sMRI_nodeg",
    "GAT_fMRI",
    "GAT_fMRI&sMRI",
]
# ─────────────────────────────────────────────────────────────────────────────


def arch_group(row):
    m = re.search(r'_(gcn|gat)_(fmri|multi)$', row["trial_id"])
    if m is None:
        return "unknown"
    arch = m.group(1).upper()
    mod  = "fMRI&sMRI" if m.group(2) == "multi" else "fMRI"
    deg  = str(row["degree_norm"]).strip().lower() in ("true", "1")
    return f"{arch}_{mod}" + ("" if deg or arch == "GAT" else "_nodeg")


def already_tested(split, trial_id):
    results_csv = os.path.join(DIR, split, "test_results.csv")
    if not os.path.exists(results_csv):
        return False
    df = pd.read_csv(results_csv)
    return trial_id in df["trial_id"].values


def get_top_trials(split):
    df = pd.read_csv(os.path.join(DIR, split, "results_table.csv"))
    df["arch_group"] = df.apply(arch_group, axis=1)
    trials = []
    for group in GROUPS:
        sub = df[df["arch_group"] == group].nsmallest(TOP_K, "best_state_val_loss")
        for tid in sub["trial_id"]:
            trials.append((tid, group))
    return trials


# ── Main ──────────────────────────────────────────────────────────────────────
total = ok = failed = skipped = 0

for split in LOO_SPLITS:
    print(f"\n{'='*65}")
    print(f"  {split}")
    print(f"{'='*65}")

    trials = get_top_trials(split)
    total += len(trials)

    for tid, group in trials:
        if already_tested(split, tid):
            print(f"  [skip] {tid}  ({group})")
            skipped += 1
            continue

        print(f"\n  [run]  {tid}  ({group})")
        cmd = [PYTHON, TESTING_PY, tid, "--experiment", split]
        result = subprocess.run(cmd, cwd=_SRC)

        if result.returncode == 0:
            print(f"  [ok]   {tid}")
            ok += 1
        else:
            print(f"  [FAIL] {tid}  (exit code {result.returncode})")
            failed += 1

print(f"\n{'='*65}")
print(f"Done.  Total={total}  OK={ok}  Failed={failed}  Skipped={skipped}")
