"""
Parameter counts for every model setup in the study.

GNNs: one fMRINet instance per hyperparameter-grid configuration (the grids of src/runner.py), counted with
      sum(p.numel() for p in model.parameters()); reported as min / max per setup.
Baselines: Ridge is exact (coefficients + intercept). SVR and XGBoost depend on the fitted model, so the
      script reports their definitions and upper bounds; exact counts need the fitted models.

Run from anywhere:
    python helper_scripts/count_parameters.py [--csv counts.csv]
"""

import argparse
import csv
import itertools
import json
import os
import re
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "src"))

import model as model_lib  # src/model.py (fMRINet)

ROI = 300
N_SMRI_FEATS = 7                      # "extended" FreeSurfer feature set used in the training script
FC_EDGES = ROI * (ROI - 1) // 2       # upper-triangular FC features used by the baselines

# Keep in sync with src/runner.py:  name -> (gcn_mode, feat_dim, hidden_dims, layers, heads)
SETUPS = {
    "GCN (fMRI)":          ("gcn", ROI,                 [32, 64, 128], [1, 2, 3], [1]),
    "GCN (fMRI+sMRI)":     ("gcn", N_SMRI_FEATS,        [8, 16, 64],   [1, 2, 3], [1]),
    "GAT (fMRI)":          ("gat", ROI,                 [32, 64, 128], [1, 2, 3], [1, 2, 4]),
    "GAT (fMRI+sMRI)":     ("gat", N_SMRI_FEATS,        [8, 16, 64],   [1, 2, 3], [1, 2, 4]),
    "GCN (concat)":        ("gcn", ROI + N_SMRI_FEATS,  [32, 64, 128], [1, 2, 3], [1]),
    "GAT (concat)":        ("gat", ROI + N_SMRI_FEATS,  [32, 64, 128], [1, 2, 3], [1, 2, 4]),
}
# runner trial prefix -> setup (to check the grids against the trials that were actually run)
PREFIX_TO_SETUP = {"trial_1": "GCN (fMRI)", "trial_1_nodeg": "GCN (fMRI)", "trial_2": "GCN (fMRI+sMRI)",
                   "trial_2_nodeg": "GCN (fMRI+sMRI)", "trial_3": "GAT (fMRI)", "trial_4": "GAT (fMRI+sMRI)"}
LOO_TRAIN = {  # src/runner.py _loo_splits
    "PNC": "NKItrimmed_BHRC_HBN_CCNP", "NKI": "BHRC_HBN_CCNP_PNC", "CCNP": "NKItrimmed_BHRC_HBN_PNC",
    "HBN": "NKItrimmed_BHRC_CCNP_PNC", "BHRC": "NKItrimmed_HBN_CCNP_PNC",
}


def count_gnn(mode, feat_dim, hidden, layers, heads, degree_normalize=True):
    m = model_lib.fMRINet(ROInum=ROI, feat_dim=feat_dim, activation="sigmoid", hidden_dim=hidden, gcn_mode=mode,
                          num_heads=heads, num_gnn_layers=layers, degree_normalize=degree_normalize)
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def grid_ran(split="LOO_NKI"):
    """(hidden, layers, heads) combinations that appear in a results table, per setup."""
    path = os.path.join(PROJECT, split, "results_table.csv")
    if not os.path.exists(path):
        return None
    seen = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            parts = row["trial_id"].split(".")
            setup = PREFIX_TO_SETUP.get(parts[0])
            if setup is None:
                continue
            gat = setup.startswith("GAT")
            seen.setdefault(setup, set()).add((int(parts[1]), int(parts[2]), int(parts[3]) if gat else 1))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="optional: write every GNN configuration and its count to this CSV")
    args = ap.parse_args()

    rows, summary = [], []
    for name, (mode, feat_dim, hiddens, layer_list, heads) in SETUPS.items():
        counts = []
        for h, L, nh in itertools.product(hiddens, layer_list, heads):
            n = count_gnn(mode, feat_dim, h, L, nh)
            counts.append((n, h, L, nh))
            rows.append({"setup": name, "hidden_dim": h, "layers": L, "heads": nh, "params": n})
        lo, hi = min(counts), max(counts)
        summary.append((name, len(counts), lo, hi))

    # the no-degree-normalization GCN has the same architecture; check that the flag adds no parameters
    assert count_gnn("gcn", ROI, 64, 2, 1, True) == count_gnn("gcn", ROI, 64, 2, 1, False)

    def fmt(c, gat):
        return f"{c[0]:>9,}  (hidden={c[1]}, layers={c[2]}" + (f", heads={c[3]})" if gat else ")")

    print("=" * 100)
    print("GNN parameter counts (fMRINet = GNN backbone + BN/Linear(64)/BN/Linear(1) head), min and max over the grid")
    print("=" * 100)
    print(f"{'setup':<18}{'configs':>8}   {'minimum':<42}{'maximum'}")
    for name, n, lo, hi in summary:
        gat = name.startswith("GAT")
        print(f"{name:<18}{n:>8}   {fmt(lo, gat):<42}{fmt(hi, gat)}")
    print("GCN no-deg variants have the same architecture and therefore the same counts as the GCN rows.")
    print("The concat rows are the new fMRI_concat_sMRI setups (feat_dim = 300 + 7); they have not been run yet.")

    ran = grid_ran()
    if ran is not None:
        print("\nCheck against the trials that ran (LOO_NKI/results_table.csv):")
        for name, combos in ran.items():
            _, _, hiddens, layer_list, heads = SETUPS[name]
            expected = set(itertools.product(hiddens, layer_list, heads))
            print(f"  {name:<18} grid in this script {'MATCHES' if expected == combos else 'DIFFERS FROM'} the trials that ran"
                  f" ({len(combos)} combinations)")

    print("\n" + "=" * 100)
    print("Baselines")
    print("=" * 100)
    n_fmri, n_multi = FC_EDGES, FC_EDGES + ROI * N_SMRI_FEATS
    print(f"Ridge   exact: coefficients + intercept   fMRI {n_fmri + 1:,}   fMRI+sMRI {n_multi + 1:,}"
          f"   (features: {n_fmri:,} FC edges, + {ROI * N_SMRI_FEATS:,} sMRI values)")

    sd = json.load(open(os.path.join(PROJECT, "saved_variables.json")))
    n_train = {}
    for fold, studies in LOO_TRAIN.items():
        pool = sum(len(sd.get(f"{s.lower()}_age_trainval_subjects", [])) + len(sd.get(f"{s.lower()}_age_test_subjects", []))
                   for s in studies.split("_"))
        n_train[fold] = int(pool * 0.9)
    lo_f, hi_f = min(n_train, key=n_train.get), max(n_train, key=n_train.get)
    print(f"SVR     depends on the fit. Dual coefficients + intercept <= n_train + 1 = {n_train[lo_f] + 1:,} ({lo_f} fold)"
          f" .. {n_train[hi_f] + 1:,} ({hi_f} fold).")
    print(f"        If the stored support vectors are counted too: <= n_train x n_features = "
          f"{n_train[lo_f] * n_fmri:,} .. {n_train[hi_f] * n_multi:,} values.")
    print("XGBoost depends on the fit (early stopping). Upper bound per ensemble: n_estimators (2000) x nodes per tree,"
          " nodes <= 2^(depth+1) - 1 -> depth 4: "
          f"{2000 * (2 ** 5 - 1):,}, depth 6: {2000 * (2 ** 7 - 1):,}, depth 8: {2000 * (2 ** 9 - 1):,} nodes.")
    print("        Exact count of a fitted model: len(booster.trees_to_dataframe()); SVR: model.n_support_.sum() (+1).")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["setup", "hidden_dim", "layers", "heads", "params"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} configurations to {args.csv}")


if __name__ == "__main__":
    main()
