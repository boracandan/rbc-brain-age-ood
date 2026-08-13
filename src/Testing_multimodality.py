"""Loads a trained model from a trial directory and evaluates it on the test set
(e.g. PNC for the NKItrimmed_BHRC_HBN_CCNP -> PNC OOD setting).

Usage:
    python Testing_multimodality.py --trial_id trial_20.3.2.3.1_gcn_fmri

Reads the trial's saved config.json to reconstruct the exact data pipeline and
model architecture that training used, loads best_model.pth, runs inference on
the test split, and writes test predictions + a results row.
"""

import pandas as pd
import numpy as np
import glob
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import model
import matplotlib.pyplot as plt
import json
import csv
import os

from pre_processing_util import pre_process_fMRI, pre_process_sMRI

print(torch.cuda.get_device_name())

parser = argparse.ArgumentParser(description="MAHGCN Age Regression — Test-set evaluation")
parser.add_argument("trial_id", type=str, help="Trial directory under the experiment folder to evaluate")
parser.add_argument("--experiment", type=str, default="MAHGCNExperiments")
parser.add_argument("--cache_dir", type=str, default=None,
                    help="Cache directory override (default: {dir_path}/cache_gen/cache)")
args = parser.parse_args()

dir_path = os.environ.get("RBC_DIR", r"C:\Users\Faruk\Code\rbc-brain-age-ood")
trial_dir = rf"{dir_path}/{args.experiment}/{args.trial_id}"

# === Load config from disk (must match training) ===
with open(rf"{trial_dir}/config.json") as f:
    config = json.load(f)

print(f"Testing {args.trial_id}")
print(f"  modality={config['modality']}  arch={config['gcn_mode']}  hd={config['hidden_dim']}  L={config['num_gnn_layers']}")
print(f"  train_datasets={config['train_datasets']}  test_datasets={config['test_datasets']}")

with open(rf"{dir_path}/saved_variables.json") as f:
    split_dict = json.load(f)

ATLAS_NAMES = {
    200:  "Schaefer2018p200n17",
    300:  "Schaefer2018p300n17",
    400:  "Schaefer2018p400n17",
    1000: "Schaefer2018p1000n17",
}

# Atlas-name convention used inside the FreeSurfer regionsurfacestats.tsv files
SMRI_ATLAS_NAMES = {
    200:  "Schaefer2018_200Parcels_17Networks_order",
    300:  "Schaefer2018_300Parcels_17Networks_order",
    400:  "Schaefer2018_400Parcels_17Networks_order",
    1000: "Schaefer2018_1000Parcels_17Networks_order",
}

# Standard FreeSurfer morphometry features used in brain-age studies
_SMRI_FEATURE_SETS = {
    "standard": ["SurfArea", "GrayVol", "ThickAvg", "MeanCurv"],
    "extended": ["SurfArea", "GrayVol", "ThickAvg", "MeanCurv", "ThickStd", "GausCurv", "FoldInd"],
}
SMRI_FEATURES = _SMRI_FEATURE_SETS["extended"]
NUM_SMRI_FEATS = len(SMRI_FEATURES)


def _resolve_paths(study, sub, n):
    """Return (path_fMRI, path_sMRI) for a subject, or (None, None) if not resolvable."""
    if study == "NKI":
        sid, ses = sub.split("-")
        path_fMRI = (
            rf"{dir_path}/NKI_CPAC/cpac_RBCv0/sub-{sid}/ses-{ses}/func"
            rf"/sub-{sid}_ses-{ses}_task-rest_acq-645_atlas-{ATLAS_NAMES[n]}"
            "_space-MNI152NLin6ASym_reg-36Parameter_desc-PartialNilearn_correlations.tsv"
        )
        path_sMRI = rf"{dir_path}/NKI_FreeSurfer/freesurfer/sub-{sid}_ses-{ses}/sub-{sid}_ses-{ses}_regionsurfacestats.tsv"
    elif study == "PNC":
        path_fMRI = (
            rf"{dir_path}/PNC_CPAC/cpac_RBCv0/sub-{sub}/ses-PNC1/func"
            rf"/sub-{sub}_ses-PNC1_task-rest_acq-singleband_atlas-{ATLAS_NAMES[n]}"
            "_space-MNI152NLin6ASym_reg-36Parameter_desc-PartialNilearn_correlations.tsv"
        )
        path_sMRI = rf"{dir_path}/PNC_FreeSurfer/freesurfer/sub-{sub}/sub-{sub}_regionsurfacestats.tsv"
    elif study == "BHRC":
        path_fMRI = None
        for run in ["run-1", "run-2"]:
            matches = glob.glob(
                rf"{dir_path}/BHRC_CPAC/cpac_RBCv0/sub-{sub}/ses-1/func"
                rf"/sub-{sub}*_task-rest_{run}_atlas-{ATLAS_NAMES[n]}"
                "_space-MNI152NLin6ASym_reg-36Parameter_desc-PartialNilearn_correlations.tsv"
            )
            if matches:
                path_fMRI = matches[0]
                break
        path_sMRI = rf"{dir_path}/BHRC_FreeSurfer/freesurfer/sub-{sub}_ses-1/sub-{sub}_ses-1_regionsurfacestats.tsv"
    elif study == "HBN":
        path_fMRI = None
        for run in ["run-1", "run-2"]:
            matches = glob.glob(
                rf"{dir_path}/HBN_CPAC/cpac_RBCv0/sub-{sub}/ses-HBNsite*/func"
                rf"/sub-{sub}_ses-*_task-rest_{run}_atlas-{ATLAS_NAMES[n]}"
                "_space-MNI152NLin6ASym_reg-36Parameter_desc-PartialNilearn_correlations.tsv"
            )
            if matches:
                path_fMRI = matches[0]
                break
        sMRI_matches = glob.glob(rf"{dir_path}/HBN_FreeSurfer/freesurfer/sub-{sub}/sub-{sub}_regionsurfacestats.tsv")
        path_sMRI = sMRI_matches[0] if sMRI_matches else None
    elif study == "CCNP":
        path_fMRI = None
        for run in ["run-01", "run-02"]:
            matches = glob.glob(
                rf"{dir_path}/CCNP_CPAC/cpac_RBCv0/sub-{sub}/ses-1/func"
                rf"/sub-{sub}_ses-*_task-rest_{run}_atlas-{ATLAS_NAMES[n]}"
                "_space-MNI152NLin6ASym_reg-36Parameter_desc-PartialNilearn_correlations.tsv"
            )
            if matches:
                path_fMRI = matches[0]
                break
        path_sMRI = rf"{dir_path}/CCNP_FreeSurfer/freesurfer/sub-{sub}/sub-{sub}_regionsurfacestats.tsv"
    else:
        return None, None

    if path_fMRI is not None and not os.path.exists(path_fMRI):
        path_fMRI = None
    if path_sMRI is not None and not os.path.exists(path_sMRI):
        path_sMRI = None
    return path_fMRI, path_sMRI


def _load_data(study, sub, n, modality):
    path_fMRI, path_sMRI = _resolve_paths(study, sub, n)

    # In multimodal mode we need both; in fMRI-only we only need the FC.
    if path_fMRI is None or (modality == "fMRI&sMRI" and path_sMRI is None):
        raise FileNotFoundError(f"Missing files for {study}/{sub} at scale {n} (modality={modality})")

    # --- fMRI: FC matrix + processing ---
    mat = pd.read_csv(path_fMRI, sep="\t", header=None).values.astype(np.float32)
    if mat.shape != (n, n):
        raise ValueError(f"FC for {study}/{sub} at scale {n} has wrong shape {mat.shape} -- corrupt or stub file? path={path_fMRI}")
    
    mat = pre_process_fMRI(mat, config)
    
    if modality != "fMRI&sMRI":
        return (torch.tensor(mat),)

    # --- sMRI: regional surface stats for the matching atlas, FC-column order ---
    reg_surfs = pd.read_csv(path_sMRI, sep="\t")
    feat_mat = pre_process_sMRI(reg_surfs, SMRI_FEATURES, SMRI_ATLAS_NAMES, resolution=n, config=config)

    if feat_mat.shape != (n, len(SMRI_FEATURES)):
        raise ValueError(f"sMRI features for {study}/{sub} at scale {n} have wrong shape {feat_mat.shape}, expected ({n}, {len(SMRI_FEATURES)}) -- corrupt TSV? path={path_sMRI}")
    
    return (torch.tensor(mat), torch.tensor(feat_mat))


_fc_cache       = None  # {scale: np.ndarray (N, scale, scale)} — set before RBCDataset is instantiated
_fc_cache_index = None  # {subject_id: row_index}

class RBCDataset(Dataset):
    def __init__(self, study_subject_ids, ages, modality):
        self.modality = modality
        self.ages = torch.tensor(ages, dtype=torch.float32)
        if _fc_cache is not None:
            print(f"Mapping {len(study_subject_ids)} subjects from FC cache...")
            self.indices = [_fc_cache_index[sub] for sub in study_subject_ids]
            self.data = None
        else:
            print(f"Preloading {len(study_subject_ids)} subjects into RAM...")
            self.indices = None
            self.data = [
                _load_data(*sub.split("_", 1), config["roi_scale"], modality) for sub in tqdm(study_subject_ids)
            ]

    def __len__(self):
        return len(self.ages)

    def __getitem__(self, idx):
        if self.data is not None:
            return (*self.data[idx],), self.ages[idx]
        i = self.indices[idx]
        return (
            *(torch.from_numpy(_fc_cache[modality][i]) for modality in self.modality.split("&")),
        ), self.ages[idx]


def normalize_ages(age_lookup, *ids):
    train_ids, val_ids, test_ids = ids
    if config["age_norm"] == "standardize":
        mean = age_lookup.loc[train_ids].mean()
        std  = age_lookup.loc[train_ids].std()

        train_ages = (age_lookup.loc[train_ids].values - mean) / std
        val_ages   = (age_lookup.loc[val_ids].values - mean) / std
        test_ages  = (age_lookup.loc[test_ids].values  - mean) / std

    elif config["age_norm"] == "minmax_0_1":
        max_age = 22 # 100 if "NKItrimmed" not in config["train_datasets"].split("_") else 22

        train_ages = age_lookup.loc[train_ids].values / max_age
        val_ages   = age_lookup.loc[val_ids].values   / max_age
        test_ages  = age_lookup.loc[test_ids].values  / max_age

    return (train_ages, val_ages, test_ages)


def _load_demo_df(bids_study):
    """Load participants TSV and return df with prefixed participant_id ready for age lookup."""
    tsv = rf"{dir_path}/{bids_study}_BIDS/study-{bids_study}_desc-participants.tsv"
    if bids_study == "NKI":
        df = pd.read_csv(tsv, sep="\t")[["participant_id", "session_id", "age"]].dropna(subset=["age"])
        df["participant_id"] = "NKI_" + df["participant_id"].astype(str) + "-" + df["session_id"].astype(str)
    else:
        df = pd.read_csv(tsv, sep="\t").drop_duplicates(subset="participant_id")[["participant_id", "age"]]
        df["participant_id"] = bids_study + "_" + df["participant_id"].astype(str)
    return df[["participant_id", "age"]]


def get_meta_data(train_studies, test_studies):
    if train_studies == test_studies:
        study = train_studies
        bids_study = "NKI" if study == "NKItrimmed" else study

        demo_df = _load_demo_df(bids_study)

        raw_train_val = split_dict[f"{study.lower()}_age_trainval_subjects"]
        raw_test      = split_dict[f"{study.lower()}_age_test_subjects"]

        train_val_ids = [f"{bids_study}_{sid}" for sid in raw_train_val]
        test_ids      = [f"{bids_study}_{sid}" for sid in raw_test]

        train_ids      = train_val_ids[:int(len(train_val_ids) * .9)]
        validation_ids = train_val_ids[int(len(train_val_ids) * .9):]

        age_lookup = demo_df.set_index("participant_id")["age"]
        return age_lookup, train_ids, validation_ids, test_ids

    else:
        all_train_ids = []
        test_ids = []
        df_list = []

        for study in train_studies.split("_"):
            bids_study = "NKI" if study == "NKItrimmed" else study
            study_subs = (split_dict.get(f"{study.lower()}_age_trainval_subjects", []) +
                          split_dict.get(f"{study.lower()}_age_test_subjects", []))
            all_train_ids.extend([f"{bids_study}_{sid}" for sid in study_subs])
            df_list.append(_load_demo_df(bids_study))

        for study in test_studies.split("_"):
            bids_study = "NKI" if study == "NKItrimmed" else study
            study_subs = (split_dict.get(f"{study.lower()}_age_trainval_subjects", []) +
                          split_dict.get(f"{study.lower()}_age_test_subjects", []))
            test_ids.extend([f"{bids_study}_{sid}" for sid in study_subs])
            df_list.append(_load_demo_df(bids_study))

        df_combined = pd.concat(df_list).drop_duplicates(subset="participant_id")
        age_lookup = df_combined.set_index("participant_id")["age"]

        import random
        random.seed(42)
        random.shuffle(all_train_ids)

        return age_lookup, all_train_ids[:int(len(all_train_ids) * .9)], all_train_ids[int(len(all_train_ids) * .9):], test_ids


age_lookup, train_ids, validation_ids, test_ids = get_meta_data(config["train_datasets"], config["test_datasets"])
train_ages, val_ages, test_ages = normalize_ages(age_lookup, train_ids, validation_ids, test_ids)

# === Cache loading (test only — no build) ===
_cache_dir  = args.cache_dir if args.cache_dir else rf"{dir_path}/cache_gen/cache"
_cache_key  = "_".join([config["fc_processing"], config["train_datasets"], config["test_datasets"],
                        config["modality"], str(config["roi_scale"])])
_cache_subs = rf"{_cache_dir}/{_cache_key}_subjects.json"
_modalities = ["fMRI", "sMRI"] if config["modality"] == "fMRI&sMRI" else ["fMRI"]
_cache_npy  = {m: rf"{_cache_dir}/{_cache_key}_{m}.npy" for m in _modalities}

if all(os.path.exists(p) for p in [_cache_subs] + list(_cache_npy.values())):
    print(f"Cache found — loading '{_cache_key}'...")
    _cached_ids     = json.load(open(_cache_subs))
    _fc_cache_index = {sid: i for i, sid in enumerate(_cached_ids)}
    _fc_cache       = {m: np.load(_cache_npy[m]) for m in _modalities}
    print("Cache loaded.")
else:
    print("Cache not found — will preload test subjects into RAM.")

test_dataset = RBCDataset(test_ids, test_ages, config["modality"])
test_loader  = DataLoader(test_dataset, batch_size=config["batch_size"], pin_memory=True)

# === Model: reconstruct from config, load checkpoint ===
feat_dim = len(SMRI_FEATURES) if config["modality"] == "fMRI&sMRI" else config["roi_scale"]
model = model.fMRINet(
    ROInum=config["roi_scale"],
    feat_dim=feat_dim,
    activation=config["age_output"],
    hidden_dim=config["hidden_dim"],
    gcn_mode=config["gcn_mode"],
    num_heads=config["num_heads"],
    num_gnn_layers=config["num_gnn_layers"],
    degree_normalize=config["degree_normalize"],
)

ckpt_path = rf"{trial_dir}/ckpt/best_model.pth"
model.load_state_dict(torch.load(ckpt_path, map_location="cuda"))
model.cuda()
model.eval()

# === Inference on test set ===
test_preds = []
test_true  = []
with torch.no_grad():
    for inputs, y in tqdm(test_loader, desc="Test inference"):
        inputs = [t.cuda(non_blocking=True) for t in inputs]
        out = model(*inputs)
        test_preds.extend(out.squeeze(-1).cpu().tolist())
        test_true.extend(y.tolist())

test_preds = np.array(test_preds)
test_true  = np.array(test_true)

# Save raw predictions (matches training's val/train_preds convention — normalized space)
np.save(rf"{trial_dir}/best_test_preds.npy", test_preds)
np.save(rf"{trial_dir}/best_test_true.npy",  test_true)
# Also save the subject IDs for downstream per-study / per-subject analysis
with open(rf"{trial_dir}/best_test_subjects.json", "w") as f:
    json.dump(test_ids, f)

# Invert normalization so all metrics + plot are in years (matches old Testing.py convention)
if config["age_norm"] == "minmax_0_1":
    max_age = 22 # 100 if "NKItrimmed" not in config["train_datasets"].split("_") else 22
    true_yrs  = test_true * max_age
    preds_yrs = test_preds * max_age
elif config["age_norm"] == "standardize":
    train_mean = float(age_lookup.loc[train_ids].mean())
    train_std  = float(age_lookup.loc[train_ids].std())
    true_yrs  = test_true * train_std + train_mean
    preds_yrs = test_preds * train_std + train_mean
else:
    true_yrs, preds_yrs = test_true, test_preds

# Metrics in years
test_r    = float(np.corrcoef(true_yrs, preds_yrs)[0, 1])
test_mae  = float(np.mean(np.abs(preds_yrs - true_yrs)))
test_mse  = float(np.mean((preds_yrs - true_yrs) ** 2))
ss_res    = float(np.sum((true_yrs - preds_yrs) ** 2))
ss_tot    = float(np.sum((true_yrs - true_yrs.mean()) ** 2))
test_r2   = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

print(f"\nTest results for {args.trial_id} (n={len(test_preds)}):")
print(f"  Pearson r:  {test_r:.3f}")
print(f"  R²:         {test_r2:.4f}")
print(f"  MAE (yrs):  {test_mae:.4f}")
print(f"  MSE (yrs²): {test_mse:.4f}")

# Append to test_results.csv (matches old Testing.py schema)
test_results_path = rf"{dir_path}/{args.experiment}/test_results.csv"
row = {
    "trial_id":       args.trial_id,
    "train_datasets": config["train_datasets"],
    "test_datasets":  config["test_datasets"],
    "test_r":         round(test_r, 4),
    "test_r2":        round(test_r2, 4),
    "test_mae":       round(test_mae, 4),
    "test_mse":       round(test_mse, 4),
}
write_header = not os.path.exists(test_results_path)
with open(test_results_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    if write_header:
        writer.writeheader()
    writer.writerow(row)

print(f"\nResults appended to {test_results_path}")

# === Scatter plot: predicted vs true ages (in years) ===
plots_dir = rf"{trial_dir}/test_plots"
os.makedirs(plots_dir, exist_ok=True)

age_min = min(true_yrs.min(), preds_yrs.min())
age_max = max(true_yrs.max(), preds_yrs.max())

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(true_yrs, preds_yrs, alpha=0.4, s=15, edgecolors="none")
ax.plot([age_min, age_max], [age_min, age_max], "r--", linewidth=1)
ax.set_xlabel("True Age (years)")
ax.set_ylabel("Predicted Age (years)")
ax.set_title(f"{args.trial_id}\nr={test_r:.3f}  MAE={test_mae:.2f}y  R²={test_r2:.3f}")
ax.set_xlim(age_min, age_max)
ax.set_ylim(age_min, age_max)
ax.set_aspect("equal")
plt.tight_layout()
plot_path = rf"{plots_dir}/pred_vs_true.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Plot saved to {plot_path}")
