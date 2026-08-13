import pandas as pd
import numpy as np
import glob
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import model
from sklearn import metrics
import json
import csv
import os

from pre_processing_util import pre_process_fMRI, pre_process_sMRI

try:
    torch.amp.GradScaler
    _GradScaler = lambda: torch.amp.GradScaler('cuda')
    _autocast   = lambda: torch.amp.autocast('cuda')
except AttributeError:
    _GradScaler = torch.cuda.amp.GradScaler
    _autocast   = torch.cuda.amp.autocast

print(torch.cuda.get_device_name())

parser = argparse.ArgumentParser(description="MAHGCN Age Regression Training")
parser.add_argument("--use_args",        action="store_true",                   help="Override config with CLI flags")
parser.add_argument("--trial_id",        type=str,   default=None)
parser.add_argument("--fc_processing",   type=str,   default=None)
parser.add_argument("--age_norm",        type=str,   default=None)
parser.add_argument("--age_output",      type=str,   default=None)
parser.add_argument("--lr",              type=float, default=None)
parser.add_argument("--weight_decay",    type=float, default=None)
parser.add_argument("--epochs",          type=int,   default=None)
parser.add_argument("--batch_size",      type=int,   default=None)
parser.add_argument("--gcn_mode",        type=str,   default=None)
parser.add_argument("--hidden_dim",      type=int,   default=None)
parser.add_argument("--num_heads",       type=int,   default=None)
parser.add_argument("--num_gnn_layers",  type=int,   default=None)
parser.add_argument("--roi_scale",       type=int,   default=None)
parser.add_argument("--degree_normalize",action=argparse.BooleanOptionalAction, default=None)
parser.add_argument("--save_fc_cache",   action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--experiment",      type=str, default="MAHGCNExperiments")
parser.add_argument("--train_datasets",   type=str, default=None)
parser.add_argument("--test_datasets",   type=str, default=None)
parser.add_argument("--modality",        type=str, default=None)
parser.add_argument("--patience",   type=int, default=None)
args = parser.parse_args()

dir_path = os.environ.get("RBC_DIR", r"C:\Users\Faruk\Code\rbc-brain-age-ood")

# Trial Settings

config = {
    "trial_id":         "trial_01_absthreshold",
    "fc_processing":    "abs_threshold_0.2",   # or "clip_zero", "raw", "abs_threshold_per_roi_th200_th300_th400_th1000"
    "age_norm":         "standardize",          # or "minmax_0_1" (divides by 100)
    "age_output":       "sigmoid",              # or "linear"
    "lr":               0.001,
    "weight_decay":     1e-4,
    "epochs":           300,
    "batch_size":       64,
    "accumulate_steps": 2,
    "roi_scale":       400, # 200 | 300 | 400 | 1000
    "gcn_mode":       "gcn",               # "gcn" | "gat"
    "hidden_dim":      8,
    "num_heads":       1,                       # GAT only
    "num_gnn_layers":  2,
    "degree_normalize": True,                   # the GCN fix
    "train_datasets":  "NKI", # dataset1_dataset2_...
    "test_datasets": "NKI", # dataset1_dataset2_...
    "modality": "fMRI", # "fMRI" | "fMRI&sMRI"
    "patience": 20,
}

if args.use_args:
    for key in ["trial_id", "fc_processing", "age_norm", "age_output", "lr", "gcn_mode", "modality", "roi_scale",
                "hidden_dim", "num_heads", "num_gnn_layers",
                "weight_decay", "epochs", "batch_size", "degree_normalize", "train_datasets", "test_datasets", "patience"]:
        val = getattr(args, key)
        if val is not None:
            config[key] = val

trial_dir = rf"{dir_path}/{args.experiment}/{config['trial_id']}"
os.makedirs(trial_dir, exist_ok=True)
os.makedirs(rf"{trial_dir}/plots", exist_ok=True)
os.makedirs(rf"{trial_dir}/ckpt",  exist_ok=True)

with open(rf"{trial_dir}/config.json", "w") as f:
    json.dump(config, f, indent=2)


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

    if feat_mat.shape != (n, NUM_SMRI_FEATS):
        raise ValueError(f"sMRI features for {study}/{sub} at scale {n} have wrong shape {feat_mat.shape}, expected ({n}, {NUM_SMRI_FEATS}) -- corrupt TSV? path={path_sMRI}")
    
    return (torch.tensor(mat), torch.tensor(feat_mat))


_fc_cache       = None  # {scale: np.ndarray (N, scale, scale)} — set before RBCDataset is instantiated
_fc_cache_index = None  # {subject_id: row_index}

class RBCDataset(Dataset):
    def __init__(self, study_subject_ids, ages, modality, smri_mean=None, smri_std=None):
        self.modality = modality
        self.ages = torch.tensor(ages, dtype=torch.float32)
        self.smri_mean = smri_mean
        self.smri_std  = smri_std
        if _fc_cache is not None:
            print(f"Mapping {len(study_subject_ids)} subjects from FC cache...")
            self.indices = [_fc_cache_index[sub] for sub in study_subject_ids]
            self.data = None
        else:
            print(f"Preloading {len(study_subject_ids)} subjects into RAM...")
            self.indices = None
            raw = [_load_data(*sub.split("_", 1), config["roi_scale"], modality) for sub in tqdm(study_subject_ids)]
            if modality == "fMRI&sMRI":
                fMRI_tensors = [d[0] for d in raw]
                smri_arr = np.stack([d[1].numpy() for d in raw])  # (N, 300, 4)
                if self.smri_mean is None:
                    self.smri_mean = smri_arr.mean(axis=0)
                    self.smri_std  = smri_arr.std(axis=0)
                smri_arr = (smri_arr - self.smri_mean) / (self.smri_std + 1e-8)
                self.data = list(zip(fMRI_tensors, [torch.tensor(s) for s in smri_arr]))
            else:
                self.data = [(d[0],) for d in raw]

    def __len__(self):
        return len(self.ages)

    def __getitem__(self, idx):
        if self.data is not None:
            return (*self.data[idx],), self.ages[idx]
        i = self.indices[idx]
        return (
            *(_fc_cache[modality][i] for modality in self.modality.split("&")),
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
        max_age = 22

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

# Cache: one .npy per modality + a subjects index JSON. Cache key includes roi_scale.
# Always load from cache if present; build+save only when --save_fc_cache is passed.
_cache_key  = "_".join([config["fc_processing"], config["train_datasets"], config["test_datasets"],
                        config["modality"], str(config["roi_scale"])])
_cache_dir  = rf"{dir_path}/{args.experiment}/cache" if os.path.exists(rf"{dir_path}/{args.experiment}/cache/{_cache_key}_subjects.json") else rf"{dir_path}/cache_gen/cache"
_cache_subs = rf"{_cache_dir}/{_cache_key}_subjects.json"
_modalities = ["fMRI", "sMRI"] if config["modality"] == "fMRI&sMRI" else ["fMRI"]
_cache_npy  = {m: rf"{_cache_dir}/{_cache_key}_{m}.npy" for m in _modalities}

if all(os.path.exists(p) for p in [_cache_subs] + list(_cache_npy.values())):
    print(f"Cache found — loading '{_cache_key}'...")
    _cached_ids     = json.load(open(_cache_subs))
    _fc_cache_index = {sid: i for i, sid in enumerate(_cached_ids)}
    _fc_cache = {m: torch.from_numpy(np.load(_cache_npy[m])) for m in _modalities}
    print("Cache loaded.")
elif args.save_fc_cache:
    os.makedirs(_cache_dir, exist_ok=True)
    print(f"Building cache for '{_cache_key}'...")
    all_subs = list(train_ids) + list(validation_ids) + list(test_ids)
    fc_list, smri_list = [], []
    for sub in tqdm(all_subs):
        # _load_data returns (fc,) for fMRI-only or (fc, feat) for fMRI&sMRI
        data = _load_data(*sub.split("_", 1), config["roi_scale"], config["modality"])
        fc_list.append(data[0].numpy())
        if config["modality"] == "fMRI&sMRI":
            smri_list.append(data[1].numpy())
    _fc_cache = {"fMRI": np.stack(fc_list)}
    if config["modality"] == "fMRI&sMRI":
        n_train = len(train_ids) + len(validation_ids)
        temp_smri_list = np.stack(smri_list)
        mean = temp_smri_list[:n_train].mean(axis=0)
        std = temp_smri_list[:n_train].std(axis=0)
        temp_smri_list = (temp_smri_list - mean) / (std + 1e-8)
        _fc_cache["sMRI"] = temp_smri_list
    for m in _modalities:
        np.save(_cache_npy[m], _fc_cache[m])
    json.dump(all_subs, open(_cache_subs, "w"))
    _fc_cache_index = {sid: i for i, sid in enumerate(all_subs)}
    _fc_cache = {m: torch.from_numpy(_fc_cache[m]) for m in _modalities}
    print(f"Cache saved to {_cache_dir}")

train_dataset = RBCDataset(train_ids, train_ages, config["modality"])
val_dataset   = RBCDataset(validation_ids, val_ages,  config["modality"],
                            smri_mean=train_dataset.smri_mean, smri_std=train_dataset.smri_std)
test_dataset  = RBCDataset(test_ids,  test_ages,  config["modality"],
                            smri_mean=train_dataset.smri_mean, smri_std=train_dataset.smri_std)

import platform
_nw = 0 if platform.system() == "Windows" else 2
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, pin_memory=True, num_workers=_nw, persistent_workers=_nw > 0)
test_loader  = DataLoader(test_dataset,  batch_size=config["batch_size"], pin_memory=True, num_workers=_nw, persistent_workers=_nw > 0)
val_loader   = DataLoader(val_dataset,   batch_size=config["batch_size"], pin_memory=True, num_workers=_nw, persistent_workers=_nw > 0)

# Training

counter = 0
stop_early = False

train_losses = []
val_losses   = []
val_rs       = [] 

qual_all = []

loss_func = nn.MSELoss()

lr = config["lr"]
best_val_loss = float("inf")
best_train_loss = float("inf")

# Node-feature dimension depends on modality:
#   fMRI-only   → features are FC-profile rows (length = ROInum)
#   fMRI&sMRI   → features are per-ROI sMRI morphometry values (length = NUM_SMRI_FEATS)
feat_dim = NUM_SMRI_FEATS if config["modality"] == "fMRI&sMRI" else config["roi_scale"]
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


optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config["weight_decay"])
model.cuda()

scaler = _GradScaler()
accumulate_steps = config["accumulate_steps"]
optimizer.zero_grad(set_to_none=True)

for epoch in range(config["epochs"]):
    # Training
    model.train() 
    epoch_losses = []

    for step, (inputs, y) in enumerate(train_loader):
        # inputs is (fc,) for fMRI-only or (fc, feat) for fMRI&sMRI
        y = y.cuda(non_blocking=True)
        inputs = [t.cuda(non_blocking=True) for t in inputs]

        with _autocast():
            output = model(*inputs)  # fMRINet.forward(g_matrix, node_features=None)
            loss = loss_func(output.squeeze(), y) / accumulate_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulate_steps == 0 or (step + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_losses.append(loss.detach() * accumulate_steps)

        # print(f'[Epoch {epoch+1}, Batch {step+1}] train loss: {loss.item():.3f}') Batch Level Printing

    train_losses.append(torch.stack(epoch_losses).mean().item())

    # Validation
    model.eval()

    val_preds = []
    val_true = []
    with torch.no_grad():
        for inputs, y in val_loader:
            inputs = [t.cuda(non_blocking=True) for t in inputs]

            with _autocast():
                val_output = model(*inputs)

            val_preds.extend(val_output.squeeze(-1).cpu().tolist())
            val_true.extend(y.tolist())
    
    # R² = 0.683, Pearson r = 0.886 (p = 0.0000), Mean Absolute Error: 8.926162762162495
    val_loss = np.mean((np.array(val_preds) - np.array(val_true))**2)
    val_r    = np.corrcoef(val_true, val_preds)[0, 1]
    val_losses.append(val_loss)
    val_rs.append(val_r)
    print(f'Epoch {epoch+1} | Train Loss: {train_losses[-1]:.3f}, Val Loss: {val_loss:.3f}, r: {val_r:.3f}') # Epoch Level Printing

    if val_loss < best_val_loss:
        counter = 0 # Reset on Improvement
        best_val_loss = val_loss
        best_train_loss = train_losses[-1]
        torch.save(model.state_dict(), rf"{trial_dir}/ckpt/best_model.pth")
        # For Plotting
        np.save(rf"{trial_dir}/best_val_preds.npy", np.array(val_preds))
        np.save(rf"{trial_dir}/best_val_true.npy",  np.array(val_true))
        train_preds_full = []
        train_true_full  = []
        with torch.no_grad():
            for inputs, y in train_loader:
                inputs = [t.cuda(non_blocking=True) for t in inputs]
                with _autocast():
                    out = model(*inputs)
                train_preds_full.extend(out.squeeze(-1).cpu().tolist())
                train_true_full.extend(y.tolist())
        np.save(rf"{trial_dir}/best_train_preds.npy", np.array(train_preds_full))
        np.save(rf"{trial_dir}/best_train_true.npy",  np.array(train_true_full))


    else:
        counter += 1
        if counter >= config["patience"]:
            stop_early = True
            break

# Logging

np.save(rf"{trial_dir}/train_val_loss.npy",
        np.stack([train_losses, val_losses], axis=1))

result_row = {
    "trial_id":      config["trial_id"],
    "gcn_mode":      config["gcn_mode"],
    "fc_processing": config["fc_processing"],
    "age_norm":      config["age_norm"],
    "lr":            config["lr"],
    "weight_decay":  config["weight_decay"],
    "degree_norm":   config["degree_normalize"],
    "best_state_val_loss":    best_val_loss,
    "best_state_train_loss": best_train_loss,
    "final_train_loss": train_losses[-1],
    "final_val_loss": val_losses[-1],
    "stopped_early": stop_early
}

results_path = rf"{dir_path}/{args.experiment}/results_table.csv"
write_header = not os.path.exists(results_path)
with open(results_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=result_row.keys())
    if write_header:
        writer.writeheader()
    writer.writerow(result_row)

