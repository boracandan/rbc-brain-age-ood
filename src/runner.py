import subprocess
import time
from collections import defaultdict
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test_set", type=str, default=None, choices=["NKI", "CCNP", "HBN", "BHRC", "PNC"],
                    help="LOO test set to run. Omit to run all trials in the hardcoded list.")
parser.add_argument("--trial", type=str, nargs="+", default=None, choices=["gcn_fmri", "gcn_multi", "gat_fmri", "gat_multi"],
                    help="Trial type(s) to run. Omit to run all four. E.g. --trial gcn_fmri gcn_multi")
parser.add_argument("--degree_norm", action=argparse.BooleanOptionalAction, default=True,
                    help="GCN only: degree normalization (default: on). GAT always ignores this.")
args = parser.parse_args()

dir_path = os.environ.get("RBC_DIR", r"C:\Users\Faruk\Code\rbc-brain-age-ood")

trials = []


def generate_trials(
    trial_id,
    fc_processing,
    hidden_dims,
    layers,
    weight_decays,
    lrs,
    age_norm="minmax_0_1",
    age_output="sigmoid",
    degree_normalize=True,
    gcn_mode="gcn",
    num_heads=[1],
    roi_scale=300,
    modality="fMRI&sMRI",
    train_datasets="NKItrimmed_BHRC_HBN_CCNP",
    test_datasets="PNC",
    save_fc_cache=False,
    experiment="MAHGCNExperiments",
    _script="Training_multimodality.py",
):
    entries = []
    for processing in fc_processing:
        pro_entries = []
        pro_str = processing.split("_")[-1]
        for hidden_dim in hidden_dims:
            for layer_num in layers:
                for head_num in num_heads:
                    for l, weight_decay in enumerate(weight_decays):
                        for m, lr in enumerate(lrs):
                            identifier = f"{trial_id}.{hidden_dim}.{layer_num}.{l+1}.{m+1}" if gcn_mode == "gcn" else f"{trial_id}.{hidden_dim}.{layer_num}.{head_num}.{l+1}.{m+1}"
                            pro_entries.append(
                                {
                                    "trial_id": f"{identifier}.{pro_str}_{gcn_mode}_{'multi' if modality == 'fMRI&sMRI' else 'fmri'}",
                                    "_script": _script,
                                    "fc_processing": processing,
                                    "age_norm": age_norm,
                                    "age_output": age_output,
                                    "degree_normalize": degree_normalize,
                                    "lr": lr,
                                    "weight_decay": weight_decay,
                                    "gcn_mode": gcn_mode,
                                    "hidden_dim": hidden_dim,
                                    "num_heads": head_num,
                                    "num_gnn_layers": layer_num,
                                    "roi_scale": roi_scale,
                                    "modality": modality,
                                    "train_datasets": train_datasets,
                                    "test_datasets": test_datasets,
                                    "experiment": experiment,
                                    **( {"save_fc_cache": True} if (save_fc_cache and not pro_entries) else {} ),
                                }
                            )
        entries.extend(pro_entries)
    return entries


# ── Leave-one-out experiments ────────────────────────────────────
_loo_fcs = ["per_10", "per_20", "per_50"]
_loo_splits = {
    "PNC":  ("NKItrimmed_BHRC_HBN_CCNP", "PNC"),
    "NKI":  ("BHRC_HBN_CCNP_PNC",        "NKItrimmed"),
    "CCNP": ("NKItrimmed_BHRC_HBN_PNC",  "CCNP"),
    "HBN":  ("NKItrimmed_BHRC_CCNP_PNC", "HBN"),
    "BHRC": ("NKItrimmed_HBN_CCNP_PNC",  "BHRC"),
}

if args.test_set is not None:
    train_ds, test_ds = _loo_splits[args.test_set]
    exp = f"LOO_{args.test_set}"
    run_all = args.trial is None
    _deg_suffix = "" if args.degree_norm else "_nodeg"

    if run_all or "gcn_fmri" in args.trial:
        trials += generate_trials(f"trial_1{_deg_suffix}", _loo_fcs, [32, 64, 128], [1, 2, 3], [1e-5, 1e-4, 1e-3], [1e-4, 1e-3, 1e-2],
            gcn_mode="gcn", modality="fMRI", degree_normalize=args.degree_norm,
            train_datasets=train_ds, test_datasets=test_ds, save_fc_cache=True, experiment=exp)
    if run_all or "gcn_multi" in args.trial:
        trials += generate_trials(f"trial_2{_deg_suffix}", _loo_fcs, [8, 16, 64], [1, 2, 3], [1e-5, 1e-4, 1e-3], [1e-4, 1e-3, 1e-2],
            gcn_mode="gcn", modality="fMRI&sMRI", degree_normalize=args.degree_norm,
            train_datasets=train_ds, test_datasets=test_ds, experiment=exp)
    if run_all or "gat_fmri" in args.trial:
        trials += generate_trials(f"trial_3{_deg_suffix}", _loo_fcs, [32, 64, 128], [1, 2, 3], [1e-5, 1e-4, 1e-3], [1e-4, 1e-3, 1e-2],
            gcn_mode="gat", num_heads=[1, 2, 4], modality="fMRI", degree_normalize=False,
            train_datasets=train_ds, test_datasets=test_ds, experiment=exp)
    if run_all or "gat_multi" in args.trial:
        trials += generate_trials(f"trial_4{_deg_suffix}", _loo_fcs, [8, 16, 64], [1, 2, 3], [1e-5, 1e-4, 1e-3], [1e-4, 1e-3, 1e-2],
            gcn_mode="gat", num_heads=[1, 2, 4], modality="fMRI&sMRI", degree_normalize=False,
            train_datasets=train_ds, test_datasets=test_ds, experiment=exp)

times = defaultdict(list)

total = len(trials)
completed = 0
print(f"\n=== Starting {total} trials ===\n", flush=True)

for i, t in enumerate(trials):
    t = dict(t)
    script = t.pop("_script", "Training_multimodality.py")
    trial_id     = t.get("trial_id", "")
    trial_prefix = trial_id.split(".")[0]
    if os.path.isdir(f"{dir_path}/{t.get('experiment', exp)}/{trial_id}"):
        completed += 1
        continue
    cmd = ["python", "-u", script, "--use_args"]
    for k, v in t.items():
        if isinstance(v, bool):
            cmd.append(f"--{k}" if v else f"--no-{k}")
        else:
            cmd += [f"--{k}", str(v)]
    t_start = time.time()
    subprocess.run(cmd, cwd=os.path.join(dir_path, "src"))
    elapsed = time.time() - t_start
    completed += 1
    if trial_id:
        times[trial_prefix].append(elapsed)
    print(f"[{completed}/{total}] ({100*completed/total:.1f}%) {trial_id} — {elapsed/60:.1f} min", flush=True)

if times:
    for trial_id, times in times.items():
        print(f"\n=== {trial_id} timing summary ===")
        avg = sum(times) / len(times)
        print(f" runs, avg {avg/60:.2f} min ({avg:.0f}s) — total {sum(times)/3600:.1f}h")
