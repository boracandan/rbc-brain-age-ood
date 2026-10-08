import subprocess
import time
from collections import defaultdict
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test_set", type=str, default=None, choices=["NKI", "CCNP", "HBN", "BHRC", "PNC"],
                    help="LOO test set to run. Omit to run all trials in the hardcoded list.")
parser.add_argument("--trial", type=str, nargs="+", default=None,
                    choices=["gcn_fmri", "gcn_multi", "gat_fmri", "gat_multi", "gcn_concat", "gat_concat"],
                    help="Trial type(s) to run. Omit to run the four original types; the concat types must be "
                         "requested explicitly. E.g. --trial gcn_fmri gcn_concat")
parser.add_argument("--degree_norm", action=argparse.BooleanOptionalAction, default=True,
                    help="GCN only: degree normalization (default: on). GAT always ignores this.")
parser.add_argument("--dry_run", action="store_true",
                    help="Print an audit of what would run (counts, grids, existing trials, cache check, first command) and exit.")
parser.add_argument("--smoke", type=int, default=None, metavar="N",
                    help="Keep only N evenly spaced trials per trial type, run them for --smoke_epochs epochs and write "
                         "them to a separate '_smoke_<experiment>' folder (the real results are untouched).")
parser.add_argument("--smoke_epochs", type=int, default=2, help="Epochs per trial in --smoke mode (default: 2).")
parser.add_argument("--train_script", type=str, default=None,
                    help="Training script to use instead of the default Training_multimodality.py.")
args = parser.parse_args()

dir_path = os.environ.get("RBC_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

trials = []

# Suffix of the trial_id for each modality (run_testing.py parses it to group trials)
MODALITY_TAG = {"fMRI": "fmri", "fMRI&sMRI": "multi", "fMRI_concat_sMRI": "concat"}
# Which cache a modality reads (concat reuses the fMRI&sMRI cache; see Modality.cache_name in the training script)
CACHE_MODALITY = {"fMRI": "fMRI", "fMRI&sMRI": "fMRI&sMRI", "fMRI_concat_sMRI": "fMRI&sMRI"}


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
                                    "trial_id": f"{identifier}.{pro_str}_{gcn_mode}_{MODALITY_TAG[modality]}",
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
    if run_all or "gcn_concat" in args.trial:
        trials += generate_trials(f"trial_5{_deg_suffix}", _loo_fcs, [32, 64, 128], [1, 2, 3], [1e-5, 1e-4, 1e-3], [1e-4, 1e-3, 1e-2],
            gcn_mode="gcn", modality="fMRI_concat_sMRI", degree_normalize=args.degree_norm,
            train_datasets=train_ds, test_datasets=test_ds, experiment=exp)
    if run_all or "gat_concat" in args.trial:
        trials += generate_trials(f"trial_6{_deg_suffix}", _loo_fcs, [32, 64, 128], [1, 2, 3], [1e-5, 1e-4, 1e-3], [1e-4, 1e-3, 1e-2],
            gcn_mode="gat", num_heads=[1, 2, 4], modality="fMRI_concat_sMRI", degree_normalize=False,
            train_datasets=train_ds, test_datasets=test_ds, experiment=exp)

if args.train_script:
    for t in trials:
        t["_script"] = args.train_script


def _preview_cmd(t):
    """Display-only copy of the command the loop below builds (the loop itself is unchanged)."""
    t = dict(t)
    script = t.pop("_script", "Training_multimodality.py")
    cmd = ["python", "-u", script, "--use_args"]
    for k, v in t.items():
        if isinstance(v, bool):
            cmd.append(f"--{k}" if v else f"--no-{k}")
        else:
            cmd += [f"--{k}", str(v)]
    return " ".join(f'"{c}"' if any(ch in c for ch in "&| ") else c for c in cmd)


def _trial_type(t):
    return t["trial_id"].split(".")[0]


def audit(trials):
    print(f"\n=== Dry run: --test_set {args.test_set} (train={train_ds}, test={test_ds}) ===")
    ids = [t["trial_id"] for t in trials]
    if len(ids) != len(set(ids)):
        print(f"!! {len(ids) - len(set(ids))} duplicate trial_ids in this list")
    by_type = defaultdict(list)
    for t in trials:
        by_type[_trial_type(t)].append(t)
    for prefix, group in by_type.items():
        exists = sum(os.path.isdir(f"{dir_path}/{t['experiment']}/{t['trial_id']}") for t in group)
        g0 = group[0]
        print(f"\n{prefix}: {len(group)} trials  ({exists} already exist and are skipped, {len(group) - exists} to run)")
        print(f"  gcn_mode={g0['gcn_mode']}  modality={g0['modality']}  degree_normalize={g0['degree_normalize']}  "
              f"roi_scale={g0['roi_scale']}  epochs={g0.get('epochs', 'script default')}  folder={g0['experiment']}")
        for key in ["fc_processing", "hidden_dim", "num_gnn_layers", "num_heads", "weight_decay", "lr"]:
            print(f"  {key}: {sorted({t[key] for t in group})}")
    print("\nCache check (without a cache the training script preloads from disk, which is much slower):")
    seen = set()
    for t in trials:
        cm = CACHE_MODALITY[t["modality"]]
        key = "_".join([t["fc_processing"], t["train_datasets"], t["test_datasets"], cm, str(t["roi_scale"])])
        if key in seen:
            continue
        seen.add(key)
        exp_cache = f"{dir_path}/{t['experiment']}/cache"
        cdir = exp_cache if os.path.exists(f"{exp_cache}/{key}_subjects.json") else f"{dir_path}/cache_gen/cache"
        need = ["_subjects.json", "_fMRI.npy"] + ([] if cm == "fMRI" else ["_sMRI.npy"])
        missing = [s for s in need if not os.path.exists(f"{cdir}/{key}{s}")]
        status = "OK" if not missing else f"MISSING {missing}" + ("  (this trial builds it: save_fc_cache)" if t.get("save_fc_cache") else "")
        print(f"  {status:<8} {key}")
    if trials:
        print(f"\nFirst command (run from {os.path.join(dir_path, 'src')}):\n  {_preview_cmd(trials[0])}")
    print(f"\nTotal in this list: {len(trials)} trials. Nothing was run.")


if (args.dry_run or args.smoke) and args.test_set is None:
    parser.error("--dry_run and --smoke need --test_set")

if args.smoke:
    # N evenly spaced trials per trial type, a few epochs, in a fresh folder so real results are untouched
    # (the timestamp keeps repeated smoke runs, e.g. an A/B timing, from being skipped as "already done")
    _smoke_exp = f"_smoke_{exp}_{time.strftime('%H%M%S')}"
    by_type = defaultdict(list)
    for t in trials:
        by_type[_trial_type(t)].append(t)
    picked = []
    for group in by_type.values():
        for j in range(min(args.smoke, len(group))):
            picked.append(dict(group[j * len(group) // args.smoke], experiment=_smoke_exp, epochs=args.smoke_epochs))
    trials = picked
    print(f"Smoke mode: {len(trials)} trials, {args.smoke_epochs} epochs each, folder {_smoke_exp}")

if args.dry_run:
    audit(trials)
    raise SystemExit(0)

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
