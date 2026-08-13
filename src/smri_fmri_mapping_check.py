"""
Verify that the FreeSurfer regionsurfacestats.tsv row order (after dropping
background rows and sorting by hemisphere, Index) matches the canonical
Schaefer 2018 ROI order from nilearn.

If this passes for all studies, sMRI features can be aligned to FC matrix
columns by sort order alone — no name lookup or nilearn dependency needed
at training time.
"""

import glob
import os
import pandas as pd
from nilearn import datasets

dir_path = r"C:\Users\Faruk\Code\rbc-brain-age-ood"
SCALES = [200, 300, 400, 1000]

STUDIES = {
    "NKI":  rf"{dir_path}\NKI_FreeSurfer\freesurfer\sub-*\sub-*_regionsurfacestats.tsv",
    "PNC":  rf"{dir_path}\PNC_FreeSurfer\freesurfer\sub-*\sub-*_regionsurfacestats.tsv",
    "HBN":  rf"{dir_path}\HBN_FreeSurfer\freesurfer\sub-*\sub-*_regionsurfacestats.tsv",
    "CCNP": rf"{dir_path}\CCNP_FreeSurfer\freesurfer\sub-*\sub-*_regionsurfacestats.tsv",
}

# BHRC TSVs are mostly git-annex stubs; hardcode a known-downloaded sample
BHRC_SAMPLE = rf"{dir_path}\BHRC_FreeSurfer\freesurfer\sub-20004_ses-1\sub-20004_ses-1_regionsurfacestats.tsv"


def canonical_labels(n_rois):
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=17, verbose=0)
    labels = [s.decode() if isinstance(s, bytes) else s for s in atlas.labels]
    if labels[0].lower() == "background":
        labels = labels[1:]
    return labels


def tsv_order(tsv_path, n_rois):
    df = pd.read_csv(tsv_path, sep="\t")
    rows = df[df["atlas"] == f"Schaefer2018_{n_rois}Parcels_17Networks_order"]
    rows = rows[rows["StructName"].str.startswith("17Networks")]
    rows = rows.sort_values(["hemisphere", "Index"])
    return rows["StructName"].tolist()


def check_study(study, glob_pattern):
    matches = glob.glob(glob_pattern)
    if not matches:
        print(f"[{study}] no FreeSurfer TSVs found at {glob_pattern}")
        return
    sample = matches[130]
    sub_id = os.path.basename(sample).split("_regionsurfacestats")[0]
    print(f"[{study}] checking subject {sub_id} ({len(matches)} TSVs available)")
    for n in SCALES:
        canon = canonical_labels(n)
        try:
            got = tsv_order(sample, n)
        except Exception as e:
            print(sample)
            print(f"  Schaefer{n}: ERROR reading TSV: {e}")
            continue
        if len(got) == 0:
            print(f"  Schaefer{n}: atlas not present in TSV")
            continue
        match = got == canon
        flag = "OK" if match else "MISMATCH"
        print(f"  Schaefer{n}:  TSV={len(got)}  canonical={len(canon)}  {flag}")
        if not match:
            mism = [(i, g, c) for i, (g, c) in enumerate(zip(got, canon)) if g != c][:3]
            for i, g, c in mism:
                print(f"    idx {i}: TSV={g!r}  canonical={c!r}")


def check_study_fixed(study, tsv_path):
    if not os.path.exists(tsv_path):
        print(f"[{study}] TSV not found at {tsv_path}")
        return
    sub_id = os.path.basename(tsv_path).split("_regionsurfacestats")[0]
    print(f"[{study}] checking subject {sub_id} (hardcoded path)")
    for n in SCALES:
        canon = canonical_labels(n)
        try:
            got = tsv_order(tsv_path, n)
        except Exception as e:
            print(f"  Schaefer{n}: ERROR reading TSV: {e}")
            continue
        if len(got) == 0:
            print(f"  Schaefer{n}: atlas not present in TSV")
            continue
        match = got == canon
        flag = "OK" if match else "MISMATCH"
        print(f"  Schaefer{n}:  TSV={len(got)}  canonical={len(canon)}  {flag}")
        if not match:
            mism = [(i, g, c) for i, (g, c) in enumerate(zip(got, canon)) if g != c][:3]
            for i, g, c in mism:
                print(f"    idx {i}: TSV={g!r}  canonical={c!r}")


if __name__ == "__main__":
    for study, pattern in STUDIES.items():
        check_study(study, pattern)
        print()
    check_study_fixed("BHRC", BHRC_SAMPLE)
    print()
