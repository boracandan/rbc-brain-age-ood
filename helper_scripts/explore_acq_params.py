"""
Explore acquisition parameter distribution for task-rest in NKI CPAC data.
Breaks down how subjects are distributed across acq variants.
"""

import os
import glob
from collections import defaultdict

dir_path = r"C:\Users\Faruk\Code\rbc-brain-age-ood"

# Get all rest-task TSV files only
rest_files = glob.glob(
    rf"{dir_path}\NKI_CPAC\cpac_RBCv0\sub-*\ses-BAS1\func\*_task-rest_*_correlations.tsv"
)

print(f"Total task-rest files (BAS1): {len(rest_files)}")

# Parse filenames into structured metadata
# Track unique acq values and which subjects have them
acq_to_subjects = defaultdict(set)
subject_to_acqs = defaultdict(set)

for f in rest_files:
    fname = os.path.basename(f).replace("_correlations.tsv", "")
    parts = {p.split("-", 1)[0]: p.split("-", 1)[1] for p in fname.split("_") if "-" in p}
    sub = parts.get("sub", "")
    acq = parts.get("acq", "UNKNOWN")
    atlas = parts.get("atlas", "")

    # Only count once per subject per acq (not per atlas)
    if atlas == "CC200":
        acq_to_subjects[acq].add(sub)
        subject_to_acqs[sub].add(acq)

print("\n--- Acquisition protocols (CC200 atlas, task-rest, ses-BAS1) ---")
print(f"{'ACQ':<75} {'N subjects':>10}")
print("-" * 87)
for acq, subs in sorted(acq_to_subjects.items(), key=lambda x: -len(x[1])):
    print(f"{acq:<75} {len(subs):>10}")

print(f"\nTotal unique subjects with any rest acq: {len(subject_to_acqs)}")

# How many subjects have ONLY one acq, vs multiple
only_one = sum(1 for acqs in subject_to_acqs.values() if len(acqs) == 1)
multiple  = sum(1 for acqs in subject_to_acqs.values() if len(acqs) > 1)
print(f"Subjects with exactly 1 acq protocol: {only_one}")
print(f"Subjects with multiple acq protocols:  {multiple}")

# What combinations exist?
print("\n--- ACQ combinations per subject ---")
combo_counts = defaultdict(int)
for acqs in subject_to_acqs.values():
    combo_counts[frozenset(acqs)] += 1
for combo, count in sorted(combo_counts.items(), key=lambda x: -x[1]):
    print(f"  {count:>4} subjects: {sorted(combo)}")

# Which subjects only have VARIANT (no plain acq-645)?
has_645         = {s for s, acqs in subject_to_acqs.items() if "645" in acqs}
has_645_variant = {s for s, acqs in subject_to_acqs.items() if any("645VARIANT" in a for a in acqs)}
only_variant    = has_645_variant - has_645

print(f"\nSubjects with plain acq-645:           {len(has_645)}")
print(f"Subjects with any 645VARIANT:          {len(has_645_variant)}")
print(f"Subjects with ONLY 645VARIANT (no plain 645): {len(only_variant)}")
if only_variant:
    print(f"  Example: {list(only_variant)[:5]}")
