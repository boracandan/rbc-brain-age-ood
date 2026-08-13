"""
Check which acquisition protocol covers the most subjects,
and whether any is shared across all participants.
"""

import os
import glob
from collections import defaultdict

dir_path = r"C:\Users\Faruk\Code\rbc-brain-age-ood"

rest_files = glob.glob(
    rf"{dir_path}\NKI_CPAC\cpac_RBCv0\sub-*\ses-BAS1\func\*_task-rest_*_correlations.tsv"
)

# Map each subject to which acq protocols they have (CC200 only to avoid atlas duplication)
subject_to_acqs = defaultdict(set)
all_subjects = set()

for f in rest_files:
    fname = os.path.basename(f).replace("_correlations.tsv", "")
    parts = {p.split("-", 1)[0]: p.split("-", 1)[1] for p in fname.split("_") if "-" in p}
    sub  = parts.get("sub", "")
    acq  = parts.get("acq", "UNKNOWN")
    atlas = parts.get("atlas", "")
    if atlas == "CC200":
        all_subjects.add(sub)
        subject_to_acqs[sub].add(acq)

total = len(all_subjects)
print(f"Total unique subjects (task-rest, BAS1): {total}\n")

# Count coverage per acq (exact match)
acq_coverage = defaultdict(set)
for sub, acqs in subject_to_acqs.items():
    for acq in acqs:
        acq_coverage[acq].add(sub)

print("--- Coverage per acquisition (exact) ---")
for acq, subs in sorted(acq_coverage.items(), key=lambda x: -len(x[1])):
    pct = 100 * len(subs) / total
    print(f"  {acq:<75} {len(subs):>4} / {total}  ({pct:.1f}%)")

# Group by TR family (645, 1400, CAP) ignoring VARIANT
family_coverage = defaultdict(set)
for sub, acqs in subject_to_acqs.items():
    for acq in acqs:
        if acq.startswith("645"):
            family_coverage["645 (any)"].add(sub)
        elif acq.startswith("1400"):
            family_coverage["1400 (any)"].add(sub)
        elif acq.startswith("CAP"):
            family_coverage["CAP (any)"].add(sub)

print("\n--- Coverage by TR family (plain + VARIANT combined) ---")
for fam, subs in sorted(family_coverage.items(), key=lambda x: -len(x[1])):
    pct = 100 * len(subs) / total
    print(f"  {fam:<20} {len(subs):>4} / {total}  ({pct:.1f}%)")

# Is there ANY acq shared by ALL subjects?
universal = [acq for acq, subs in acq_coverage.items() if len(subs) == total]
print(f"\nAcquisitions shared by ALL {total} subjects: {universal if universal else 'NONE'}")

# What is the intersection of subjects that have all three families?
has_645 = family_coverage["645 (any)"]
has_1400 = family_coverage["1400 (any)"]
has_cap  = family_coverage["CAP (any)"]
all_three = has_645 & has_1400 & has_cap
print(f"\nSubjects with all 3 TR families: {len(all_three)} / {total}  ({100*len(all_three)/total:.1f}%)")

# Subjects missing each family
print(f"Missing 645 entirely:  {len(all_subjects - has_645)}")
print(f"Missing 1400 entirely: {len(all_subjects - has_1400)}")
print(f"Missing CAP entirely:  {len(all_subjects - has_cap)}")
