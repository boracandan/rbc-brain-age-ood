import numpy as np

class _FCStrategy:
    @classmethod
    def _match(cls, s: str) -> bool:
        raise NotImplementedError
    @classmethod
    def _from_str(cls, s: str) -> "_FCStrategy":
        raise NotImplementedError
    def apply(self, _m: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class _ClipZero(_FCStrategy):
    @classmethod
    def _match(cls, s): return s == "clip_zero"
    @classmethod
    def _from_str(cls, s): return cls()
    def apply(self, m): return np.clip(m, 0, None)

class _AbsThreshold(_FCStrategy):
    def __init__(self, threshold: float): self.threshold = threshold
    @classmethod
    def _match(cls, s): return s.startswith("abs")
    @classmethod
    def _from_str(cls, s): return cls(float(s.split("_")[-1]))
    def apply(self, m):
        m = np.abs(m)
        m[m < self.threshold] = 0
        return m

class _Percentile(_FCStrategy):
    def __init__(self, pct: float): self.pct = pct
    @classmethod
    def _match(cls, s): return s.startswith("per")
    @classmethod
    def _from_str(cls, s): return cls(float(s.split("_")[-1]))
    def apply(self, m):
        m = np.abs(m)
        np.fill_diagonal(m, 0)
        cutoff = np.percentile(m[m > 0], 100 - self.pct)
        m[m < cutoff] = 0
        np.fill_diagonal(m, 1)
        return m

# class _Percentile(_FCStrategy):
#     def __init__(self, pct: float): self.pct = pct
#     @classmethod
#     def _match(cls, s): return s.startswith("per")
#     @classmethod
#     def _from_str(cls, s): return cls(float(s.split("_")[-1]))
#     def apply(self, m):
#         m = np.clip(m, a_min=0, a_max=None)
#         np.fill_diagonal(m, 0)
#         cutoff = np.percentile(m[m > 0], 100 - self.pct)
#         m[m < cutoff] = 0
#         np.fill_diagonal(m, 1)
#         return m

_FC_STRATEGIES: list[type[_FCStrategy]] = [_ClipZero, _AbsThreshold, _Percentile]

def _get_fc_strategy(fc_processing: str) -> _FCStrategy:
    for cls in _FC_STRATEGIES:
        if cls._match(fc_processing):
            return cls._from_str(fc_processing)
    raise ValueError(f"Unknown fc_processing: {fc_processing!r}")

def pre_process_fMRI(fMRI_matrix: np.ndarray, config: dict) -> np.ndarray:
    return _get_fc_strategy(config["fc_processing"]).apply(fMRI_matrix)

# Winsorization caps per training split, derived from p99.9 of that split's data only.
# HBN drives the high outliers for SurfArea, GausCurv, and FoldInd.
# Add a new entry whenever a new train_datasets value is introduced.
_WINSOR_CAPS_BY_SPLIT = {
    # Main OOD experiment
    "NKItrimmed_BHRC_HBN_CCNP": {"SurfArea": (None, 1843.0),  "GausCurv": (None, 0.094), "FoldInd": (1, 79)},
    "NKItrimmed_BHRC_HBN_PNC": {"SurfArea": (None, 1866.0),  "GausCurv": (None, 0.092), "FoldInd": (1, 74.101)},
    "NKItrimmed_BHRC_CCNP_PNC": {"SurfArea": (None, 1839.0),  "GausCurv": (None, 0.085), "FoldInd": (1, 53)},
    "NKItrimmed_HBN_CCNP_PNC": {"SurfArea": (None, 1871.0),  "GausCurv": (None, 0.092), "FoldInd": (1, 76)},
    "PNC_BHRC_HBN_CCNP": {"SurfArea": (None, 1867.0),  "GausCurv": (None, 0.092), "FoldInd": (1, 76)},
}

def _get_winsor_caps(train_datasets):
    if train_datasets in _WINSOR_CAPS_BY_SPLIT:
        return _WINSOR_CAPS_BY_SPLIT[train_datasets]
    # Fallback for unlisted splits: use HBN caps if HBN is present
    return ({"SurfArea": (None, 1866.0),  "GausCurv": (None, 0.092), "FoldInd": (1, 74.101)}
            if "HBN" in train_datasets.split("_")
            else {"SurfArea": (None, 1839.0),  "GausCurv": (None, 0.085), "FoldInd": (1, 53)})


def winsorize(df, caps):
    df = df.copy()
    for col, cap in caps.items():
        if cap is not None and col in df.columns:
            df[col] = df[col].clip(upper=cap[1], lower=cap[0])
    return df


def pre_process_sMRI(reg_surfs, SMRI_FEATURES, SMRI_ATLAS_NAMES, resolution, config):
    reg_surfs = reg_surfs[reg_surfs["atlas"] == SMRI_ATLAS_NAMES[resolution]]
    reg_surfs = reg_surfs[reg_surfs["StructName"].str.startswith("17Networks")]   # drop background rows
    reg_surfs = reg_surfs.sort_values(["hemisphere", "Index"])                    # LH then RH, matching FC column order

    reg_surfs = winsorize(reg_surfs, _get_winsor_caps(config["train_datasets"]))

    feat_mat = reg_surfs[SMRI_FEATURES].values.astype(np.float32)

    return feat_mat