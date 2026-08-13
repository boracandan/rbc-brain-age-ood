# Rethinking Graph Neural Networks for Out-of-distribution Generalization

**Authors:** Bora Candan 
**Paper:** [Link to paper / arXiv]

## Abstract

Machine learning models, including the recently popularized graph neural networks (GNNs), have been extensively adapted to the neuroimaging setting under the independent and identically distributed assumption. However, the systematic comparative testing of traditional machine learning baselines against vanilla GNN models in the out-of-distribution setting remains largely understudied. In this work, we systematically compare two vanilla GNN architectures to three traditional machine learning models, including single- and multimodal variants that merge fMRI and sMRI data, in terms of their out-of-distribution performance. Specifically, we evaluate each variant's performance using a leave-one-study-out evaluation across the five studies in the Reproducible Brain Charts dataset. Through such analysis, we demonstrate how traditional baselines matched or exceeded vanilla GNNs' performance under distribution shift despite greater architectural complexity, even at substantially larger parameter counts, with the strongest fMRI-only baseline (Ridge Regression, test Pearson correlation 0.582) outperforming both fMRI-only GNN models (0.495 and 0.450); and fusing sMRI improved the performance of baselines while hurting the GNNs. Our findings highlight the persistence of in-distribution criticisms of GNNs into the out-of-distribution case for the neuroimaging task of age regression.

## Project Structure

```
rbc-brain-age-ood/
│
├── src/                                  # Core source code
│   ├── gnn.py                            # GNN layer definitions (VanilleGCN, VanillaGAT)
│   ├── model.py                          # fMRINet model (GNN backbone + regression head)
│   ├── Training_multimodality.py         # Training script (single trial, called by runner)
│   ├── Testing_multimodality.py          # Testing script (evaluates a trained trial)
│   ├── runner.py                         # LOO experiment orchestrator (launches training)
│   ├── run_testing.py                    # Runs testing for top-K trials per LOO split
│   ├── smri_fmri_mapping_check.py        # Validates subject overlap between fMRI and sMRI data
│   └── pre_processing_util.py            # Preprocessing utilities (FC matrix loading, normalization)
│
├── helper_scripts/                       # Operational utilities
│   ├── compute_val_metrics.py            # Computes val r/MAE from saved .npy predictions
│   ├── downloading_data.ipynb            # DataLad download notebook for all study datasets
│   ├── acq_coverage.py                   # Checks acquisition protocol coverage across NKI subjects
│   └── explore_acq_params.py             # Explores acquisition parameter distributions in NKI
│
├── Figures/                              # Output figures
│   ├── test_r_heatmap.pdf                # Test-set Pearson r heatmap (LOO splits × architectures)
│   └── delta_r_heatmap.pdf               # Δr heatmap (val r − test r generalization gap)
│
├── analysis_of_results.ipynb             # Main results analysis and figure generation
├── baseline_OOD_training.ipynb           # Baseline model (Ridge/XGBoost) training and evaluation
├── study_splitting.ipynb                 # Subject-level train/val/test split construction
├── saved_variables.json                  # Serialized LOO subject splits (used by download script)
│
└── .gitignore
```

## Data

All datasets are part of the [Reproducible Brain Charts (RBC)](https://reproduciblebrainchart.github.io/) initiative and are publicly available via [DataLad](https://www.datalad.org/) through FCP-INDI. The five studies used are:

| Study | Modalities |
|-------|-----------|
| NKI Rockland Sample (NKI) | fMRI, sMRI |
| Brain Health Research Cohort (BHRC) | fMRI, sMRI |
| Healthy Brain Network (HBN) | fMRI, sMRI |
| Chinese Color Nest Project (CCNP) | fMRI, sMRI |
| Philadelphia Neurodevelopmental Cohort (PNC) | fMRI, sMRI |

See `helper_scripts/downloading_data.ipynb` for the DataLad download workflow. Subject-level train/val/test splits are pre-computed in `saved_variables.json`.

## Requirements

- Python ≥ 3.9
- [PyTorch](https://pytorch.org/) ≥ 1.12
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- scikit-learn
- scipy
- pandas
- numpy
- datalad *(for data download only)*
- tqdm
- matplotlib

Install the core dependencies:

```bash
pip install torch torch-geometric scikit-learn scipy pandas numpy tqdm matplotlib
```

## Usage

**1. Download the data**

Run `helper_scripts/downloading_data.ipynb` cell by cell for each study. Requires DataLad and access to FCP-INDI.

**2. Construct subject splits**

Run `study_splitting.ipynb` to build and inspect the train/val/test splits per LOO fold. The output is already saved in `saved_variables.json`.

**3. Train GNN models (leave-one-study-out)**

```bash
# Replace NKI with BHRC, CCNP, HBN, or PNC to run each LOO fold
python src/runner.py --test_set NKI

# Run only specific architectures (gcn_fmri, gcn_multi, gat_fmri, gat_multi)
python src/runner.py --test_set NKI --trial gcn_fmri gat_fmri

# Disable degree normalization for GCN variants
python src/runner.py --test_set NKI --no-degree_norm
```

**4. Run testing on top-K trials**

```bash
python src/run_testing.py
```

This evaluates the top-5 trials per architecture group per LOO split, ranked by validation loss.

**5. Compute validation metrics**

```bash
python helper_scripts/compute_val_metrics.py
```

Reads saved `.npy` predictions and writes `best_val_r` / `best_val_mae` back into each split's `results_table.csv`.

**6. Analyse results**

Open `analysis_of_results.ipynb` for result tables and figure generation (test-r heatmap, Δr heatmap). Baseline model training and evaluation is in `baseline_OOD_training.ipynb`.

## Citation

If you use this code, please cite:

```bibtex
@article{[citekey],
  title   = {Rethinking Graph Neural Networks for Out-of-distribution Generalization},
  author  = {[Authors]},
  journal = {[Journal / Conference]},
  year    = {[Year]},
  url     = {[Link]}
}
```