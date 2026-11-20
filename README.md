# ptbxl-ecg-classification

Multi-label classification of 12-lead ECGs with deep learning, built in public on the open PTB-XL dataset.

> **Educational project. Not a medical device and not for clinical use.**

## Goal

Train and evaluate a 1D CNN (PyTorch) that predicts diagnostic superclasses from 10-second, 12-lead ECG recordings:

| Code | Meaning |
|------|---------|
| NORM | Normal ECG |
| MI | Myocardial infarction |
| STTC | ST/T changes |
| CD | Conduction disturbance |
| HYP | Hypertrophy |

One ECG can carry several labels at once, so this is a **multi-label** problem.

## Dataset

[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) (PhysioNet), a large public ECG dataset annotated by cardiologists. Please read the dataset page for its license and citation requirements.

The data is **not included** in this repository. Download it from PhysioNet and set `DATA_PATH` at the top of each script.

I use the 100 Hz recordings and the official `strat_fold` column, which splits patients across folds: folds 1-8 for training, 9 for validation, 10 for testing.

## Progress

- [x] **Week 1 - Data exploration:** reading ECGs, understanding P/QRS/T waves, the 12 leads, sampling rate, diagnostic labels, class distribution
- [ ] Week 2 - Data preparation: loading all signals, folds, filtering, normalization
- [ ] Week 3 - Baseline model (classical ML)
- [ ] Weeks 4-5 - 1D CNN in PyTorch
- [ ] Week 6 - Evaluation (AUROC, F1 per class), error analysis, explainability
- [ ] Week 7 - Small demo app
- [ ] Week 8 - Write-up

## Week 1 findings

- Each recording has 1000 samples for 12 leads (10 s at 100 Hz).
- The same heartbeat looks different in each lead (for example, the R peak points down in V1).
- Classes are imbalanced: NORM is the largest class, HYP the smallest.
- Many ECGs carry more than one label.

Figures are in the [`figures/`](figures/) folder.

## Files

| File | Purpose |
|------|---------|
| `01_exploration.py` | Load the metadata, plot ECGs, inspect labels and class counts |
| `prepare_data.py` | Load all signals and labels and save them as `.npy` arrays |

## Setup

```bash
pip install wfdb pandas numpy matplotlib
```

PyTorch will be needed from week 4 (see [pytorch.org](https://pytorch.org/get-started/locally/)).

## Limitations

This is a learning project on a single public dataset. Results will not be validated on external data and must not be used for any medical decision.

## Author

Mohammed Lekchiri - L3 SPI, Université de Lorraine. Background in biomedical sciences. Looking for an alternance in medical AI / medical imaging.
