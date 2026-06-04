# How Early Can Heart Disease Be Predicted?
## A Staged Machine Learning Analysis of the Cleveland UCI Dataset

AI for Medicine project using the Cleveland Heart Disease dataset.

The goal is not only to predict the presence of heart disease, but to study how prediction changes when progressively more clinical information becomes available. The dataset is organised into four clinical stages:

1. Routine cardiovascular risk factors
2. Symptoms and resting ECG
3. Exercise stress test variables
4. Advanced diagnostic measurements

This follows the main methodology of the course: first understand the medical problem, then build a reproducible and leakage-safe machine learning pipeline.

## Main Files

```text
notebooks/heart_disease_clinical_stages.ipynb    Main executable notebook
Heart_Disease_Project_Report.docx              Final project report
Heart_Disease_Project_Report.pdf               Final project report as PDF
data/raw/heart_cleveland_upload.csv              Local dataset
src/                                             Helper Python modules
results/figures/                                Saved plots
results/tables/                                 Saved result tables
requirements.txt                                Reproducible Python environment
```

## Method

The project uses a binary target:
- `condition = 0`: no heart disease
- `condition = 1`: heart disease

The modelling workflow uses:
- a locked stratified test set;
- scikit-learn `Pipeline` and `ColumnTransformer`;
- imputation, scaling and one-hot encoding inside the pipeline;
- `GridSearchCV` for hyperparameter tuning;
- nested cross-validation on the development set;
- final evaluation on the locked test set;
- threshold analysis, confusion matrices, ROC curves and SHAP interpretation.

The pipeline is important because preprocessing is fitted only inside the training folds. This reduces data leakage and makes the experiment reproducible.

<img src="heart_disease_staged_pipeline_en (1).svg" width="650">

*Figure 1. Overview of the staged, leakage-safe pipeline, from the Cleveland dataset to final staged evaluation and SHAP interpretation.*

## How to Run

Create an environment and install the pinned dependencies:

```bash
pip install -r requirements.txt
```

Then open and run:

```text
notebooks/heart_disease_clinical_stages.ipynb
```

The notebook is prepared to run in Google Colab from the GitHub repository.

## Main Results

Nested cross-validation shows a staged improvement in ROC-AUC:

| Stage | Logistic Regression ROC-AUC | Random Forest ROC-AUC |
| --- | ---: | ---: |
| Stage 1 - Routine risk factors | 0.658 | 0.692 |
| Stage 2 - Symptoms and resting ECG | 0.806 | 0.801 |
| Stage 3 - Exercise stress test | 0.847 | 0.851 |
| Stage 4 - Advanced diagnostic tests | 0.890 | 0.889 |

The most relevant clinical finding is not simply that Stage 4 performs best. The largest improvement appears from Stage 1 to Stage 2, suggesting that symptoms and resting ECG add important information before more advanced tests are available.

Final test results also support the staged interpretation. Bootstrap 95% confidence intervals are reported because the locked test set contains only 60 patients:

| Stage | Final ROC-AUC (95% CI) | Selected threshold |
| --- | ---: | ---: |
| Stage 1 - Routine risk factors | 0.823 (0.703-0.920) | 0.5 |
| Stage 2 - Symptoms and resting ECG | 0.919 (0.842-0.975) | 0.5 |
| Stage 3 - Exercise stress test | 0.908 (0.826-0.967) | 0.6 |
| Stage 4 - Advanced diagnostic tests | 0.955 (0.900-0.993) | 0.5 |

Stage 1 misses many diseased patients, while Stage 2 substantially improves early detection. Stage 4 reaches the best final performance, but it uses advanced variables that are closer to the final diagnostic process. The confidence intervals overlap between later stages, so small differences should not be overinterpreted.

## Reproducibility

The repository includes:
- fixed random seed;
- pinned package versions;
- saved tables and figures;
- bootstrap confidence intervals for final ROC-AUC;
- source code separated into small helper modules;
- final report and executable notebook.

The local virtual environment, executed notebooks, caches and temporary render files are intentionally ignored by Git.
