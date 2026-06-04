# Clinically staged heart disease prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SergioPascualRedondo/Proyect_SP_AI_FOR_MEDICINE/blob/main/notebooks/heart_disease_clinical_stages.ipynb)

This project studies the Cleveland Heart Disease dataset as a clinically staged machine learning problem.

The main question is not only whether a model can predict heart disease, but how prediction performance changes when progressively more specialised clinical information is available:

1. Routine cardiovascular risk factors
2. Symptoms and resting ECG
3. Exercise stress test variables
4. Advanced diagnostic measurements

The project follows the AI for Medicine course principles: study the medical issue first, build a reproducible pipeline, avoid data leakage, and interpret results in clinical context.

## Structure

```text
data/raw/                         Original CSV used by the project
notebooks/heart_disease_clinical_stages.ipynb
src/config.py                      Global constants and clinical feature groups
src/data.py                        Dataset loading and data quality checks
src/preprocessing.py               Leakage-safe preprocessing with ColumnTransformer
src/models.py                      Candidate classifiers and hyperparameter grids
src/validation.py                  Train/test split and nested cross-validation
src/evaluation.py                  Metrics and threshold analysis
src/visualization.py               Matplotlib/seaborn plotting helpers
results/figures/                   Generated figures
results/tables/                    Generated result tables
```

## Methodological principles

- The test set is locked before model selection.
- All preprocessing is fitted inside the training folds only.
- Hyperparameter tuning is performed inside the inner validation loop.
- The outer loop estimates generalization performance.
- Clinical stages are compared using the same splits to make results comparable.
- The notebook contains the full narrative, including medical motivation, dataset inspection, pipeline design, results, limitations, ethics, and reproducibility.

