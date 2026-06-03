import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def classification_metrics(y_true, y_score, threshold=0.5):
    """Compute clinically readable binary classification metrics."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred),
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def threshold_metrics_table(y_true, y_score, thresholds=None):
    """Evaluate sensitivity/specificity trade-offs for several thresholds."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 1.0, 0.1), 2)

    rows = [classification_metrics(y_true, y_score, threshold=t) for t in thresholds]
    return pd.DataFrame(rows)
