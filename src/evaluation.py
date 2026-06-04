import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict


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


def evaluate_selected_strategy(
    train_df,
    test_df,
    selected_features,
    selected_spec,
    build_pipeline,
    split_features_target,
    seed,
    threshold_grid=None,
    report_threshold=0.5,
):
    """Tune the selected model, inspect thresholds and score the locked test set."""
    if threshold_grid is None:
        threshold_grid = np.round(np.arange(0.1, 1.0, 0.1), 2)

    X_dev, y_dev = split_features_target(train_df, selected_features)
    X_test, y_test = split_features_target(test_df, selected_features)

    pipeline = build_pipeline(selected_features, selected_spec["estimator"])
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=selected_spec["param_grid"],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        scoring="roc_auc",
        refit=True,
    )

    oof_score = cross_val_predict(
        search,
        X_dev,
        y_dev,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    dev_threshold_table = threshold_metrics_table(y_dev, oof_score, thresholds=threshold_grid)

    search.fit(X_dev, y_dev)
    y_test_score = search.predict_proba(X_test)[:, 1]
    test_threshold_table = threshold_metrics_table(
        y_test,
        y_test_score,
        thresholds=dev_threshold_table["threshold"].to_numpy(),
    )
    y_test_pred = (y_test_score >= report_threshold).astype(int)

    return {
        "search": search,
        "X_dev": X_dev,
        "y_dev": y_dev,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_score": y_test_score,
        "y_test_pred": y_test_pred,
        "dev_threshold_table": dev_threshold_table,
        "test_threshold_table": test_threshold_table,
        "report_threshold": report_threshold,
    }


def evaluate_stages_on_final_test(
    train_df,
    test_df,
    stage_features,
    model_name,
    model_spec,
    build_pipeline,
    split_features_target,
    seed,
    threshold_grid=None,
):
    """Train and evaluate the same model at each clinical stage."""
    if threshold_grid is None:
        threshold_grid = np.round(np.arange(0.1, 1.0, 0.1), 2)

    threshold_rows = []
    test_rows = []
    score_rows = []

    for stage_name, features in stage_features.items():
        X_dev, y_dev = split_features_target(train_df, features)
        X_test, y_test = split_features_target(test_df, features)

        pipeline = build_pipeline(features, model_spec["estimator"])
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=model_spec["param_grid"],
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
            scoring="roc_auc",
            refit=True,
        )

        oof_score = cross_val_predict(
            search,
            X_dev,
            y_dev,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
            method="predict_proba",
            n_jobs=1,
        )[:, 1]

        dev_thresholds = threshold_metrics_table(y_dev, oof_score, thresholds=threshold_grid)
        dev_thresholds.insert(0, "stage", stage_name)
        threshold_rows.append(dev_thresholds)

        best_threshold_row = dev_thresholds.sort_values(
            ["balanced_accuracy", "sensitivity"],
            ascending=False,
        ).iloc[0]
        selected_threshold = float(best_threshold_row["threshold"])

        search.fit(X_dev, y_dev)
        y_test_score = search.predict_proba(X_test)[:, 1]
        test_metrics = threshold_metrics_table(
            y_test,
            y_test_score,
            thresholds=np.array([selected_threshold]),
        ).iloc[0].to_dict()

        row = {
            "stage": stage_name,
            "model": model_name,
            "selected_threshold_from_development": selected_threshold,
            "development_balanced_accuracy_at_threshold": best_threshold_row["balanced_accuracy"],
            "best_params": str(search.best_params_),
        }
        row.update(test_metrics)
        test_rows.append(row)

        score_rows.append({
            "stage": stage_name,
            "y_true": y_test,
            "y_score": y_test_score,
            "threshold": selected_threshold,
            "roc_auc": test_metrics["roc_auc"],
        })

    return (
        pd.concat(threshold_rows, ignore_index=True),
        pd.DataFrame(test_rows),
        score_rows,
    )
