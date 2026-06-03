import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_validate, train_test_split

from .config import SEED, TARGET
from .data import split_features_target
from .models import build_pipeline


def make_locked_split(df, test_size=0.2):
    """Create one stratified train/test split before any model selection."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[TARGET],
        random_state=SEED,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def nested_cv_stage(train_df, stage_name, stage_features, model_name, model_spec, n_repeats=5):
    """Evaluate one model and one clinical stage using repeated nested CV."""
    X, y = split_features_target(train_df, stage_features)
    pipeline = build_pipeline(stage_features, model_spec["estimator"])

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=model_spec["param_grid"],
        cv=inner_cv,
        scoring="roc_auc",
        refit=True,
    )
    scores = cross_validate(
        search,
        X,
        y,
        cv=outer_cv,
        scoring=["roc_auc", "accuracy", "balanced_accuracy"],
        return_train_score=False,
        n_jobs=-1,
    )

    rows = []
    for i, auc in enumerate(scores["test_roc_auc"]):
        rows.append({
            "stage": stage_name,
            "model": model_name,
            "fold": i,
            "roc_auc": auc,
            "accuracy": scores["test_accuracy"][i],
            "balanced_accuracy": scores["test_balanced_accuracy"][i],
        })
    return pd.DataFrame(rows)
