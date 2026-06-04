from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import SEED
from .preprocessing import build_preprocessor


def build_pipeline(stage_features, classifier):
    """Create the complete preprocessing + classifier pipeline."""
    return Pipeline([
        ("preprocessor", build_preprocessor(stage_features)),
        ("clf", classifier),
    ])


def candidate_models():
    """Return simple, interpretable candidate classifiers and their grids."""
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=SEED),
            "param_grid": {
                "clf__C": [0.01, 0.1, 1, 10],
                "clf__solver": ["liblinear"],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=SEED),
            "param_grid": {
                "clf__n_estimators": [100],
                "clf__max_depth": [None, 5],
                "clf__min_samples_leaf": [1, 5],
            },
        },
    }

