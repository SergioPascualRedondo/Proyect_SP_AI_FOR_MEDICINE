from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor(stage_features):
    """Build a leakage-safe preprocessor for the selected clinical stage.

    The returned transformer is fitted only inside the training folds when it is
    placed in a scikit-learn Pipeline.
    """
    numeric = [f for f in NUMERIC_FEATURES if f in stage_features]
    binary = [f for f in BINARY_FEATURES if f in stage_features]
    categorical = [f for f in CATEGORICAL_FEATURES if f in stage_features]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("binary", binary_pipe, binary),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")
