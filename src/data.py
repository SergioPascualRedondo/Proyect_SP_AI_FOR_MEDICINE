import pandas as pd

from .config import DATA_PATH, TARGET


def load_heart_data(path=DATA_PATH):
    """Load the cleaned Cleveland Heart Disease CSV used in this project."""
    return pd.read_csv(path)


def split_features_target(df, features, target=TARGET):
    """Return X and y for a selected clinical stage."""
    return df[features].copy(), df[target].copy()


def data_quality_report(df):
    """Compute basic checks used before any modelling step."""
    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "missing_values": df.isna().sum(),
        "question_mark_values": df.astype(str).eq("?").sum(),
        "empty_string_values": df.astype(str).apply(lambda col: col.str.strip().eq("")).sum(),
        "duplicated_rows": int(df.duplicated().sum()),
        "target_counts": df[TARGET].value_counts().sort_index(),
    }
