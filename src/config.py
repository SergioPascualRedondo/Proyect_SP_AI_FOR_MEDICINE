from pathlib import Path

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_cleveland_upload.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
TARGET = "condition"

STAGE_FEATURES = {
    "Stage 1 - Routine risk factors": ["age", "sex", "trestbps", "chol", "fbs"],
    "Stage 2 - Symptoms and resting ECG": ["age", "sex", "trestbps", "chol", "fbs", "cp", "restecg"],
    "Stage 3 - Exercise stress test": [
        "age", "sex", "trestbps", "chol", "fbs", "cp", "restecg",
        "thalach", "exang", "oldpeak", "slope",
    ],
    "Stage 4 - Advanced diagnostic tests": [
        "age", "sex", "trestbps", "chol", "fbs", "cp", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    ],
}

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
