
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def save_figure(fig, path):
    """Save a figure with consistent settings."""
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")


def plot_target_distribution(df, target="condition"):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x=target, ax=ax)
    ax.set_title("Heart disease class distribution")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Patients")
    return fig, ax


def plot_numeric_by_condition(df, column, target="condition"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x=target, y=column, ax=ax)
    sns.stripplot(data=df, x=target, y=column, ax=ax, color="black", alpha=0.35, size=3)
    ax.set_title(f"{column} by heart disease condition")
    ax.set_xlabel("Condition")
    return fig, ax


def plot_numeric_histograms(df, columns, target="condition"):
    n_cols = 2
    n_rows = (len(columns) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))
    axes = axes.ravel()

    for i, column in enumerate(columns):
        sns.histplot(data=df, x=column, hue=target, kde=True, bins=20, ax=axes[i], alpha=0.45)
        axes[i].set_title(f"Distribution of {column}")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    return fig, axes


def plot_categorical_proportions(df, columns, target="condition"):
    n_cols = 2
    n_rows = (len(columns) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 4 * n_rows))
    axes = axes.ravel()

    for i, column in enumerate(columns):
        prop = df.groupby([column, target]).size().reset_index(name="count")
        prop["proportion"] = prop["count"] / prop.groupby(column)["count"].transform("sum")
        sns.barplot(data=prop, x=column, y="proportion", hue=target, ax=axes[i])
        axes[i].set_ylim(0, 1)
        axes[i].set_title(f"Condition proportion by {column}")
        axes[i].set_ylabel("Proportion")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    return fig, axes


def plot_correlation_heatmap(df, target="condition"):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Correlation matrix of encoded variables")
    return fig, ax


def plot_target_correlations(df, target="condition"):
    corr = df.corr(numeric_only=True)[target].drop(target).sort_values(key=lambda s: s.abs(), ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=corr.values, y=corr.index, ax=ax)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Correlation with heart disease condition")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Feature")
    return fig, ax


def plot_stage_sizes(stage_features):
    names = list(stage_features.keys())
    sizes = [len(v) for v in stage_features.values()]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=sizes, y=names, ax=ax)
    ax.set_title("Number of variables available at each clinical stage")
    ax.set_xlabel("Number of features")
    ax.set_ylabel("")
    return fig, ax


def plot_cv_auc_by_stage(cv_results):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=cv_results, x="stage", y="roc_auc", hue="model", ax=ax)
    ax.set_title("Nested CV ROC-AUC by clinical stage")
    ax.set_xlabel("")
    ax.set_ylabel("ROC-AUC")
    ax.tick_params(axis="x", rotation=25)
    return fig, ax


def plot_cv_auc_lines(cv_results):
    summary = cv_results.groupby(["stage", "model"], as_index=False)["roc_auc"].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=summary, x="stage", y="roc_auc", hue="model", marker="o", ax=ax)
    ax.set_title("Mean nested CV ROC-AUC across clinical stages")
    ax.set_xlabel("")
    ax.set_ylabel("Mean ROC-AUC")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0.5, 1.0)
    return fig, ax


def plot_threshold_tradeoff(threshold_table, title="Threshold trade-off"):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(threshold_table["threshold"], threshold_table["sensitivity"], marker="o", label="Sensitivity")
    ax.plot(threshold_table["threshold"], threshold_table["specificity"], marker="o", label="Specificity")
    ax.plot(threshold_table["threshold"], threshold_table["precision"], marker="o", label="Precision")
    ax.set_title(title)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax


def stage_filename(stage_name):
    """Create a simple filename from a clinical stage name."""
    stage_id = stage_name.split(" - ")[0].lower().replace(" ", "_")
    return stage_id.replace("/", "_")


def plot_confusion_matrix_from_scores(y_true, y_score, threshold, title):
    y_pred = (y_score >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["No disease", "Disease"],
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    return fig, ax


def plot_roc_curve_from_scores(y_true, y_score, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_roc_curves_by_stage(stage_score_rows):
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in stage_score_rows:
        label = f"{row['stage']} (AUC={row['roc_auc']:.3f})"
        RocCurveDisplay.from_predictions(row["y_true"], row["y_score"], name=label, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("Final test ROC curves by clinical stage")
    ax.grid(True, alpha=0.3)
    return fig, ax

