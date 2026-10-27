"""
PD model, step 4 (continued): a few extra validation views requested on
review of the first logistic regression run -- ROC curve, precision-recall
curve, accuracy_score
(checked and flagged as not meaningful here: 96.9% accuracy equals always predicting no default), and
default rate by ProcessingMethod on the modeling population (the
60-month-window version of the ProcessingMethod finding from early EDA).

Reads the saved model artifact, does not refit anything.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from woe_binning import build_modeling_population
from train_logistic import time_based_split, transform, TARGET

import pandas as pd
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score, accuracy_score, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_DIR = str(Path(__file__).resolve().parents[2] / "outputs" / "model_artifacts")
OUT = str(Path(__file__).resolve().parents[2] / "outputs" / "figures" / "pd_model")
EDA_OUT = str(Path(__file__).resolve().parents[2] / "outputs" / "figures")  # ProcessingMethod chart
# belongs with the other EDA charts, not the pd_model-only folder


def plot_roc(y_test, p_test):
    fpr, tpr, _ = roc_curve(y_test, p_test)
    auc = roc_auc_score(y_test, p_test)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, color="#4C72B0", linewidth=2, label=f"model (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="random guess (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve, test set (2019-2021 vintages)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/08_roc_curve.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/08_roc_curve.png, AUC={auc:.3f}")


def plot_precision_recall(y_test, p_test):
    """Of the loans the model flags as riskiest, what share really default
    (precision), at each level of how many of the real defaults are caught
    (recall). The dashed line is the share of defaults among all test loans:
    a model with no skill would sit on it. Lift = average precision / that line."""
    precision, recall, _ = precision_recall_curve(y_test, p_test)
    ap = average_precision_score(y_test, p_test)
    base = y_test.mean()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(recall, precision, color="#4C72B0", linewidth=2, label=f"model (average precision={ap:.3f})")
    ax.axhline(base, color="gray", linestyle="--", linewidth=1, label=f"no skill ({base:.1%} of test loans default)")
    ax.set_xlabel("Recall (share of all real defaults caught)")
    ax.set_ylabel("Precision (share of flagged loans that default)")
    ax.set_title("Precision-recall curve, test set (2019-2021 vintages)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/04_precision_recall_curve.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/04_precision_recall_curve.png, average precision={ap:.3f} vs. baseline {base:.3f} (lift {ap / base:.1f}x)")


def check_accuracy(model, X_test, y_test):
    pred_labels = model.predict(X_test)
    acc = accuracy_score(y_test, pred_labels)
    naive_acc = accuracy_score(y_test, [0] * len(y_test))
    print(f"model accuracy (0.5 threshold): {acc:.4f}")
    print(f"naive 'always predict no-default' accuracy: {naive_acc:.4f}")
    print("confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_test, pred_labels))
    print(f"loans flagged as predicted-default: {pred_labels.sum()} out of {len(pred_labels)}")
    print("-- not a metric worth reporting on its own for this model: it matches always predicting no default --")


def plot_default_rate_by_processing_method(df, min_n=200):
    g = df.groupby("ProcessingMethod", observed=True)["target_60mo"].agg(["size", "mean"])
    g = g.rename(columns={"size": "n", "mean": "default_rate"})
    g["default_rate_pct"] = g["default_rate"] * 100
    g = g[g["n"] >= min_n].sort_values("default_rate_pct")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(g.index.astype(str), g["default_rate_pct"], color="#4C72B0")
    overall = df["target_60mo"].mean() * 100
    ax.axvline(overall, color="gray", linestyle="--", linewidth=1, label=f"overall avg ({overall:.1f}%)")
    ax.set_xlabel("default rate % (target_60mo)")
    ax.set_title(f"Default rate by ProcessingMethod, modeling population (n>={min_n})")
    ax.legend()
    fig.tight_layout()
    Path(EDA_OUT).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{EDA_OUT}/36_default_rate_by_processingmethod_60mo.png", dpi=150)
    plt.close(fig)
    print(f"saved {EDA_OUT}/36_default_rate_by_processingmethod_60mo.png")


def main():
    artifact = pd.read_pickle(f"{ARTIFACTS_DIR}/logistic_model_v1.pkl")
    model, fitted_bins = artifact["model"], artifact["fitted_bins"]

    df = build_modeling_population()
    train, test = time_based_split(df)
    X_test = transform(test, fitted_bins, artifact["numeric_features"], artifact["categorical_features"],
                        artifact.get("interaction_pairs"), artifact.get("include_term_missing_flag", False))
    y_test = test[TARGET]
    p_test = model.predict_proba(X_test)[:, 1]

    Path(OUT).mkdir(parents=True, exist_ok=True)
    plot_roc(y_test, p_test)
    plot_precision_recall(y_test, p_test)
    print()
    check_accuracy(model, X_test, y_test)
    print()
    plot_default_rate_by_processing_method(df)


if __name__ == "__main__":
    main()
