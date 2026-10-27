"""
PD model, step 4: validation metrics -- KS statistic and a calibration
chart, on top of the AUC/Gini already printed by train_logistic.py.

Reads the saved model artifact rather than refitting, so this can be
re-run any time after train_logistic.py without redoing the fit. If the
model changes, just re-run train_logistic.py then this -- the chart file
gets overwritten in place, nothing to clean up by hand.

KS statistic: how far apart the model can pull the predicted-risk
distributions of loans that actually defaulted vs. loans that didn't --
the bigger the separation, the better the model discriminates. Standard
credit-scoring rule of thumb: <20 weak, 20-40 acceptable/good, >40 strong
(and worth double-checking for leakage if it looks unusually high).

Calibration chart: same idea as the decile table already printed by
train_logistic.py, plotted so predicted vs. actual default rate by risk
decile can be read at a glance instead of row by row.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from woe_binning import build_modeling_population
from train_logistic import time_based_split, transform, TARGET, decile_calibration_table

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_DIR = str(Path(__file__).resolve().parents[2] / "outputs" / "model_artifacts")
OUT = str(Path(__file__).resolve().parents[2] / "outputs" / "figures" / "pd_model")


def compute_ks(y_true, y_pred_proba):
    """Max distance between the predicted-score distributions of actual
    bads vs. actual goods -- the credit-scoring-standard way to compute KS
    for a model (equivalent to the classic cumulative-%-bad vs.
    cumulative-%-good curve, just via the two-sample KS test directly)."""
    scores_bad = y_pred_proba[y_true == 1]
    scores_good = y_pred_proba[y_true == 0]
    result = ks_2samp(scores_bad, scores_good)
    return result.statistic


def plot_calibration(table, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    x = table.index.astype(int)
    ax.plot(x, table["predicted_avg_pct"], marker="o", label="predicted (avg)", color="#4C72B0")
    ax.plot(x, table["actual_rate_pct"], marker="o", label="actual", color="#DD8452")
    ax.set_xlabel("risk decile (0=safest, 9=riskiest)")
    ax.set_ylabel("default rate %")
    ax.set_title("Calibration: predicted vs. actual default rate by decile (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    artifact = pd.read_pickle(f"{ARTIFACTS_DIR}/logistic_model_v1.pkl")
    model, fitted_bins = artifact["model"], artifact["fitted_bins"]

    df = build_modeling_population()
    train, test = time_based_split(df)
    X_test = transform(test, fitted_bins, artifact["numeric_features"], artifact["categorical_features"],
                        artifact.get("interaction_pairs"), artifact.get("include_term_missing_flag", False))
    y_test = test[TARGET]
    p_test = model.predict_proba(X_test)[:, 1]

    ks = compute_ks(y_test, p_test)
    read = "weak" if ks < 0.20 else "acceptable/good" if ks < 0.40 else "strong (check for leakage if surprising)"
    print(f"KS statistic (test set): {ks:.3f}  -- {read}")

    table = decile_calibration_table(y_test, p_test)
    print("\ncalibration table (test set):")
    print(table.round(2).to_string())

    Path(OUT).mkdir(parents=True, exist_ok=True)
    plot_calibration(table, f"{OUT}/05_calibration_chart.png")
    print(f"\nsaved {OUT}/05_calibration_chart.png")


if __name__ == "__main__":
    main()
