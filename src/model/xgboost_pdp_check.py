"""
Chart 11: does the XGBoost model find the same stepped (tiered) shapes in
TermInMonths and GuaranteePct that the WOE bin check found independently
(src/features/woe_monotonicity_check.py)?

The chart is a "partial dependence" plot. It is a SIMULATED view of what a
trained model does with one feature. It is not real default rates from the
data.

How it is built, for TermInMonths (GuaranteePct works the same way):
1. Train an XGBoost model on the training loans, using the raw term and all
   the other features.
2. Take every loan and set its term to 36 months, leaving everything else
   about that loan as it is. Such loans may not exist in reality.
3. The model predicts a default chance for each of those loans. The average
   is one point on the chart.
4. Repeat for 37 months, 38 months and so on. The line joins those averages.

How to read it:
- x-axis: the value the feature was set to.
- y-axis: the model's average predicted chance of default (0.25 = 25%).
- The shape shows how the model's prediction moves as that one feature
  changes, everything else held constant.
- The small ticks along the bottom mark where most loans actually sit.

The model here uses the raw term, not the WOE-smoothed version used in the
reported XGBoost comparison, so it reacts sharply at exact term lengths
(60, 84 months and so on). This run is for shape inspection only, not a
performance claim. The reported comparison smooths TermInMonths because the raw-term model's AUC is inflated by the same
window-entanglement artifact this chart partly shows.

Companion chart, 11_modified_observed_vs_model.png (the chart above is kept
as it was): for the common term lengths and the four SBA guarantee tiers,
bars show the REAL default rate among training loans with exactly that term
or tier, and dots show the model's simulated average at the same value.
Where a dot sits close to its bar, the model agrees with the data. A gap
does not by itself mean the model is wrong: the dots describe a simulated
set of loans (every loan set to that one value, everything else unchanged),
which can differ from the real loans that have that value.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_logistic import prepare_population, time_based_split, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
from train_xgboost import prepare_xgb_frame

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.inspection import PartialDependenceDisplay

OUT = str(Path(__file__).resolve().parents[2] / "outputs" / "figures" / "pd_model")


def chart_observed_vs_model(model, X_train, train):
    """Real default rate (bars) vs. the model's simulated average (dots), at
    the common term lengths and the four guarantee tiers."""
    panels = [
        ("TermInMonths", [36, 48, 60, 72, 84, 96, 120, 180, 240, 300],
         ["3 yrs", "4 yrs", "5 yrs", "6 yrs", "7 yrs", "8 yrs", "10 yrs", "15 yrs", "20 yrs", "25 yrs"],
         "Loan term", "loan term"),
        ("GuaranteePct", [50, 75, 85, 90], ["50%", "75%", "85%", "90%"],
         "SBA guarantee share", "share of the loan guaranteed by SBA"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (col, values, labels, title, xlabel) in zip(axes, panels):
        observed, simulated, counts = [], [], []
        for v in values:
            at_value = train[col].round(0) == v
            observed.append(train.loc[at_value, TARGET].mean() * 100)
            counts.append(int(at_value.sum()))
            X_v = X_train.copy()
            X_v[col] = float(v)
            simulated.append(model.predict_proba(X_v)[:, 1].mean() * 100)
        labels_n = [f"{lab}\nn={n:,}" for lab, n in zip(labels, counts)]
        ax.bar(labels_n, observed, color="#4C72B0", label="real default rate")
        ax.scatter(labels_n, simulated, color="black", marker="D", s=40, zorder=3,
                   label="model, simulated average")
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("chance of default within 5 years (%)")
        ax.legend(fontsize=8)
    fig.suptitle("Real default rates vs. what the XGBoost model simulates, at the common values")
    fig.text(0.5, 0.005, "Bars: real loans with exactly this value. Dots: the model's average prediction if every loan "
             "were set to this value. A gap is a prompt to look closer, not proof the model is wrong.",
             ha="center", fontsize=8, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{OUT}/11_modified_observed_vs_model.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/11_modified_observed_vs_model.png")


def main():
    df = prepare_population(impute_term_zero=False)
    train, test = time_based_split(df)

    X_train, train_categories = prepare_xgb_frame(train)  # all raw, including TermInMonths
    y_train = train[TARGET]

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        enable_categorical=True, eval_metric="auc", random_state=42,
    )
    model.fit(X_train, y_train)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    PartialDependenceDisplay.from_estimator(
        model, X_train, features=["TermInMonths"], ax=axes[0], target=1,
    )
    axes[0].set_title("Partial dependence: TermInMonths (raw)")
    PartialDependenceDisplay.from_estimator(
        model, X_train, features=["GuaranteePct"], ax=axes[1], target=1,
    )
    axes[1].set_title("Partial dependence: GuaranteePct (raw)")
    fig.suptitle("XGBoost partial dependence -- does the tree find the same tiered shapes WOE found?")
    fig.tight_layout()

    Path(OUT).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}/11_xgboost_partial_dependence.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/11_xgboost_partial_dependence.png")

    chart_observed_vs_model(model, X_train, train)


if __name__ == "__main__":
    main()
