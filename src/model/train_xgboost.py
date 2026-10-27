"""
PD model: XGBoost comparison against the finalized logistic regression --
the logistic regression vs. tree-based model comparison.

Same modeling population, same time-based split (train ApprovalFY
<=2018, test >2018), same finalized 9-feature list (no RevolverStatus --
CV showed it doesn't help; term=0 left as-is -- ablation showed imputing
it moves nothing) as src/model/train_logistic.py's v1.

Different input format on purpose: XGBoost gets the RAW, cleaned feature
values directly, not the WOE-transformed versions, for every feature
EXCEPT TermInMonths. Unlike logistic regression, XGBoost doesn't need a
univariate pre-transform -- it can discover non-linear shapes and
interactions on its own, directly from raw values, and feeding it WOE
values across the board would throw away exactly the fine-grained cutoffs
it's good at finding.

TermInMonths is the deliberate exception, and it's not a modeling
preference -- it's a correction. A first run with TermInMonths raw
produced an implausible test AUC of 0.953 (vs. 0.803 for logistic
regression) -- investigated and traced to the same window-entanglement
issue already flagged for the logistic model's TermInMonths coefficient
(long-term loans are disproportionately STILL_OPEN, not genuinely
resolved-safe, at the 60-month mark). Raw-valued trees could carve out a
razor-precise cutoff and ride that artifact far harder than the
logistic model's smoothed WOE bins could. Smoothing TermInMonths to its
WOE value (same ~10 bins the logistic model uses) collapses the score to
a more defensible 0.843 (logistic regression: 0.803). This script reports the corrected
version by default; the inflated all-raw version is not reproduced here.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from train_logistic import prepare_population, time_based_split, fit_bins_on_train, transform, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
from woe_binning import apply_woe_numeric

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

ARTIFACTS_DIR = str(Path(__file__).resolve().parents[2] / "outputs" / "model_artifacts")


def prepare_xgb_frame(df, categories=None):
    """Raw values, not WOE -- numeric columns as float, categorical
    columns as pandas 'category' dtype so XGBoost can split on them
    natively (enable_categorical=True). `categories`: {col: fitted
    categories from TRAIN}, passed when preparing the test frame so any
    category only seen in test becomes missing (NaN) rather than an
    unrecognized value XGBoost has never encoded -- same "fit on train
    only" discipline used for the WOE bins elsewhere in this project."""
    X = df[NUMERIC_FEATURES].astype(float).copy()
    fitted_categories = {}
    for col in CATEGORICAL_FEATURES:
        cats = categories[col] if categories else None
        X[col] = pd.Categorical(df[col], categories=cats) if cats is not None else df[col].astype("category") # The allowed categories in a cat column.
        fitted_categories[col] = X[col].cat.categories
    return X, fitted_categories


def main():
    df = prepare_population(impute_term_zero=False)
    train, test = time_based_split(df)
    print(f"train (ApprovalFY <= 2018): {len(train)} rows, default rate {train[TARGET].mean()*100:.2f}%")
    print(f"test  (ApprovalFY >  2018): {len(test)} rows, default rate {test[TARGET].mean()*100:.2f}%")

    X_train, train_categories = prepare_xgb_frame(train)
    X_test, _ = prepare_xgb_frame(test, categories=train_categories)

    # correction: TermInMonths smoothed to its WOE value (fit on train only),
    # every other feature stays raw -- see module docstring for why
    term_bins = fit_bins_on_train(train, ["TermInMonths"], [])["TermInMonths"]
    X_train["TermInMonths"] = apply_woe_numeric(train["TermInMonths"], term_bins)
    X_test["TermInMonths"] = apply_woe_numeric(test["TermInMonths"], term_bins)

    y_train, y_test = train[TARGET], test[TARGET]

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        enable_categorical=True, eval_metric="auc", random_state=42,
    )
    model.fit(X_train, y_train)

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    p_test = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, p_test)
    ks = ks_2samp(p_test[y_test == 1], p_test[y_test == 0]).statistic

    print(f"\n=== XGBoost results ===")
    print(f"train AUC: {train_auc:.3f}")
    print(f"test  AUC: {test_auc:.3f}")
    print(f"test  KS:  {ks:.3f}")

    # logistic side of the comparison: read from the saved v1 model, so run
    # train_logistic.py first -- the numbers are never typed in by hand
    lr = pd.read_pickle(f"{ARTIFACTS_DIR}/logistic_model_v1.pkl")
    X_lr = transform(test, lr["fitted_bins"], lr["numeric_features"], lr["categorical_features"])
    p_lr = lr["model"].predict_proba(X_lr)[:, 1]
    lr_auc = roc_auc_score(y_test, p_lr)
    lr_ks = ks_2samp(p_lr[y_test == 1], p_lr[y_test == 0]).statistic

    print(f"\n=== comparison: XGBoost (corrected) vs. logistic regression v1 (same split, same features) ===")
    print(f"{'metric':<12}{'logistic':>12}{'xgboost':>12}")
    print(f"{'test AUC':<12}{lr_auc:>12.3f}{test_auc:>12.3f}")
    print(f"{'test KS':<12}{lr_ks:>12.3f}{ks:>12.3f}")

    print(f"\n=== feature importance (gain-based) ===")
    importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(importance.round(3).to_string())

    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"model": model, "numeric_features": NUMERIC_FEATURES,
                  "categorical_features": CATEGORICAL_FEATURES},
                 f"{ARTIFACTS_DIR}/xgboost_model_v1.pkl")
    print(f"\nsaved {ARTIFACTS_DIR}/xgboost_model_v1.pkl")


if __name__ == "__main__":
    main()
