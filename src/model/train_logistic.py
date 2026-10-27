"""
PD model, step 3: logistic regression.

This file builds the model that actually ships. It's called "v1" only to
tell it apart from the one-off comparison runs in run_ablation.py -- it
is NOT version 1 of a series; nothing here has ever replaced it. Kept
deliberately simple: no cross-validation, term=0 loans left untouched,
and a 9-feature list chosen after checking each feature's individual
predictive strength (an IV review of 13 candidates). Fit once,
understand it, before adding any complexity.

Split: time-based, not random -- train on ApprovalFY 2010-2018, test on
2019-2021. A random split would let the model "see" loans from the same
vintage-cycle it's later tested on, overstating how well it'd really do on
loans it hasn't encountered yet. This is a real out-of-time test.

WOE bins are fit on the TRAINING split only, then applied unchanged to the
test split (src.features.woe_binning.apply_woe_numeric/categorical) --
never re-fit bins on data the model is about to be evaluated on, that
would leak test information into the transform.

v1 feature list, decided after the IV review:
  numeric:     TermInMonths, GrossApproval, InitialInterestRate,
               GuaranteePct, geo_risk_pct
  categorical: ProcessingMethod, CollateralInd, NaicsSectorName,
               BusinessAgeBucket
Left out of v1: SoldSecMrktInd, FixedorVariableInterestInd, JobsSupported
(all weak/redundant). RevolverStatus and ApprovalFY also left out of v1 --
see run_ablation.py for the pass-2 variants that add these back in, one
change at a time.

Testing changes, one at a time: run_ablation.py reuses run_variant()
below to try small changes against v1 -- adding back a dropped feature,
imputing term=0, etc. Calling run_variant() with no arguments always
reproduces v1 exactly, which is what makes these fair, apples-to-apples
comparisons. Each test is named "v2_<what changed>" (e.g.
v2_term_imputed) purely as a label for that one experiment -- these are
NOT a new model version, and none of them replace v1 unless a comparison
shows a real improvement and that's explicitly decided.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRAIN_YEARS_MAX
from woe_binning import (
    build_modeling_population, bin_numeric, bin_categorical,
    apply_woe_numeric, apply_woe_categorical,
)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

ARTIFACTS_DIR = str(Path(__file__).resolve().parents[2] / "outputs" / "model_artifacts")

NUMERIC_FEATURES = ["TermInMonths", "GrossApproval", "InitialInterestRate",
                     "GuaranteePct", "geo_risk_pct"]
CATEGORICAL_FEATURES = ["ProcessingMethod", "CollateralInd", "NaicsSectorName",
                         "BusinessAgeBucket"]
TARGET = "target_60mo"
TERM_TYPICAL_MONTHS = 84  # SBA Express typical term, used only if impute_term_zero=True


def prepare_population(impute_term_zero=False):
    """v1 population, with one optional change: term=0 rows get their term
    replaced with the typical SBA Express value, plus a term_was_missing
    flag (1 for the imputed rows, 0 otherwise) so the fact that they were
    originally unrecorded isn't silently lost."""
    df = build_modeling_population()
    df["term_was_missing"] = 0
    if impute_term_zero:
        is_zero = df["TermInMonths"] == 0
        df.loc[is_zero, "TermInMonths"] = TERM_TYPICAL_MONTHS
        df.loc[is_zero, "term_was_missing"] = 1
    return df


def time_based_split(df, train_years_max=TRAIN_YEARS_MAX):
    train = df[df["ApprovalFY"] <= train_years_max].copy()
    test = df[df["ApprovalFY"] > train_years_max].copy()
    return train, test


def fit_bins_on_train(train, numeric_features, categorical_features):
    """Fit WOE bins on the training split only. Returns {col: bin_table}."""
    fitted = {}
    for col in numeric_features:
        fitted[col] = bin_numeric(train, col)
    for col in categorical_features:
        fitted[col] = bin_categorical(train, col)
    return fitted


def transform(df, fitted_bins, numeric_features, categorical_features,
              interaction_pairs=None, include_term_missing_flag=False):
    """Applies already-fitted bins to any dataframe, returns a DataFrame of
    WOE columns ready for the model. interaction_pairs: list of
    (feature_a, feature_b) tuples -- both must already be in
    numeric/categorical_features -- adds one new column per pair, the
    product of their two WOE columns, which lets the model see whether one feature's effect depends on the other."""
    out = pd.DataFrame(index=df.index)
    for col in numeric_features:
        out[col] = apply_woe_numeric(df[col], fitted_bins[col])
    for col in categorical_features:
        out[col] = apply_woe_categorical(df[col], fitted_bins[col])
    for a, b in (interaction_pairs or []):
        out[f"{a}_x_{b}"] = out[a] * out[b]
    if include_term_missing_flag:
        out["term_was_missing"] = df["term_was_missing"].values
    return out


def print_coefficients(model, feature_names):
    print("=== fitted coefficients (expect ALL negative -- higher WOE = safer bin, "
          "should lower predicted default probability) ===")
    coefs = pd.Series(model.coef_[0], index=feature_names).sort_values()
    for name, coef in coefs.items():
        flag = "" if coef < 0 else "  <-- unexpected sign, check this"
        print(f"  {name:<22} {coef:+.4f}{flag}")
    print(f"  {'intercept':<22} {model.intercept_[0]:+.4f}")


def decile_calibration_table(y_true, y_pred_proba):
    df = pd.DataFrame({"y": y_true, "p": y_pred_proba})
    df["decile"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
    table = df.groupby("decile").agg(
        n=("y", "size"), predicted_avg_pct=("p", lambda s: s.mean() * 100),
        actual_rate_pct=("y", lambda s: s.mean() * 100),
    )
    return table

# for runs with any changes, good idea to assign `name` variable with an appropriate string
# class weight = None - treats every loan equally. Used for penalizing mistakes
def run_variant(name="v1", numeric_features=None, categorical_features=None,
                 class_weight=None, impute_term_zero=False, interaction_pairs=None,
                 train_years_max=TRAIN_YEARS_MAX, verbose=True):
    """
    Runs one full fit -- population -> split -> fit bins on train -> WOE
    transform -> logistic regression -- and returns a one-row scorecard
    (test AUC, test KS, # of unexpected-sign coefficients, the full
    coefficient table) for comparing against v1 or any other variant.
    Every argument defaults to exactly v1's settings, so run_variant() with
    no arguments reproduces v1. Saves its own artifact
    (logistic_model_{name}.pkl) -- never overwrites another variant's.
    """
    numeric_features = numeric_features if numeric_features is not None else list(NUMERIC_FEATURES)
    categorical_features = categorical_features if categorical_features is not None else list(CATEGORICAL_FEATURES)
    include_term_missing_flag = impute_term_zero # default is false

    df = prepare_population(impute_term_zero=impute_term_zero)
    train, test = time_based_split(df, train_years_max)
    if verbose:
        print(f"[{name}] train (ApprovalFY <= {train_years_max}): {len(train)} rows, "
              f"default rate {train[TARGET].mean() * 100:.2f}%")
        print(f"[{name}] test  (ApprovalFY >  {train_years_max}): {len(test)} rows, "
              f"default rate {test[TARGET].mean() * 100:.2f}%")

    fitted_bins = fit_bins_on_train(train, numeric_features, categorical_features)
    X_train = transform(train, fitted_bins, numeric_features, categorical_features,
                         interaction_pairs, include_term_missing_flag)
    X_test = transform(test, fitted_bins, numeric_features, categorical_features,
                        interaction_pairs, include_term_missing_flag)
    y_train, y_test = train[TARGET], test[TARGET]

    model = LogisticRegression(class_weight=class_weight)
    model.fit(X_train, y_train)

    if verbose:
        print_coefficients(model, X_train.columns)

    # predict_proba() is used instead of predict() since latter uses a 0.5 cutoff
    # which is used to assign 0/1 and is unsuitable for rare events.
    # former returns the actual probability
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_pred = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_pred)
    ks = ks_2samp(test_pred[y_test == 1], test_pred[y_test == 0]).statistic

    coefs = pd.Series(model.coef_[0], index=X_train.columns)
    n_wrong_sign = int((coefs > 0).sum())

    if verbose:
        print(f"[{name}] test AUC: {test_auc:.3f} | test KS: {ks:.3f} | "
              f"unexpected-sign coefficients: {n_wrong_sign}")
        print("=== calibration on test set, by predicted-risk decile ===")
        print(decile_calibration_table(y_test, test_pred).round(2).to_string())

    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    pd.to_pickle(
        {"model": model, "fitted_bins": fitted_bins,
         "numeric_features": numeric_features, "categorical_features": categorical_features,
         "interaction_pairs": interaction_pairs, "include_term_missing_flag": include_term_missing_flag},
        f"{ARTIFACTS_DIR}/logistic_model_{name}.pkl",
    )
    if verbose:
        print(f"[{name}] saved {ARTIFACTS_DIR}/logistic_model_{name}.pkl\n")

    return {"name": name, "test_auc": test_auc, "test_ks": ks,
            "n_wrong_sign": n_wrong_sign, "coefficients": coefs}


def main():
    run_variant("v1")


if __name__ == "__main__":
    main()
