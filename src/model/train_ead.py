"""
EAD, two steps.

Step 1 -- term (non-revolving) loans: EAD = GrossApproval. LGD has already been implicitly
assuming as the exposure denominator all along.

Step 2 -- revolving loans, where GrossApproval demonstrably fails (LGD
work found realized loss exceeds GrossApproval for a meaningful share of
revolvers). Historical EAD label used for fitting/evaluation:
    EAD_actual = max(GrossApproval, GrossChargeOffAmount)
A loan can't lose more dollars than it was exposed to, so this is the
smallest EAD value consistent with the observed loss -- self-consistent
by construction (the reconstructed loss rate GrossChargeOffAmount /
EAD_actual can never exceed 100%). This label only exists for loans that
have already charged off; it's what "should" have been the exposure,
given what we know happened.

Two options tried and compared, for revolvers only:
  A. Flat multiplier -- one ratio (EAD_actual / GrossApproval), averaged
     across all revolvers in training, applied to any revolver's
     GrossApproval at scoring time. The lightweight option.
  B. Regression (XGBoost, given it won clearly for LGD) predicting
     EAD_actual directly from origination-time features. The more
     sophisticated option, same feature set as the LGD models.

Consistency check, built in: for revolvers, compare EAD_estimate x
LGD_estimate (using the already-chosen XGBoost LGD model) against the
actual historical GrossChargeOffAmount -- do the two pieces, multiplied
together, reconstruct something close to real losses, or do they
contradict each other?

Population: same 60-month-seasoned CHGOFF population as the rest of the
LGD work, same time-based split, for consistency.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRAIN_YEARS_MAX
from train_lgd import build_population, time_based_split, add_geo_lgd_estimate
from train_lgd_xgboost import prepare_frame as prepare_lgd_frame, FEATURES_NUMERIC as LGD_NUMERIC, FEATURES_CATEGORICAL as LGD_CATEGORICAL

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ARTIFACTS_DIR = str(Path(__file__).resolve().parents[2] / "outputs" / "model_artifacts")


def add_ead_actual(df):
    df = df.copy()
    df["ead_actual"] = np.maximum(df["GrossApproval"], df["GrossChargeOffAmount"])
    return df


def step1_term_loan_ead(test):
    """EAD = GrossApproval for non-revolving loans -- report how well this
    already matches ead_actual, as a sanity check."""
    term = test[test["RevolverStatus"] == "N"]
    pred = term["GrossApproval"]
    actual = term["ead_actual"]
    r2 = r2_score(actual, pred)
    mae = mean_absolute_error(actual, pred)
    pct_exact = (pred == actual).mean() * 100
    print(f"[term loans] n={len(term)}, GrossApproval vs ead_actual: R2={r2:.4f}, MAE=${mae:,.0f}, "
          f"exact match (loss never exceeded approval): {pct_exact:.1f}%")
    return pred


def step2a_revolver_multiplier(train, test):
    """Option A: flat multiplier, fit on train revolvers only."""
    rev_train = train[train["RevolverStatus"] == "Y"]
    ratio = (rev_train["ead_actual"] / rev_train["GrossApproval"]).mean()
    print(f"[revolver, option A] flat multiplier (mean ead_actual/GrossApproval, train): {ratio:.4f}")

    rev_test = test[test["RevolverStatus"] == "Y"]
    pred = rev_test["GrossApproval"] * ratio
    return pred, rev_test, ratio


def step2b_revolver_regression(train, test):
    """Option B: XGBoost regression predicting ead_actual directly, same
    feature set as the LGD XGBoost model."""
    rev_train = train[train["RevolverStatus"] == "Y"].copy()
    rev_test = test[test["RevolverStatus"] == "Y"].copy()

    rev_train["geo_lgd_estimate"] = add_geo_lgd_estimate(train, rev_train)
    rev_test["geo_lgd_estimate"] = add_geo_lgd_estimate(train, rev_test)

    # RevolverStatus is constant (all Y) within this subset -- drop it, no information left to give
    numeric = LGD_NUMERIC # numeric feature set
    categorical = [c for c in LGD_CATEGORICAL if c != "RevolverStatus"]

    X_train = rev_train[numeric].astype(float).copy()
    X_test = rev_test[numeric].astype(float).copy()
    for col in categorical:
        X_train[col] = rev_train[col].astype("category")
        X_test[col] = pd.Categorical(rev_test[col], categories=X_train[col].cat.categories)

    y_train = rev_train["ead_actual"]
    model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                          enable_categorical=True, random_state=42)
    model.fit(X_train, y_train)
    pred = pd.Series(model.predict(X_test), index=rev_test.index)
    print(f"[revolver, option B] XGBoost regression on ead_actual, n_train={len(rev_train)}")
    return pred, rev_test, model


def consistency_check(rev_test, ead_pred, name):
    """Does EAD_estimate x LGD_estimate (XGBoost LGD model) reconstruct
    something close to actual GrossChargeOffAmount? Re-uses the LGD
    model's EXACT training categories (saved alongside it) -- refitting
    a fresh categorical dtype on just this revolver subset would silently
    produce different category codes than the model was trained on."""
    lgd_artifact = pd.read_pickle(f"{ARTIFACTS_DIR}/lgd_model_xgboost_v1.pkl")
    lgd_model = lgd_artifact["model"]
    X_lgd, _ = prepare_lgd_frame(rev_test, categories=lgd_artifact["train_categories"]) # rev_test as base dataset ensures only revolver loans are included. 
    lgd_pred = lgd_model.predict(X_lgd)

    reconstructed_loss = ead_pred.values * lgd_pred
    actual_loss = rev_test["GrossChargeOffAmount"].values

    r2 = r2_score(actual_loss, reconstructed_loss)
    mae = mean_absolute_error(actual_loss, reconstructed_loss)
    # portfolio-level (sum) ratio -- robust option, instead of individual loans with zero
    # actual loss, which would blow up a per-loan ratio average. hence summed
    portfolio_ratio = reconstructed_loss.sum() / actual_loss.sum()
    print(f"[{name}] consistency check: EAD_est x LGD_est vs actual GrossChargeOffAmount: "
          f"R2={r2:.4f}, MAE=${mae:,.0f}, portfolio total ratio (reconstructed/actual)={portfolio_ratio:.3f}")
    return {"r2": r2, "mae": mae, "portfolio_ratio": portfolio_ratio}


def main():
    df = build_population(apply_seasoning_filter=True)
    df = add_ead_actual(df)
    train, test = time_based_split(df)
    test["geo_lgd_estimate"] = add_geo_lgd_estimate(train, test)
    print(f"train: {len(train)} rows, test: {len(test)} rows\n")

    print("=== Step 1: term loans ===")
    step1_term_loan_ead(test)

    print("\n=== Step 2, Option A: revolver flat multiplier ===")
    pred_a, rev_test, ratio = step2a_revolver_multiplier(train, test)
    r2_a = r2_score(rev_test["ead_actual"], pred_a)
    mae_a = mean_absolute_error(rev_test["ead_actual"], pred_a)
    print(f"[revolver, option A] test: R2={r2_a:.4f}, MAE=${mae_a:,.0f}")

    print("\n=== Step 2, Option B: revolver XGBoost regression ===")
    pred_b, rev_test_b, model_b = step2b_revolver_regression(train, test)
    r2_b = r2_score(rev_test_b["ead_actual"], pred_b)
    mae_b = mean_absolute_error(rev_test_b["ead_actual"], pred_b)
    print(f"[revolver, option B] test: R2={r2_b:.4f}, MAE=${mae_b:,.0f}")

    print(f"\n=== comparison: option A vs option B, revolver EAD, test set ===")
    print(f"{'flat multiplier':<20}R2={r2_a:.4f}  MAE=${mae_a:,.0f}")
    print(f"{'XGBoost regression':<20}R2={r2_b:.4f}  MAE=${mae_b:,.0f}")

    print("\n=== consistency check: EAD_est x LGD_est vs actual loss, revolvers only ===")
    consistency_check(rev_test, pred_a, "option A (multiplier)")
    consistency_check(rev_test_b, pred_b, "option B (regression)")

    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    pd.to_pickle(
        {"term_loan_ead": "GrossApproval", "revolver_multiplier": ratio,
         "revolver_regression_model": model_b, "revolver_regression_features":
         {"numeric": LGD_NUMERIC, "categorical": [c for c in LGD_CATEGORICAL if c != "RevolverStatus"]}},
        f"{ARTIFACTS_DIR}/ead_model_v1.pkl",
    )
    print(f"\nsaved {ARTIFACTS_DIR}/ead_model_v1.pkl")


if __name__ == "__main__":
    main()
