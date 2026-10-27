"""
LGD model: two deliverables (unwindowed and seasoned), evaluated against 
each other on the same held-out test set.

1. Segment-average lookup, RevolverStatus x CollateralInd (4 groups) --
   the lightweight approach, consistent with the brief's "LGD as a lighter
   extension" scope. Fit on train only, frozen for reuse, same discipline
   as the PD WOE bins.
2. A proper regression model (OLS), using the real severity drivers
   (RevolverStatus, CollateralInd, log-loan-size, TermInMonths, geo
   severity estimate, sector tier) -- tests whether combining several
   modest single-factor effects (each one found significant on its own,
   during the eta-squared analysis) adds up to meaningfully more
   predictive power than the simple segment average.

Same time-based split as PD (train ApprovalFY <= 2018, test > 2018) for
consistency -- not because LGD needs the same rigor as PD (it doesn't;
severity isn't a rare event and doesn't have PD's cycle-sensitivity
argument riding on it), but because reusing the same cutoff makes any
cross-references between the two models simpler to reason about.

Two populations are built and compared: all charged-off loans with no
filter (see lgd_severity_analysis.py for why this was the original
starting point), and the same loans restricted to those at least 60
months past approval -- the same seasoning rule already used for PD. The
unfiltered population showed a real problem: recent charge-offs are
disproportionately early, fast failures, which lose a larger share of the
loan (less time for principal paydown) -- the same distortion already
fixed for PD's target, but here it inflates severity instead of default
rate. The seasoned population is the one that ships (train_lgd_xgboost.py
trains on it); the unfiltered comparison exists to show why the filter
was necessary, not as an alternative.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRAIN_YEARS_MAX
from target_construction import load_model_base

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, r2_score

ARTIFACTS_DIR = str(Path(__file__).resolve().parents[2] / "outputs" / "model_artifacts")

SECTOR_TIER_MAP = {
    'Agriculture, Forestry, Fishing and Hunting': 'T1', 'Mining, Quarrying, and Oil and Gas Extraction': 'T1',
    'Accommodation and Food Services': 'T2', 'Arts, Entertainment, and Recreation': 'T2', 'Manufacturing': 'T2',
    'Health Care and Social Assistance': 'T3', 'Real Estate and Rental and Leasing': 'T3',
    'Other Services (except Public Administration)': 'T3', 'Educational Services': 'T3',
    'Retail Trade': 'T4', 'Other': 'T4', 'Information': 'T4', 'Finance and Insurance': 'T4',
    'Transportation and Warehousing': 'T5', 'Administrative and Support and Waste Management and Remediation Services': 'T5',
    'Professional, Scientific, and Technical Services': 'T5', 'Wholesale Trade': 'T5', 'Utilities': 'T5',
    'Construction': 'T6', 'Unclassified or Missing': 'T6',
}


def build_population(apply_seasoning_filter=False):
    raw = load_model_base()
    chgoff = raw[raw["LoanStatus"] == "CHGOFF"].copy()
    if apply_seasoning_filter: # for loans at least 60 months old
        months_since_approval = (chgoff["AsOfDate"] - chgoff["ApprovalDate"]).dt.days / 30.44
        chgoff = chgoff[months_since_approval >= 60].copy()
    chgoff["realized_lgd"] = chgoff["GrossChargeOffAmount"] / chgoff["GrossApproval"]
    chgoff["log_gross_approval"] = np.log(chgoff["GrossApproval"])
    chgoff["sector_tier"] = chgoff["NaicsSectorName"].map(SECTOR_TIER_MAP)

    chgoff["zip3"] = chgoff["BorrZip"].str[:3]
    return chgoff


def add_geo_lgd_estimate(train, apply_to):
    """Fit the zip3-w/-state-backoff LGD estimate on TRAIN only, apply to
    any dataframe (train or test) -- never let test rows leak into the
    lookup itself."""
    zip3_lgd = train.groupby("zip3")["realized_lgd"].agg(["size", "mean"])
    trusted_zip3 = zip3_lgd[zip3_lgd["size"] >= 30]["mean"]
    state_lgd = train.groupby("BorrState")["realized_lgd"].mean()
    overall_mean = train["realized_lgd"].mean()

    zip3_lookup = apply_to["zip3"].map(trusted_zip3)
    state_lookup = apply_to["BorrState"].map(state_lgd)
    return zip3_lookup.fillna(state_lookup).fillna(overall_mean)


def time_based_split(df):
    train = df[df["ApprovalFY"] <= TRAIN_YEARS_MAX].copy()
    test = df[df["ApprovalFY"] > TRAIN_YEARS_MAX].copy()
    return train, test


def fit_segment_lookup(train):
    """Deliverable 1: RevolverStatus x CollateralInd segment averages, fit
    on train only."""
    lookup = train.groupby(["RevolverStatus", "CollateralInd"], observed=True)["realized_lgd"].mean()
    return lookup


def predict_segment(df, lookup, fallback_mean):
    keys = list(zip(df["RevolverStatus"], df["CollateralInd"]))
    return pd.Series([lookup.get(k, fallback_mean) for k in keys], index=df.index)


def fit_regression(train):
    """Deliverable 2: OLS regression on the real severity drivers."""
    X = pd.get_dummies(
        train[["RevolverStatus", "CollateralInd", "sector_tier", "log_gross_approval", "TermInMonths", "geo_lgd_estimate"]],
        columns=["RevolverStatus", "CollateralInd", "sector_tier"], drop_first=True,
    )
    X = sm.add_constant(X.astype(float)) # from stats model
    y = train["realized_lgd"]
    model = sm.OLS(y, X).fit()
    return model, X.columns


def predict_regression(df, model, columns):
    X = pd.get_dummies(
        df[["RevolverStatus", "CollateralInd", "sector_tier", "log_gross_approval", "TermInMonths", "geo_lgd_estimate"]],
        columns=["RevolverStatus", "CollateralInd", "sector_tier"], drop_first=True,
    )
    X = sm.add_constant(X.astype(float)) # the model's intercept term, w/o it, regression line is forced through zreo.
    X = X.reindex(columns=columns, fill_value=0)  # align to training columns (unseen categories -> 0)
    return model.predict(X)


def run(name, apply_seasoning_filter, verbose=True):
    df = build_population(apply_seasoning_filter=apply_seasoning_filter)
    train, test = time_based_split(df)
    if verbose:
        print(f"[{name}] train (ApprovalFY <= {TRAIN_YEARS_MAX}): {len(train)} rows")
        print(f"[{name}] test  (ApprovalFY >  {TRAIN_YEARS_MAX}): {len(test)} rows")

    train["geo_lgd_estimate"] = add_geo_lgd_estimate(train, train)
    test["geo_lgd_estimate"] = add_geo_lgd_estimate(train, test)

    lookup = fit_segment_lookup(train)
    fallback_mean = train["realized_lgd"].mean()
    test_pred_seg = predict_segment(test, lookup, fallback_mean)

    model, columns = fit_regression(train)
    test_pred_reg = predict_regression(test, model, columns)

    if verbose:
        print(f"\n[{name}] segment lookup (RevolverStatus x CollateralInd), fit on train:")
        print(lookup.round(3).to_string())
        print(f"\n[{name}] OLS regression, in-sample R2={model.rsquared:.4f}")
        print(model.summary())

    results = {}
    print(f"\n=== [{name}] comparison on TEST set ===")
    for label, pred in [("segment_lookup", test_pred_seg), ("ols_regression", test_pred_reg),
                         ("flat_mean_baseline", pd.Series(fallback_mean, index=test.index))]:
        r2 = r2_score(test["realized_lgd"], pred)
        mae = mean_absolute_error(test["realized_lgd"], pred)
        print(f"[{name}] {label:<25} R2={r2:.4f}  MAE={mae:.4f}")
        results[label] = {"r2": r2, "mae": mae}

    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    pd.to_pickle(
        {"segment_lookup": lookup, "fallback_mean": fallback_mean,
         "regression_model": model, "regression_columns": list(columns), "train_rsquared": model.rsquared},
        f"{ARTIFACTS_DIR}/lgd_model_{name}.pkl",
    )
    print(f"[{name}] saved {ARTIFACTS_DIR}/lgd_model_{name}.pkl")
    return {"in_sample_r2": model.rsquared, "test": results, "n_train": len(train), "n_test": len(test)}


def main():
    unwindowed = run("v1_unwindowed", apply_seasoning_filter=False)
    windowed = run("v2_seasoned60mo", apply_seasoning_filter=True)

    print("\n=== side-by-side comparison: unwindowed vs. 60-month-seasoned ===")
    print(f"{'':<28}{'unwindowed':>14}{'seasoned':>14}")
    print(f"{'n_train':<28}{unwindowed['n_train']:>14}{windowed['n_train']:>14}")
    print(f"{'n_test':<28}{unwindowed['n_test']:>14}{windowed['n_test']:>14}")
    print(f"{'in-sample R2 (regression)':<28}{unwindowed['in_sample_r2']:>14.4f}{windowed['in_sample_r2']:>14.4f}")
    for label in ["segment_lookup", "ols_regression", "flat_mean_baseline"]:
        u, w = unwindowed["test"][label]["r2"], windowed["test"][label]["r2"]
        print(f"{'test R2: ' + label:<28}{u:>14.4f}{w:>14.4f}")


if __name__ == "__main__":
    main()
