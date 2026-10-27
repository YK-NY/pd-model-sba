"""
score_loan() assembly -- PD, LGD, EAD, EL for one or more new loans.

Loads the three already-fitted, saved model artifacts, extracts each
model's own feature subset via loan_feature_extraction.py, applies each
model's frozen transform, and combines into EL = PD x LGD x EAD.

Geo lookups (geo_risk_pct for PD, geo_lgd_estimate for LGD) were only ever
built inline during training and never persisted -- fitted here, ONCE, on
the correct training populations (same train/test split each model
actually used), and cached for reuse. This is the one piece of "frozen
transform" bookkeeping that was missing before this module existed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from loan_feature_extraction import extract_pd_features, extract_lgd_features, extract_ead_revolver_features
from woe_binning import build_modeling_population, apply_woe_numeric, apply_woe_categorical
from train_logistic import time_based_split as pd_time_split
from train_lgd import build_population as build_lgd_population, time_based_split as lgd_time_split
from train_lgd_xgboost import prepare_frame as prepare_lgd_frame
from target_construction import load_model_base

import pandas as pd

ARTIFACTS_DIR = str(Path(__file__).resolve().parents[2] / "outputs" / "model_artifacts")


def _fit_geo_risk_lookup_pd(min_zip3_n=30):
    """PD's geo_risk_pct, fit on PD's own training split only."""
    df = build_modeling_population()
    train, _ = pd_time_split(df)
    train = train.copy()
    train["zip3"] = train["BorrZip"].str[:3]
    zip3_rate = train.groupby("zip3")["target_60mo"].agg(["size", "mean"])
    zip3_rate = zip3_rate[zip3_rate["size"] >= min_zip3_n]["mean"]
    state_rate = train.groupby("BorrState")["target_60mo"].mean()
    return {"zip3_rate": zip3_rate, "state_rate": state_rate, "overall_mean": train["target_60mo"].mean()}


def _fit_geo_lgd_lookup(min_zip3_n=30):
    """LGD's geo_lgd_estimate, fit on LGD's own (60-month-seasoned)
    training split only."""
    df = build_lgd_population(apply_seasoning_filter=True)
    train, _ = lgd_time_split(df)
    train = train.copy()
    train["zip3"] = train["BorrZip"].str[:3]
    zip3_rate = train.groupby("zip3")["realized_lgd"].agg(["size", "mean"])
    zip3_rate = zip3_rate[zip3_rate["size"] >= min_zip3_n]["mean"]
    state_rate = train.groupby("BorrState")["realized_lgd"].mean()
    return {"zip3_rate": zip3_rate, "state_rate": state_rate, "overall_mean": train["realized_lgd"].mean()}


def load_scoring_context():
    """Everything score_loan() needs, fit/loaded once. Cache this and pass
    it around rather than calling load_scoring_context() per loan."""
    pd_artifact = pd.read_pickle(f"{ARTIFACTS_DIR}/logistic_model_v1.pkl")
    lgd_artifact = pd.read_pickle(f"{ARTIFACTS_DIR}/lgd_model_xgboost_v1.pkl")
    ead_artifact = pd.read_pickle(f"{ARTIFACTS_DIR}/ead_model_v1.pkl")
    return {
        "pd": pd_artifact,
        "lgd": lgd_artifact,
        "ead": ead_artifact,
        "geo_risk_lookup_pd": _fit_geo_risk_lookup_pd(),
        "geo_lgd_lookup": _fit_geo_lgd_lookup(),
    }


def score_loan(loan_df, ctx):
    """loan_df: raw loan record(s), one row per loan, with at minimum the
    origination-time fields every model needs (see loan_feature_extraction.py).
    ctx: output of load_scoring_context(). Returns a DataFrame, one row per
    input loan, columns PD/LGD/EAD/EL."""

    # ---------- PD ----------
    pd_raw = extract_pd_features(loan_df, ctx["geo_risk_lookup_pd"])
    pd_woe = pd.DataFrame(index=pd_raw.index)
    for col in ctx["pd"]["numeric_features"]:
        pd_woe[col] = apply_woe_numeric(pd_raw[col], ctx["pd"]["fitted_bins"][col])
    for col in ctx["pd"]["categorical_features"]:
        pd_woe[col] = apply_woe_categorical(pd_raw[col], ctx["pd"]["fitted_bins"][col])
    # a raw numeric field missing on the input loan produces NaN here (no
    # bin to look up) -- same neutral fallback already used for an unseen
    # CATEGORY (WOE=0, "no information, assume average risk"), applied
    # here for consistency rather than crashing or silently dropping rows.
    pd_woe = pd_woe.fillna(0.0)
    PD = ctx["pd"]["model"].predict_proba(pd_woe)[:, 1]

    # ---------- LGD ----------
    lgd_raw = extract_lgd_features(loan_df, ctx["geo_lgd_lookup"])
    lgd_X, _ = prepare_lgd_frame(lgd_raw, categories=ctx["lgd"]["train_categories"])
    LGD = ctx["lgd"]["model"].predict(lgd_X)

    # ---------- EAD (branch by loan type) ----------
    EAD = loan_df["GrossApproval"].to_numpy(dtype=float).copy()
    is_revolver = (loan_df["RevolverStatus"] == "Y").to_numpy()
    if is_revolver.any():
        rev_loans = loan_df[is_revolver]
        ead_raw = extract_ead_revolver_features(rev_loans, ctx["geo_lgd_lookup"])
        ead_features = ctx["ead"]["revolver_regression_features"]
        ead_X = ead_raw[ead_features["numeric"]].astype(float).copy()
        for col in ead_features["categorical"]:
            ead_X[col] = pd.Categorical(ead_raw[col])
        ead_pred = ctx["ead"]["revolver_regression_model"].predict(ead_X)
        EAD[is_revolver] = ead_pred

    # ---------- EL ----------
    EL = PD * LGD * EAD # each is a numpy series

    # PD_annual: the equivalent constant annual default rate, for
    # comparing against annual external benchmarks only. NOT used in EL
    # (EL uses the real 5-year PD).
    #
    # PD = the model's predicted probability of default within 5 years.
    # PD_annual = the constant yearly rate that, applied independently
    # every year for 5 years, would reproduce that same 5-year outcome
    # (assumes the same risk applies each year).
    #
    # Surviving any one year independently has probability (1 - PD_annual).
    # Surviving all 5 years independently is that, five times over:
    #   (1 - PD_annual) ^ 5
    # which must equal the model's own 5-year survival chance, (1 - PD):
    #   (1 - PD_annual) ^ 5 = 1 - PD
    # Taking the fifth root of both sides and rearranging:
    #   PD_annual = 1 - (1 - PD) ^ (1/5)
    PD_annual = 1 - (1 - PD) ** (1 / 5)

    return pd.DataFrame({"PD": PD, "PD_annual": PD_annual, "LGD": LGD, "EAD": EAD, "EL": EL},
                         index=loan_df.index)


def main(n=20):
    """Demo: score real COMMIT loans (approved, not yet disbursed) -- the
    closest thing in this dataset to a genuinely new loan. n=None scores the FULL COMMIT
    population instead of a sample."""
    raw = load_model_base()
    commit = raw[raw["LoanStatus"] == "COMMIT"].copy()
    sample = commit if n is None else commit.sample(n=n, random_state=42)

    print(f"scoring {len(sample)} COMMIT loans (approved, not yet disbursed)...")
    ctx = load_scoring_context()
    results = score_loan(sample, ctx)

    if n is not None:
        display = pd.concat([
            sample[["GrossApproval", "TermInMonths", "RevolverStatus", "CollateralInd", "NaicsSectorName"]].reset_index(drop=True),
            results.reset_index(drop=True).round(4),
        ], axis=1)
        print(display.to_string())

    print(f"\n=== portfolio-level summary, n={len(sample)} ===")
    print(f"total EAD (sum): ${results['EAD'].sum():,.0f}")
    print(f"total EL (sum, portfolio expected loss): ${results['EL'].sum():,.0f}")
    print(f"EL as % of total EAD: {results['EL'].sum() / results['EAD'].sum() * 100:.2f}%")
    print(f"mean PD (60mo cumulative): {results['PD'].mean()*100:.2f}%")
    print(f"mean PD (annualized): {results['PD_annual'].mean()*100:.2f}%")
    print(f"mean LGD: {results['LGD'].mean()*100:.2f}%")

    # External benchmarks: reported side by side with our own numbers,
    # nothing more. Deliberately NOT turned into a ratio or multiplier --
    # a ratio implies the two are measuring the same thing closely enough
    # to be comparable, and that assumption doesn't hold here. Our loans,
    # Moody's, and Crestmont's each come from different loan sizes,
    # company profiles, and underwriting standards -- too many
    # uncontrolled differences to treat a computed ratio as meaningful.
    # This does not change EL above in any way.
    #
    # CDT = Central Default Tendency, Moody's own term for a benchmark default rate over a stated horizon.
    MOODY_CDT_MEDIUM = 0.1186   # Moody's SME ABS methodology, Appendix 5, US medium companies, 5yr
    MOODY_CDT_MICRO = 0.149     # same, US micro companies, 5yr
    CRESTMONT_ANNUAL_LOW = 0.04   # Crestmont Capital, business loan default rates 2026, low end
    CRESTMONT_ANNUAL_HIGH = 0.12  # same, high end
    our_pd_5yr = results["PD"].mean()
    our_pd_annual = results["PD_annual"].mean()
    print(f"\n=== external benchmarks, for reference only (not a calibrated comparison) ===")
    print(f"  our mean PD (5yr cumulative): {our_pd_5yr*100:.2f}%")
    print(f"  our mean PD (annualized):     {our_pd_annual*100:.2f}%")
    print(f"  Moody's CDT, medium companies (5yr): {MOODY_CDT_MEDIUM*100:.2f}%")
    print(f"  Moody's CDT, micro companies (5yr):  {MOODY_CDT_MICRO*100:.2f}%")
    print(f"  Crestmont, annual (low end):          {CRESTMONT_ANNUAL_LOW*100:.2f}%")
    print(f"  Crestmont, annual (high end):         {CRESTMONT_ANNUAL_HIGH*100:.2f}%")

    return results


if __name__ == "__main__":
    main()
