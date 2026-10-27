"""
Per-model feature extraction for score_loan(). Kept as its own module,
separate from score_loan.py's orchestration, because PD/LGD/EAD each need
a different subset of a raw loan's fields.Each model gets an explicit list 
of the fields it uses, in a fixed order, rather than a shared pass-through, 
so a field cannot silently be matched to the wrong model.

Each function takes a raw loan DataFrame (one row per loan, any number of
rows) with at least the origination-time fields available, and returns
only what that specific model needs, built the same way it was built
during training (e.g. GuaranteePct/geo estimates are derived here, not
assumed to already exist on the input).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from train_lgd import SECTOR_TIER_MAP  # single source of truth -- do not redefine here

PD_NUMERIC = ["TermInMonths", "GrossApproval", "InitialInterestRate", "GuaranteePct", "geo_risk_pct"]
PD_CATEGORICAL = ["ProcessingMethod", "CollateralInd", "NaicsSectorName", "BusinessAgeBucket"]

LGD_NUMERIC = ["GrossApproval", "TermInMonths", "geo_lgd_estimate"]
LGD_CATEGORICAL = ["RevolverStatus", "CollateralInd", "sector_tier"]

EAD_REVOLVER_NUMERIC = ["GrossApproval", "TermInMonths", "geo_lgd_estimate"]
EAD_REVOLVER_CATEGORICAL = ["CollateralInd", "sector_tier"]  # RevolverStatus dropped -- constant within revolver-only scoring


def _add_guarantee_pct(df):
    df = df.copy()
    if "GuaranteePct" not in df.columns:
        df["GuaranteePct"] = df["SBAGuaranteedApproval"] / df["GrossApproval"] * 100
    return df


def _add_geo_risk_pct(df, geo_risk_lookup):
    """geo_risk_lookup: the fitted {zip3: rate} / {state: rate} lookups
    saved alongside the PD model at training time -- never recomputed live
    from the scoring batch itself, same discipline as everything else."""
    df = df.copy()
    df["zip3"] = df["BorrZip"].str[:3]
    zip3_lookup = df["zip3"].map(geo_risk_lookup["zip3_rate"])
    state_lookup = df["BorrState"].map(geo_risk_lookup["state_rate"])
    df["geo_risk_pct"] = (zip3_lookup.fillna(state_lookup).fillna(geo_risk_lookup["overall_mean"])) * 100
    return df


def _add_geo_lgd_estimate(df, geo_lgd_lookup):
    df = df.copy()
    df["zip3"] = df["BorrZip"].str[:3]
    zip3_lookup = df["zip3"].map(geo_lgd_lookup["zip3_rate"])
    state_lookup = df["BorrState"].map(geo_lgd_lookup["state_rate"])
    df["geo_lgd_estimate"] = zip3_lookup.fillna(state_lookup).fillna(geo_lgd_lookup["overall_mean"])
    return df


def extract_pd_features(loan_df, geo_risk_lookup):
    """Returns a raw-value DataFrame with exactly PD's 9 fields, ready for
    the WOE transform (apply_woe_numeric/apply_woe_categorical) -- this
    function does NOT apply WOE itself, just assembles the right raw
    columns, same separation of concerns as train_logistic.py's transform()."""
    df = _add_guarantee_pct(loan_df)
    df = _add_geo_risk_pct(df, geo_risk_lookup)
    return df[PD_NUMERIC + PD_CATEGORICAL].copy()


def extract_lgd_features(loan_df, geo_lgd_lookup):
    """Returns a DataFrame with exactly LGD's 6 fields, raw (XGBoost
    handles categoricals natively -- see prepare_frame in
    train_lgd_xgboost.py for the categorical-dtype step, done separately
    with the model's saved train_categories, not here)."""
    df = loan_df.copy()
    df["sector_tier"] = df["NaicsSectorName"].map(SECTOR_TIER_MAP)
    df = _add_geo_lgd_estimate(df, geo_lgd_lookup)
    return df[LGD_NUMERIC + LGD_CATEGORICAL].copy()


def extract_ead_revolver_features(loan_df, geo_lgd_lookup):
    """Returns a DataFrame with exactly the revolver EAD model's fields.
    Only meaningful for RevolverStatus == 'Y' rows -- term loans use
    GrossApproval directly, no extraction needed."""
    df = loan_df.copy()
    df["sector_tier"] = df["NaicsSectorName"].map(SECTOR_TIER_MAP)
    df = _add_geo_lgd_estimate(df, geo_lgd_lookup)
    return df[EAD_REVOLVER_NUMERIC + EAD_REVOLVER_CATEGORICAL].copy()
