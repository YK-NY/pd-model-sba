"""
PD backtest: FY2000-2009 (includes the 2008 recession), held out of the
entire modeling pass so far. Applies the ALREADY-TRAINED, FROZEN logistic
regression PD model exactly as-is -- no refitting, no new bins -- to see
whether it generalizes to a recession era it never learned from. Also saves
a predicted-vs-actual calibration chart by decile (13_backtest_calibration_by_decile.png).

Kept COMPLETELY SEPARATE from every other script: only imports and calls
existing functions (target construction, WOE transform, the trained
model), never edits them. Nothing here is required for the rest of the
pipeline to run -- this file can be deleted with zero effect elsewhere.

Missing InitialInterestRate (94.7% of this file, vs. 0% in 2010+) is
imputed here, self-contained, as Prime rate at ApprovalDate (FRED MPRIME)
plus a simulated spread. The spread is drawn from a distribution centered
on this era's own OWN observed spread (from the ~5.3% of loans that do
have a real rate, cleaned): mean 3.14, std 1.26 -- mean chosen over the
lower median (2.75) deliberately, since 2000-2009 has a much larger
mean-median gap (0.39) than 2010-2019 (0.12) or 2020-Curr (0.02),
indicating real right-skew in that era's spread specifically; using the
mean leans into that skew for a more conservative (not central-tendency)
imputation. Fixed random seed (42, matching this project's convention)
for reproducibility.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_prep"))
from sector_eda import load_clean_sector_df
from target_construction import remove_young_loans, remove_cancelled_and_committed, assign_target
from woe_binning import apply_woe_numeric, apply_woe_categorical
from loan_feature_extraction import extract_pd_features
from score_loan import _fit_geo_risk_lookup_pd
import load_sba

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_DIR = str(Path(__file__).resolve().parents[2] / "outputs" / "model_artifacts")
OUT = str(Path(__file__).resolve().parents[2] / "outputs" / "figures" / "pd_model")
SPREAD_MEAN = 3.14   # 2000-2009's own observed spread, cleaned data, n=36,583
SPREAD_STD = 1.26
SEED = 42


def impute_missing_interest_rate(df):
    """Self-contained: fills NaN InitialInterestRate as Prime-at-ApprovalDate
    + a simulated spread drawn from this era's own observed distribution.
    Does not touch rows that already have a real rate."""
    df = df.copy()
    missing = df["InitialInterestRate"].isna()
    n_missing = missing.sum()
    print(f"imputing InitialInterestRate for {n_missing} of {len(df)} loans "
          f"({n_missing/len(df)*100:.1f}%)")

    prime = pd.read_csv(load_sba._DATA_RAW / "MPRIME.csv", parse_dates=["observation_date"])
    prime["year_month"] = prime["observation_date"].dt.to_period("M")
    prime = prime.set_index("year_month")["MPRIME"]

    year_month = df["ApprovalDate"].dt.to_period("M")
    prime_at_approval = year_month.map(prime)

    rng = np.random.default_rng(SEED)
    simulated_spread = pd.Series(rng.normal(SPREAD_MEAN, SPREAD_STD, size=len(df)), index=df.index)
    imputed_rate = prime_at_approval + simulated_spread

    df.loc[missing, "InitialInterestRate"] = imputed_rate[missing]
    df["interest_rate_imputed"] = missing
    return df


def build_backtest_population():
    """Cleaned 2000-2009 file, through the SAME target-construction steps
    (seasoning filter, cancelled/committed drop, target_60mo label) used
    for the 2010+ training/test population -- so the backtest asks the
    identical question ("did it default within 60 months") of a
    population the model has never seen."""
    raw = load_clean_sector_df("2000_2009")
    df = remove_young_loans(raw, n_months=60)
    df = remove_cancelled_and_committed(df)
    df = assign_target(df, n_months=60)
    df = impute_missing_interest_rate(df)
    return df


def score_backtest_population(df):
    """Applies the FROZEN, already-trained PD model -- same fitted WOE
    bins, same logistic regression coefficients -- no refitting."""
    pd_artifact = pd.read_pickle(f"{ARTIFACTS_DIR}/logistic_model_v1.pkl")
    geo_risk_lookup = _fit_geo_risk_lookup_pd()  # fit on the ORIGINAL 2010+ training split, not this file

    pd_raw = extract_pd_features(df, geo_risk_lookup)
    pd_woe = pd.DataFrame(index=pd_raw.index)
    for col in pd_artifact["numeric_features"]:
        pd_woe[col] = apply_woe_numeric(pd_raw[col], pd_artifact["fitted_bins"][col])
    for col in pd_artifact["categorical_features"]:
        pd_woe[col] = apply_woe_categorical(pd_raw[col], pd_artifact["fitted_bins"][col])
    pd_woe = pd_woe.fillna(0.0)  # same neutral fallback as score_loan.py

    predicted_pd = pd_artifact["model"].predict_proba(pd_woe)[:, 1]
    return predicted_pd


def fit_recalibration(predicted_pd, actual):
    """A downturn-mode recalibration (isotonic regression).
    Leaves the model's RANKING untouched (isotonic regression is
    monotonic -- it cannot reorder loans, only rescale the probability
    values), and corrects only the absolute level using the
    predicted-vs-actual relationship this exact backtest already found.

    OUTCOME: tried and decided against. The model
    underpredicted because FY2000-2009 is a genuinely different population
    from the training era (PSI 0.570), not because its numbers are on the
    wrong scale, so rescaling would only paper over the real issue. The
    ranking is essentially unchanged (AUC 0.779 before, 0.780 after). Not used in scoring.

    CAVEAT, stated plainly: fit and evaluated on the SAME data here, so
    the "after" numbers below are somewhat definitionally close to
    actual -- this demonstrates the mechanism works, not that it would
    generalize this well to a genuinely new, unseen downturn."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(predicted_pd, actual)
    return iso


def plot_backtest_calibration(predicted, actual, out_path):
    """Predicted vs. actual default rate by predicted-PD decile on the
    FY2000-2009 backtest population. The two curves are the model's mean
    prediction and the real default rate within each decile."""
    df = pd.DataFrame({"predicted": predicted, "actual": actual})
    df["decile"] = pd.qcut(df["predicted"], q=10, labels=False, duplicates="drop")
    by_decile = df.groupby("decile").agg(
        mean_predicted=("predicted", "mean"), mean_actual=("actual", "mean")) * 100
    x = np.arange(1, len(by_decile) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(x, by_decile["mean_predicted"], marker="o", label="predicted (mean)", color="#4C72B0")
    ax.plot(x, by_decile["mean_actual"], marker="o", label="actual default rate", color="#DD8452")
    # label the riskiest decile, where the two curves are furthest apart
    ax.annotate(f"{by_decile['mean_actual'].iloc[-1]:.1f}%", (x[-1], by_decile["mean_actual"].iloc[-1]),
                textcoords="offset points", xytext=(0, 8), ha="center", color="#DD8452")
    ax.annotate(f"{by_decile['mean_predicted'].iloc[-1]:.1f}%", (x[-1], by_decile["mean_predicted"].iloc[-1]),
                textcoords="offset points", xytext=(0, -16), ha="center", color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xlabel("predicted-PD decile (1 = safest, 10 = riskiest)")
    ax.set_ylabel("default rate %")
    ax.set_title("Backtest FY2000-2009: predicted vs. actual default rate by decile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    df = build_backtest_population()
    print(f"\nbacktest population: {len(df)} loans, FY2000-2009, seasoned 60mo+")
    print(f"actual observed default rate (target_60mo): {df['target_60mo'].mean()*100:.2f}%")

    predicted_pd = score_backtest_population(df)
    actual = df["target_60mo"].values

    auc = roc_auc_score(actual, predicted_pd)
    fpr_tpr_thresh = None
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(actual, predicted_pd)
    ks = (tpr - fpr).max()

    print(f"\n=== backtest results, FROZEN model applied to FY2000-2009 (never trained on this era) ===")
    print(f"AUC: {auc:.3f}")
    print(f"KS: {ks:.3f}")
    print(f"mean predicted PD: {predicted_pd.mean()*100:.2f}%  |  actual default rate: {actual.mean()*100:.2f}%")

    imputed_pct = df["interest_rate_imputed"].mean() * 100
    print(f"\nNOTE: {imputed_pct:.1f}% of loans had InitialInterestRate imputed "
          f"(Prime + simulated spread, seed={SEED}) -- interpret results with this in mind.")

    print(f"\n=== calibration by predicted-PD decile (BEFORE recalibration) ===")
    calib = pd.DataFrame({"predicted": predicted_pd, "actual": actual})
    calib["decile"] = pd.qcut(calib["predicted"], q=10, duplicates="drop")
    print(calib.groupby("decile", observed=True).agg(
        n=("actual", "size"), mean_predicted=("predicted", "mean"), mean_actual=("actual", "mean")
    ).round(4).to_string())

    Path(OUT).mkdir(parents=True, exist_ok=True)
    plot_backtest_calibration(predicted_pd, actual, f"{OUT}/13_backtest_calibration_by_decile.png")
    print(f"\nsaved {OUT}/13_backtest_calibration_by_decile.png")

    # ---------- recalibration ----------
    iso = fit_recalibration(predicted_pd, actual)
    recalibrated_pd = iso.predict(predicted_pd)
    auc_recal = roc_auc_score(actual, recalibrated_pd)  # should be ~identical -- ranking is unchanged

    print(f"\n=== AFTER recalibration (isotonic, fit on this same backtest population -- see caveat in fit_recalibration()) ===")
    print(f"AUC after recalibration: {auc_recal:.3f}  (before: {auc:.3f} -- ranking should be ~unchanged)")
    print(f"mean recalibrated PD: {recalibrated_pd.mean()*100:.2f}%  |  actual: {actual.mean()*100:.2f}%")

    calib["recalibrated"] = recalibrated_pd
    print(calib.groupby("decile", observed=True).agg(
        n=("actual", "size"), mean_predicted=("predicted", "mean"),
        mean_recalibrated=("recalibrated", "mean"), mean_actual=("actual", "mean")
    ).round(4).to_string())

    return {"auc": auc, "ks": ks, "n": len(df), "predicted": predicted_pd, "actual": actual,
            "recalibrated": recalibrated_pd, "auc_recal": auc_recal}


if __name__ == "__main__":
    main()
