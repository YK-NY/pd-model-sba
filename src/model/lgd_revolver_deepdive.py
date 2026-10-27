"""
LGD, continued: revolving-loan deep dive. Reproduces charts 03-07 in
outputs/figures/lgd/ -- term x collateral vs. LGD, revolver x collateral
vs. LGD, term x collateral vs. MonthsToChargeoff, GuaranteePct by
revolver/collateral, and interest rate + timing by RevolverStatus. All
point at the same conclusion: revolving loans are a structurally distinct,
riskier product category (less collateral, less guarantee, higher rate,
faster default, much higher severity), not just a footnote on one
statistic.

Also investigates an apparent anomaly: for short-term (<=36 month) loans,
mean MonthsToChargeoff is far longer than the loan's own term (e.g. ~67-70
months for a <=36-month loan). Confirmed NOT a data error and NOT specific
to revolvers (94.3% of non-revolvers and 98.3% of revolvers in this bucket
have MonthsToChargeoff > TermInMonths) -- TermInMonths is the loan's
SCHEDULED repayment period, while ChargeOffDate marks a separate, later
formal write-off decision after collection/workout efforts. The gap
between "stopped performing" and "formally charged off" is expected, not
a bug.

Population: same as lgd_severity_analysis.py -- ALL charged-off loans, no
seasoning filter.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from target_construction import load_model_base

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = str(Path(__file__).resolve().parents[2] / "outputs" / "figures" / "lgd")


def build_population():
    raw = load_model_base()
    chgoff = raw[raw["LoanStatus"] == "CHGOFF"].copy()
    chgoff["realized_lgd"] = chgoff["GrossChargeOffAmount"] / chgoff["GrossApproval"]
    chgoff["MonthsToChargeoff"] = (chgoff["ChargeOffDate"] - chgoff["ApprovalDate"]).dt.days / 30.44
    chgoff["GuaranteePct"] = chgoff["SBAGuaranteedApproval"] / chgoff["GrossApproval"] * 100
    chgoff["term_bucket"] = pd.cut(chgoff["TermInMonths"], bins=[0, 36, 60, 84, 120, 180, 300, 500])
    return chgoff


def check_term_anomaly(df):
    bucket = df[(df["TermInMonths"] > 0) & (df["TermInMonths"] <= 36)].copy()
    bucket["over_term"] = bucket["MonthsToChargeoff"] > bucket["TermInMonths"]
    print("=== short-term (<=36mo) loans: MonthsToChargeoff > TermInMonths? ===")
    print(f"overall: {bucket['over_term'].mean()*100:.1f}%")
    print(bucket.groupby("RevolverStatus")["over_term"].mean().mul(100).round(1).to_string())
    print("(confirms this is a general write-off-timing pattern, not revolver-specific "
          "and not term=0 contamination -- see module docstring)")


def print_summary_tables(df):
    print("\n=== collateral rate by RevolverStatus ===")
    print((pd.crosstab(df["RevolverStatus"], df["CollateralInd"], normalize="index") * 100).round(1).to_string())

    print("\n=== GuaranteePct by RevolverStatus ===")
    print(df.groupby("RevolverStatus")["GuaranteePct"].agg(["size", "mean", "median"]).round(2).to_string())

    print("\n=== MonthsToChargeoff by RevolverStatus ===")
    print(df.groupby("RevolverStatus")["MonthsToChargeoff"].agg(["size", "mean", "median"]).round(1).to_string())

    print("\n=== InitialInterestRate by RevolverStatus x CollateralInd ===")
    print(df.groupby(["RevolverStatus", "CollateralInd"], observed=True)["InitialInterestRate"]
          .agg(["size", "mean"]).round(2).to_string())

    print("\n=== realized LGD by RevolverStatus x CollateralInd ===")
    print(df.groupby(["RevolverStatus", "CollateralInd"], observed=True)["realized_lgd"]
          .agg(["size", "mean"]).round(3).to_string())


def chart_term_x_collateral_lgd(df):
    fig, ax = plt.subplots(figsize=(9, 6))
    g = df.groupby(["term_bucket", "CollateralInd"], observed=True)["realized_lgd"].mean().unstack()
    g.plot(kind="bar", ax=ax, color=["#DD8452", "#4C72B0"])
    ax.set_ylabel("mean realized LGD")
    ax.set_title("Realized LGD by term bucket x CollateralInd")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(f"{OUT}/03_lgd_term_x_collateral.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/03_lgd_term_x_collateral.png")


def chart_revolver_x_collateral_lgd(df):
    fig, ax = plt.subplots(figsize=(7, 6))
    g = df.groupby(["RevolverStatus", "CollateralInd"], observed=True)["realized_lgd"].mean().unstack()
    g.plot(kind="bar", ax=ax, color=["#DD8452", "#4C72B0"])
    ax.set_ylabel("mean realized LGD")
    ax.set_title("Realized LGD by RevolverStatus x CollateralInd")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/04_lgd_revolver_x_collateral.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/04_lgd_revolver_x_collateral.png")


def chart_monthstochargeoff_term_x_collateral(df):
    fig, ax = plt.subplots(figsize=(9, 6))
    g = df.groupby(["term_bucket", "CollateralInd"], observed=True)["MonthsToChargeoff"].mean().unstack()
    g.plot(kind="bar", ax=ax, color=["#DD8452", "#4C72B0"])
    ax.set_ylabel("mean MonthsToChargeoff")
    ax.set_title("MonthsToChargeoff by term bucket x CollateralInd")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(f"{OUT}/05_monthstochargeoff_term_x_collateral.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/05_monthstochargeoff_term_x_collateral.png")


def chart_guaranteepct_by_revolver_and_collateral(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df.boxplot(column="GuaranteePct", by="RevolverStatus", ax=axes[0])
    axes[0].set_title("GuaranteePct by RevolverStatus")
    axes[0].set_ylabel("GuaranteePct")
    df.boxplot(column="GuaranteePct", by="CollateralInd", ax=axes[1])
    axes[1].set_title("GuaranteePct by CollateralInd")
    axes[1].set_ylabel("GuaranteePct")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(f"{OUT}/06_guaranteepct_by_revolver_and_collateral.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/06_guaranteepct_by_revolver_and_collateral.png")


def chart_revolver_rate_and_timing(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    gm = df.groupby(["RevolverStatus", "CollateralInd"], observed=True)["InitialInterestRate"].mean().unstack()
    gm.plot(kind="bar", ax=axes[0], color=["#DD8452", "#4C72B0"])
    axes[0].set_title("Mean InitialInterestRate by RevolverStatus x CollateralInd")
    axes[0].set_ylabel("mean InitialInterestRate")
    axes[0].tick_params(axis="x", rotation=0)

    gm2 = df.groupby("RevolverStatus")["MonthsToChargeoff"].mean()
    axes[1].bar(gm2.index.astype(str), gm2.values, color="#55A868")
    axes[1].set_title("Mean MonthsToChargeoff by RevolverStatus")
    axes[1].set_ylabel("mean MonthsToChargeoff")
    fig.tight_layout()
    fig.savefig(f"{OUT}/07_revolver_rate_and_timing.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/07_revolver_rate_and_timing.png")


def main():
    df = build_population()
    check_term_anomaly(df)
    print_summary_tables(df)

    Path(OUT).mkdir(parents=True, exist_ok=True)
    chart_term_x_collateral_lgd(df)
    chart_revolver_x_collateral_lgd(df)
    chart_monthstochargeoff_term_x_collateral(df)
    chart_guaranteepct_by_revolver_and_collateral(df)
    chart_revolver_rate_and_timing(df)


if __name__ == "__main__":
    main()
