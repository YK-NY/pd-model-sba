"""
LGD, steps 1-4: build the severity population, compute realized LGD, sanity
check its distribution, and check whether it varies meaningfully by
segment (collateral, sector, loan size, term) -- before deciding whether a
flat average or a segment-average is justified.

Population: ALL charged-off loans (LoanStatus == CHGOFF), no 60-month
seasoning filter applied. The seasoning filter exists to fix a PD-specific
censoring problem (STILL_OPEN loans that haven't had time to default yet)
-- it doesn't apply here, since a CHGOFF loan's severity is fully known
and resolved regardless of how old the loan is. Using the full,
unrestricted CHGOFF population maximizes sample size for LGD specifically.

Gross severity only -- GrossChargeOffAmount is gross of guaranteed +
non-guaranteed loss, with no separate recovery-amount field in this data.
No net-of-recovery proxy exists yet; this run reports gross severity and
its segment structure only, so the next step (whether/how to approximate
net severity) can be decided from what the data actually shows.
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


def build_lgd_population():
    raw = load_model_base()
    chgoff = raw[raw["LoanStatus"] == "CHGOFF"].copy()
    chgoff["realized_lgd"] = chgoff["GrossChargeOffAmount"] / chgoff["GrossApproval"]
    return chgoff


def main():
    df = build_lgd_population()
    print(f"LGD population: {len(df)} charged-off loans (all vintages/ages, no seasoning filter)")
    print()

    print("=== realized_lgd distribution ===")
    print(df["realized_lgd"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(3).to_string())
    print()
    print(f"rows with realized_lgd > 1: {(df['realized_lgd'] > 1).sum()} "
          f"({(df['realized_lgd'] > 1).mean() * 100:.2f}%)")
    print(f"rows with realized_lgd < 0: {(df['realized_lgd'] < 0).sum()} "
          f"({(df['realized_lgd'] < 0).mean() * 100:.2f}%)")
    print(f"rows with missing GrossChargeOffAmount: {df['GrossChargeOffAmount'].isna().sum()}")
    print()

    print("=== % collateralized, this subset ===")
    print(df["CollateralInd"].value_counts(dropna=False))
    print((df["CollateralInd"].value_counts(normalize=True) * 100).round(1))
    print()

    print("=== realized LGD by CollateralInd ===")
    g = df.groupby("CollateralInd", observed=True)["realized_lgd"].agg(["size", "mean", "median", "std"])
    print(g.round(3).to_string())
    print()

    print("=== realized LGD by NaicsSectorName ===")
    g = df.groupby("NaicsSectorName", observed=True)["realized_lgd"].agg(["size", "mean", "median"])
    print(g.sort_values("mean").round(3).to_string())
    print()

    df["loan_size_bucket"] = pd.qcut(df["GrossApproval"], q=5, duplicates="drop")
    print("=== realized LGD by loan-size quintile ===")
    g = df.groupby("loan_size_bucket", observed=True)["realized_lgd"].agg(["size", "mean", "median"])
    print(g.round(3).to_string())
    print()

    df["term_bucket"] = pd.cut(df["TermInMonths"], bins=[0, 36, 60, 84, 120, 180, 300, 500])
    print("=== realized LGD by term bucket ===")
    g = df.groupby("term_bucket", observed=True)["realized_lgd"].agg(["size", "mean", "median"])
    print(g.round(3).to_string())

    # charts
    Path(OUT).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].hist(df["realized_lgd"].clip(upper=1.5), bins=60, color="#4C72B0")
    axes[0, 0].set_title("Realized LGD distribution (gross severity)")
    axes[0, 0].set_xlabel("realized_lgd (clipped at 1.5 for display)")

    coll = df.groupby("CollateralInd", observed=True)["realized_lgd"].mean()
    axes[0, 1].bar(coll.index.astype(str), coll.values, color="#DD8452")
    axes[0, 1].set_title("Mean realized LGD by CollateralInd")
    axes[0, 1].set_ylabel("mean realized_lgd")

    sector = df.groupby("NaicsSectorName", observed=True)["realized_lgd"].mean().sort_values()
    axes[1, 0].barh(sector.index, sector.values, color="#55A868")
    axes[1, 0].set_title("Mean realized LGD by sector")
    axes[1, 0].set_xlabel("mean realized_lgd")
    axes[1, 0].tick_params(axis="y", labelsize=7)

    term = df.groupby("term_bucket", observed=True)["realized_lgd"].mean()
    axes[1, 1].bar(term.index.astype(str), term.values, color="#8172B2")
    axes[1, 1].set_title("Mean realized LGD by term bucket")
    axes[1, 1].set_xlabel("term bucket (months)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    fig.suptitle("LGD severity analysis: gross realized LGD by segment")
    fig.tight_layout()
    fig.savefig(f"{OUT}/01_lgd_severity_by_segment.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved {OUT}/01_lgd_severity_by_segment.png")


if __name__ == "__main__":
    main()
