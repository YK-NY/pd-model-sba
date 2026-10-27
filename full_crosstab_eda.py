"""
Full crosstab / chart pass across categorical + continuous variables vs.
chargeoff outcome, per user request (points 4 and 6). Runs on the pooled,
cleaned, NAICS-sector-assigned dataset (all three files concatenated, with
a `vintage_file` tag) since most of these are about behavioral patterns,
not era comparisons -- exceptions (MonthsToChargeoff, sector dispersion)
keep the vintage split explicit.

Outputs go to outputs/figures/, numbered 10+ (01-09 already exist from the
earlier diagnostic pass).

Run: python full_crosstab_eda.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src" / "data_prep"))
import load_sba
import naics_grouping
from sector_eda import load_clean_sector_df, ORDER

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/figures"


def load_pooled():
    frames = []
    for name in ORDER:
        df = load_clean_sector_df(name)
        df["vintage_file"] = name
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    print(f"pooled shape: {pooled.shape}")
    return pooled


def chgoff_rate_by(df, col, min_n=30):
    """resolved-only chargeoff rate (CHGOFF / (PIF+CHGOFF)) by a grouping column, event-count filtered."""
    resolved = df[df["LoanStatus"].isin(["PIF", "CHGOFF"])]
    g = resolved.groupby(col)["LoanStatus"]
    n = g.size()
    chgoff = resolved[resolved["LoanStatus"] == "CHGOFF"].groupby(col).size().reindex(n.index).fillna(0)
    rate = (chgoff / n * 100)
    out = pd.DataFrame({"n_resolved": n, "chgoff_events": chgoff, "chgoff_rate_pct": rate})
    return out[out["n_resolved"] >= min_n].sort_values("chgoff_rate_pct", ascending=False)


def main():
    pooled = load_pooled()

    # ------------------------------------------------------------------
    # 10. LoanStatus % breakdown by sector (stacked bar) -- point 4
    # ------------------------------------------------------------------
    ct = pd.crosstab(pooled["NaicsSectorName"], pooled["LoanStatus"], normalize="index") * 100
    ct = ct.loc[pooled["NaicsSectorName"].value_counts().index]  # order by sector size
    fig, ax = plt.subplots(figsize=(10, 8))
    ct.plot(kind="barh", stacked=True, ax=ax, colormap="tab10")
    ax.set_xlabel("% of loans in sector")
    ax.set_title("LoanStatus mix by sector (pooled across all 3 files)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/10_loanstatus_by_sector.png", dpi=150)
    plt.close(fig)
    print("saved 10_loanstatus_by_sector.png")

    # ------------------------------------------------------------------
    # 11. NAICS sector x BusinessAge heatmap
    # ------------------------------------------------------------------
    ct = pd.crosstab(pooled["NaicsSectorName"], pooled["BusinessAge"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(ct.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels(ct.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels(ct.index, fontsize=7)
    ax.set_title("BusinessAge mix (%) by sector")
    fig.colorbar(im, ax=ax, label="% within sector")
    fig.tight_layout()
    fig.savefig(f"{OUT}/11_sector_vs_businessage.png", dpi=150)
    plt.close(fig)
    print("saved 11_sector_vs_businessage.png")

    # ------------------------------------------------------------------
    # 12. NAICS sector x BusinessType
    # ------------------------------------------------------------------
    ct = pd.crosstab(pooled["NaicsSectorName"], pooled["BusinessType"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(10, 8))
    ct.plot(kind="barh", stacked=True, ax=ax, colormap="Set2")
    ax.set_xlabel("% of loans in sector")
    ax.set_title("BusinessType mix by sector")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/12_sector_vs_businesstype.png", dpi=150)
    plt.close(fig)
    print("saved 12_sector_vs_businesstype.png")

    # ------------------------------------------------------------------
    # 13. ProcessingMethod x chargeoff rate
    # ------------------------------------------------------------------
    t = chgoff_rate_by(pooled, "ProcessingMethod", min_n=200)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(t.index, t["chgoff_rate_pct"], color="#C44E52")
    ax.set_xlabel("chargeoff rate % (of resolved)")
    ax.set_title("Chargeoff rate by ProcessingMethod (n_resolved >= 200)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/13_processingmethod_vs_chgoffrate.png", dpi=150)
    plt.close(fig)
    print("saved 13_processingmethod_vs_chgoffrate.png")
    print(t.to_string())

    # ------------------------------------------------------------------
    # 14. GrossApproval size-bucket x chargeoff rate
    # ------------------------------------------------------------------
    bins = [0, 25000, 50000, 100000, 250000, 500000, 1_000_000, np.inf]
    labels = ["<25K", "25-50K", "50-100K", "100-250K", "250-500K", "500K-1M", "1M+"]
    pooled["GrossApproval_bucket"] = pd.cut(pooled["GrossApproval"], bins=bins, labels=labels)
    t = chgoff_rate_by(pooled, "GrossApproval_bucket", min_n=1)
    t = t.reindex(labels)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(t.index, t["chgoff_rate_pct"], color="#4C72B0")
    ax.set_ylabel("chargeoff rate % (of resolved)")
    ax.set_title("Chargeoff rate by GrossApproval size bucket")
    fig.tight_layout()
    fig.savefig(f"{OUT}/14_loansize_vs_chgoffrate.png", dpi=150)
    plt.close(fig)
    print("saved 14_loansize_vs_chgoffrate.png")

    # ------------------------------------------------------------------
    # 15. RevolverStatus x chargeoff rate, plus median GrossApproval
    # ------------------------------------------------------------------
    t = chgoff_rate_by(pooled, "RevolverStatus", min_n=1)
    med_amt = pooled.groupby("RevolverStatus")["GrossApproval"].median()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(t.index.astype(str), t["chgoff_rate_pct"], color="#55A868")
    axes[0].set_title("Chargeoff rate by RevolverStatus")
    axes[0].set_ylabel("chargeoff rate %")
    axes[1].bar(med_amt.index.astype(str), med_amt.values, color="#8172B2")
    axes[1].set_title("Median GrossApproval by RevolverStatus")
    fig.tight_layout()
    fig.savefig(f"{OUT}/15_revolverstatus_vs_chgoff_amount.png", dpi=150)
    plt.close(fig)
    print("saved 15_revolverstatus_vs_chgoff_amount.png")

    # ------------------------------------------------------------------
    # 16. FixedorVariableInterestInd x chargeoff rate, by ProcessingMethod (top 6)
    # ------------------------------------------------------------------
    top_methods = pooled["ProcessingMethod"].value_counts().head(6).index
    sub = pooled[pooled["ProcessingMethod"].isin(top_methods)]
    resolved = sub[sub["LoanStatus"].isin(["PIF", "CHGOFF"])]
    ct = resolved.groupby(["ProcessingMethod", "FixedorVariableInterestInd"])["LoanStatus"].apply(
        lambda s: (s == "CHGOFF").mean() * 100
    ).unstack()
    fig, ax = plt.subplots(figsize=(9, 5))
    ct.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("chargeoff rate %")
    ax.set_title("Chargeoff rate: Fixed vs Variable rate, by top 6 ProcessingMethods")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(f"{OUT}/16_fixedvar_vs_chgoff_by_method.png", dpi=150)
    plt.close(fig)
    print("saved 16_fixedvar_vs_chgoff_by_method.png")

    # ------------------------------------------------------------------
    # 17. CollateralInd x chargeoff rate, plus median GrossApproval
    # ------------------------------------------------------------------
    t = chgoff_rate_by(pooled, "CollateralInd", min_n=1)
    med_amt = pooled.groupby("CollateralInd")["GrossApproval"].median()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(t.index.astype(str), t["chgoff_rate_pct"], color="#C44E52")
    axes[0].set_title("Chargeoff rate by CollateralInd")
    axes[1].bar(med_amt.index.astype(str), med_amt.values, color="#64B5CD")
    axes[1].set_title("Median GrossApproval by CollateralInd")
    fig.tight_layout()
    fig.savefig(f"{OUT}/17_collateral_vs_chgoff_amount.png", dpi=150)
    plt.close(fig)
    print("saved 17_collateral_vs_chgoff_amount.png")

    # ------------------------------------------------------------------
    # 18. SoldSecMrktInd x chargeoff rate
    # ------------------------------------------------------------------
    t = chgoff_rate_by(pooled, "SoldSecMrktInd", min_n=1)
    print("SoldSecMrktInd chargeoff rate:")
    print(t.to_string())

    # ------------------------------------------------------------------
    # 19. MonthsToChargeoff distribution by sector (top 6 by CHGOFF volume) and by vintage file
    # ------------------------------------------------------------------
    chgoff_rows = pooled[pooled["LoanStatus"] == "CHGOFF"].copy()
    chgoff_rows["MonthsToChargeoff"] = (
        (chgoff_rows["ChargeOffDate"] - chgoff_rows["ApprovalDate"]).dt.days / 30.44
    )
    top_sectors = chgoff_rows["NaicsSectorName"].value_counts().head(6).index
    fig, ax = plt.subplots(figsize=(10, 6))
    for sector in top_sectors:
        vals = chgoff_rows.loc[chgoff_rows["NaicsSectorName"] == sector, "MonthsToChargeoff"].dropna()
        ax.hist(vals, bins=40, histtype="step", label=sector, linewidth=1.5)
    ax.set_xlabel("Months from approval to chargeoff")
    ax.set_ylabel("count")
    ax.set_title("MonthsToChargeoff distribution, top 6 sectors by CHGOFF volume")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/19_monthstochgoff_by_sector.png", dpi=150)
    plt.close(fig)
    print("saved 19_monthstochgoff_by_sector.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for name in ORDER:
        vals = chgoff_rows.loc[chgoff_rows["vintage_file"] == name, "MonthsToChargeoff"].dropna()
        ax.hist(vals, bins=40, histtype="step", label=name, linewidth=1.5)
    ax.set_xlabel("Months from approval to chargeoff")
    ax.set_title("MonthsToChargeoff distribution, by file/vintage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/20_monthstochgoff_by_vintage.png", dpi=150)
    plt.close(fig)
    print("saved 20_monthstochgoff_by_vintage.png")

    # ------------------------------------------------------------------
    # 21. State (ProjectState) x chargeoff rate (top 20 states by volume)
    # ------------------------------------------------------------------
    top_states = pooled["ProjectState"].value_counts().head(20).index
    t = chgoff_rate_by(pooled[pooled["ProjectState"].isin(top_states)], "ProjectState", min_n=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(t.index, t["chgoff_rate_pct"], color="#937860")
    ax.set_xlabel("chargeoff rate % (of resolved)")
    ax.set_title("Chargeoff rate by state, top 20 states by loan volume")
    fig.tight_layout()
    fig.savefig(f"{OUT}/21_state_vs_chgoffrate.png", dpi=150)
    plt.close(fig)
    print("saved 21_state_vs_chgoffrate.png")

    # ------------------------------------------------------------------
    # 22. CANCLD rate by sector
    # ------------------------------------------------------------------
    total = pooled.groupby("NaicsSectorName").size()
    cancld = pooled[pooled["LoanStatus"] == "CANCLD"].groupby("NaicsSectorName").size().reindex(total.index).fillna(0)
    cancld_rate = (cancld / total * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(cancld_rate.index, cancld_rate.values, color="#8C8C8C")
    ax.set_xlabel("CANCLD rate % (of all loans in sector)")
    ax.set_title("Cancellation rate by sector")
    fig.tight_layout()
    fig.savefig(f"{OUT}/22_cancld_rate_by_sector.png", dpi=150)
    plt.close(fig)
    print("saved 22_cancld_rate_by_sector.png")

    print("\nALL DONE")


if __name__ == "__main__":
    main()
