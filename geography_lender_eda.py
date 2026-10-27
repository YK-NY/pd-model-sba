"""
Geography risk-tier feature (zip-3 level, event-count floor + quantile
bins) and lender concentration (LocationID, top-N by volume).

Run: python geography_lender_eda.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sector_eda import load_clean_sector_df, ORDER

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/figures"
MIN_ZIP3_EVENTS = 30       # zip-3s below this many resolved loans are too thin to trust
N_RISK_TIERS = 4           # quantile tiers, labelled A (lowest risk) .. D (highest)


def build_geo_tiers(resolved, min_events=MIN_ZIP3_EVENTS, n_tiers=N_RISK_TIERS):
    """zip-3 level chargeoff rate, event-floor filtered, quantile-binned into tiers A (lowest risk) .. D (highest)."""
    resolved = resolved.copy()
    resolved["zip3"] = resolved["BorrZip"].str[:3]
    g = resolved.groupby("zip3")["target"]
    table = pd.DataFrame({"n": g.size(), "chgoff_rate_pct": g.mean() * 100})
    table = table[table["n"] >= min_events]

    # chr(65) is "A" -- one letter per tier, so labels stretch if n_tiers changes
    labels = [chr(65 + i) for i in range(n_tiers)]  # A, B, C, D...
    table["risk_tier"] = pd.qcut(table["chgoff_rate_pct"], q=n_tiers, labels=labels)
    return table


def main():
    frames = [load_clean_sector_df(n) for n in ORDER]
    pooled = pd.concat(frames, ignore_index=True)
    resolved = pooled[pooled["LoanStatus"].isin(["PIF", "CHGOFF"])].copy()

    # ------------------------------------------------------------------
    # Geography risk tiers
    # ------------------------------------------------------------------
    geo = build_geo_tiers(resolved, min_events=30, n_tiers=4)
    print(f"zip-3 units with >=30 events: {len(geo)} (out of {resolved['BorrZip'].str[:3].nunique()} total)")
    print("\nrisk tier summary (min chgoff_rate_pct, max, n zip3 units, total loans):")
    tier_summary = geo.groupby("risk_tier").agg(
        min_rate=("chgoff_rate_pct", "min"), max_rate=("chgoff_rate_pct", "max"),
        n_zip3=("n", "count"), total_loans=("n", "sum"),
    )
    print(tier_summary)

    fig, ax = plt.subplots(figsize=(8, 5))
    tier_summary["total_loans"].plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("total loans")
    ax.set_title("Geography risk tiers (zip-3, event-floor >=30): loan volume per tier")
    fig.tight_layout()
    fig.savefig(f"{OUT}/29_geo_risk_tiers.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/29_geo_risk_tiers.png")

    # coverage check: what % of loans fall in a zip3 with enough events to be tiered?
    covered_zip3 = set(geo.index)
    resolved["zip3"] = resolved["BorrZip"].str[:3]
    coverage = resolved["zip3"].isin(covered_zip3).mean() * 100
    print(f"\ncoverage: {coverage:.1f}% of resolved loans fall in a zip3 with enough events to be tiered")

    # ------------------------------------------------------------------
    # Lender concentration (LocationID, top-N by volume)
    # ------------------------------------------------------------------
    print("\n=== Lender concentration: top 20 LocationID by volume ===")
    g = resolved.groupby("LocationID")["target"]
    lender_table = pd.DataFrame({"n": g.size(), "chgoff_rate_pct": (g.mean() * 100).round(1)})
    lender_table = lender_table.sort_values("n", ascending=False)
    top20 = lender_table.head(20)
    print(top20)
    total_loans = resolved["LocationID"].notna().sum()
    print(f"\ntop 20 lenders = {top20['n'].sum()} loans, {top20['n'].sum()/total_loans*100:.1f}% of all resolved loans")
    print(f"total distinct lenders: {resolved['LocationID'].nunique()}")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top20.index.astype(str), top20["chgoff_rate_pct"], color="#DD8452")
    ax.set_xlabel("chargeoff rate %")
    ax.set_title("Top 20 lenders by volume: chargeoff rate")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(f"{OUT}/30_top_lenders_chgoffrate.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/30_top_lenders_chgoffrate.png")


if __name__ == "__main__":
    main()
