"""
Chargeoff rate by sector AND ApprovalFY jointly -- checks whether the
elevated 2000-2009 resolved-basis chargeoff rate (over 20% in several
sectors) is concentrated in specific years/sectors or spread across the
whole decade. Finding: the pattern is a measurement artifact of fixed-snapshot resolved rates, present at the tail of all three files, not a recession effect alone.

Run: python sector_year_chargeoff_eda.py
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
COLORS = {"2000_2009": "#4C72B0", "2010_2019": "#DD8452", "2020_Curr": "#55A868"}


def chgoff_rate_by_year(df, min_n=30):
    resolved = df[df["LoanStatus"].isin(["PIF", "CHGOFF"])]
    g = resolved.groupby("ApprovalFY")["target"]
    n = g.size()
    rate = g.mean() * 100
    return rate[n >= min_n]


def main():
    # 1. chargeoff rate by year, all three files on one timeline
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in ORDER:
        df = load_clean_sector_df(name)
        rate = chgoff_rate_by_year(df)
        ax.plot(rate.index, rate.values, marker="o", label=name, color=COLORS[name])
    ax.set_xlabel("ApprovalFY")
    ax.set_ylabel("chargeoff rate % (of resolved)")
    ax.set_title("Chargeoff rate by ApprovalFY, resolved-basis, across all 3 files")
    ax.axvline(2001, color="gray", linestyle=":", linewidth=1, label="2001 recession")
    ax.axvline(2008, color="gray", linestyle="--", linewidth=1, label="2008 recession")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/24_chgoffrate_by_year_all_files.png", dpi=150)
    plt.close(fig)
    print("saved 24_chgoffrate_by_year_all_files.png")

    # 2. sector x year heatmap, 2000-2009 specifically (where the puzzle is)
    df = load_clean_sector_df("2000_2009")
    resolved = df[df["LoanStatus"].isin(["PIF", "CHGOFF"])]
    pivot_rate = pd.crosstab(resolved["NaicsSectorName"], resolved["ApprovalFY"],
                              values=resolved["target"], aggfunc="mean") * 100
    pivot_n = pd.crosstab(resolved["NaicsSectorName"], resolved["ApprovalFY"])
    # sector order by overall loan volume, most to least
    order = df["NaicsSectorName"].value_counts().index
    pivot_rate = pivot_rate.reindex(order)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(pivot_rate.values, aspect="auto", cmap="Reds", vmin=0, vmax=40)
    ax.set_xticks(range(len(pivot_rate.columns)))
    ax.set_xticklabels(pivot_rate.columns, rotation=45)
    ax.set_yticks(range(len(pivot_rate.index)))
    ax.set_yticklabels(pivot_rate.index, fontsize=8)
    ax.set_title("2000-2009: chargeoff rate % by sector x ApprovalFY (resolved-basis)")
    fig.colorbar(im, ax=ax, label="chargeoff rate %")
    fig.tight_layout()
    fig.savefig(f"{OUT}/25_sector_x_year_chgoffrate_2000_2009.png", dpi=150)
    plt.close(fig)
    print("saved 25_sector_x_year_chgoffrate_2000_2009.png")
    print("\n2000-2009 chargeoff rate % by sector x year:")
    print(pivot_rate.round(1).to_string())
    print("\ncell counts (resolved loans):")
    print(pivot_n.reindex(order).to_string())


def decade_baseline_table():
    """
    Is the 2000-2009 file's elevated chargeoff rate just the 2006-2008
    recession pulling up the decade average, or is the underlying baseline
    (excluding those years) also elevated vs. the other two files? Finding: the baseline is elevated too. Even without 2006-2008, 2000-2005 runs at about 18%, more than double FY2010-2019 (8%).
    """
    df = load_clean_sector_df("2000_2009")
    resolved = df[df["LoanStatus"].isin(["PIF", "CHGOFF"])]

    slices = {
        "2000-2009, all years": resolved,
        "2000-2005 only (pre-recession)": resolved[resolved["ApprovalFY"] <= 2005],
        "2000-2009 excluding 2006-2008": resolved[~resolved["ApprovalFY"].isin([2006, 2007, 2008])],
        "2006-2008 only": resolved[resolved["ApprovalFY"].isin([2006, 2007, 2008])],
    }
    for name in ["2010_2019", "2020_Curr"]:
        d = load_clean_sector_df(name)
        slices[f"{name}, all years (reference)"] = d[d["LoanStatus"].isin(["PIF", "CHGOFF"])]

    rows = []
    for label, s in slices.items():
        rows.append({"slice": label, "chgoff_rate_pct": s["target"].mean() * 100, "n": len(s)})
    table = pd.DataFrame(rows).set_index("slice")
    table["chgoff_rate_pct"] = table["chgoff_rate_pct"].round(2)
    print(table.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0"] * 4 + ["#888888", "#888888"]
    ax.barh(table.index, table["chgoff_rate_pct"], color=colors)
    ax.set_xlabel("chargeoff rate % (of resolved)")
    ax.set_title("2000-2009 baseline vs. recession years vs. other files")
    for i, v in enumerate(table["chgoff_rate_pct"]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/27_decade_baseline_vs_recession.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved {OUT}/27_decade_baseline_vs_recession.png")

    return table


if __name__ == "__main__":
    main()
    decade_baseline_table()
