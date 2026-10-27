"""
InitialInterestRate vs. TermInMonths vs. chargeoff rate -- does rate
already "price in" term-length risk, or does term carry independent
signal once rate is controlled for? Finding: rate and term do not substitute for each other (correlation -0.18); each carries independent signal.

Run: python rate_term_eda.py
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


def main():
    frames = []
    for name in ORDER:
        df = load_clean_sector_df(name)
        df["vintage_file"] = name
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)

    resolved = pooled[pooled["LoanStatus"].isin(["PIF", "CHGOFF"])].copy()
    rated = resolved.dropna(subset=["InitialInterestRate"])
    print(f"resolved loans with a rate: {len(rated)} of {len(resolved)} resolved "
          f"({len(rated)/len(resolved)*100:.1f}%)")

    # 1. raw correlation between rate and term
    corr = rated["InitialInterestRate"].corr(rated["TermInMonths"])
    print(f"\nPearson correlation, InitialInterestRate vs TermInMonths: {corr:.3f}")

    # 2. does rate predict term-bucket? (median rate per term bucket)
    term_bins = [0, 60, 84, 120, 180, 240, np.inf]
    term_labels = ["<=60mo", "61-84mo", "85-120mo", "121-180mo", "181-240mo", "240mo+"]
    rated["term_bucket"] = pd.cut(rated["TermInMonths"], bins=term_bins, labels=term_labels)
    print("\nmedian InitialInterestRate by term bucket:")
    print(rated.groupby("term_bucket")["InitialInterestRate"].agg(["median", "mean", "count"]))

    # 3. the actual question: does chargeoff rate vary by term WITHIN a rate
    # bucket, and by rate WITHIN a term bucket? If rate already "prices in"
    # term risk, term's effect should mostly disappear once rate is fixed.
    rate_bins = [0, 5, 6, 7, 8, 10, np.inf]
    rate_labels = ["<=5%", "5-6%", "6-7%", "7-8%", "8-10%", "10%+"]
    rated["rate_bucket"] = pd.cut(rated["InitialInterestRate"], bins=rate_bins, labels=rate_labels)

    pivot_rate = pd.crosstab(rated["term_bucket"], rated["rate_bucket"],
                              values=rated["target"], aggfunc="mean") * 100
    pivot_n = pd.crosstab(rated["term_bucket"], rated["rate_bucket"])

    print("\nchargeoff rate % by term bucket (rows) x rate bucket (cols):")
    print(pivot_rate.round(1))
    print("\ncell counts:")
    print(pivot_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(pivot_rate.values, aspect="auto", cmap="Reds")
    ax.set_xticks(range(len(pivot_rate.columns)))
    ax.set_xticklabels(pivot_rate.columns)
    ax.set_yticks(range(len(pivot_rate.index)))
    ax.set_yticklabels(pivot_rate.index)
    ax.set_xlabel("InitialInterestRate bucket")
    ax.set_ylabel("TermInMonths bucket")
    ax.set_title("Chargeoff rate % by term x rate (pooled, resolved loans only)")
    for i in range(len(pivot_rate.index)):
        for j in range(len(pivot_rate.columns)):
            val = pivot_rate.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="chargeoff rate %")
    fig.tight_layout()
    fig.savefig(f"{OUT}/23_rate_x_term_x_chgoffrate.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved {OUT}/23_rate_x_term_x_chgoffrate.png")

    return rated


EXPRESS_TYPE_METHODS = {"SBAEXPRESSPROGRAM", "COMMUNITYEXPRESS"}


def by_method_group(rated):
    """
    Same term x rate x chargeoff-rate heatmap, split into Express-type vs
    Other ProcessingMethod -- tests whether the term effect (shorter term =
    higher chargeoff, see main()) survives WITHIN each group, or whether
    it's fully explained by term just proxying for Express-type/collateral
    status. rated = output of main().
    """
    rated = rated.copy()
    rated["method_group"] = rated["ProcessingMethod"].apply(
        lambda m: "Express-type" if m in EXPRESS_TYPE_METHODS else "Other"
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, group in zip(axes, ["Express-type", "Other"]):
        sub = rated[rated["method_group"] == group]
        pivot_rate = pd.crosstab(sub["term_bucket"], sub["rate_bucket"],
                                  values=sub["target"], aggfunc="mean") * 100
        im = ax.imshow(pivot_rate.values, aspect="auto", cmap="Reds", vmin=0, vmax=40)
        ax.set_xticks(range(len(pivot_rate.columns)))
        ax.set_xticklabels(pivot_rate.columns)
        ax.set_yticks(range(len(pivot_rate.index)))
        ax.set_yticklabels(pivot_rate.index)
        ax.set_xlabel("InitialInterestRate bucket")
        ax.set_title(f"{group} (n={len(sub)})")
        for i in range(len(pivot_rate.index)):
            for j in range(len(pivot_rate.columns)):
                val = pivot_rate.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="chargeoff rate %")
    axes[0].set_ylabel("TermInMonths bucket")
    fig.suptitle("Chargeoff rate % by term x rate, split by ProcessingMethod group")
    fig.tight_layout()
    fig.savefig(f"{OUT}/26_rate_x_term_by_methodgroup.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/26_rate_x_term_by_methodgroup.png")


if __name__ == "__main__":
    rated = main()
    by_method_group(rated)
