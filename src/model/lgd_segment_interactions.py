"""
LGD, continued: does loss severity change when two loan features are
looked at together?

On charged-off loans, this checks:
- severity by term length and collateral together
- severity by sector and collateral together
- whether severity differs by geography (same area-rate idea as the PD
  feature, measured on loss instead of default)
- which sectors can be grouped because their average loss is not
  meaningfully different. Same merge idea as the PD binning, but with a
  t-test on average loss, since loss is a number and not a yes/no outcome.

Population: every charged-off loan, as in lgd_severity_analysis.py. The
60-month age filter is not used: a defaulted loan's loss is already known,
whatever its age.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from target_construction import load_model_base

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = str(Path(__file__).resolve().parents[2] / "outputs" / "figures" / "lgd")
MIN_SECTOR_N = 100


def build_lgd_population():
    raw = load_model_base()
    chgoff = raw[raw["LoanStatus"] == "CHGOFF"].copy()
    chgoff["realized_lgd"] = chgoff["GrossChargeOffAmount"] / chgoff["GrossApproval"]
    chgoff["term_bucket"] = pd.cut(chgoff["TermInMonths"], bins=[0, 36, 60, 84, 120, 180, 300, 500])
    return chgoff


def term_x_collateral(df):
    g = df.groupby(["term_bucket", "CollateralInd"], observed=True)["realized_lgd"].agg(["size", "mean"])
    print("=== realized LGD: term bucket x CollateralInd ===")
    print(g.round(3).to_string())
    return g


def sector_x_collateral(df):
    g = df.groupby(["NaicsSectorName", "CollateralInd"], observed=True)["realized_lgd"].mean().unstack()
    print("\n=== realized LGD: sector x CollateralInd ===")
    print(g.round(3).to_string())
    return g


def geo_tier_lgd(df):
    """Reuses the same zip-3-with-state-backoff geo risk construction used
    for PD (src/features/woe_binning.py::add_geo_risk_pct), but measured
    against realized LGD instead of default rate -- does severity, not
    just default probability, vary geographically?"""
    df = df.copy()
    df["zip3"] = df["BorrZip"].str[:3]
    zip3_lgd = df.groupby("zip3")["realized_lgd"].agg(["size", "mean"])
    trusted_zip3 = zip3_lgd[zip3_lgd["size"] >= 30]["mean"]
    state_lgd = df.groupby("BorrState")["realized_lgd"].mean()
    zip3_lookup = df["zip3"].map(trusted_zip3)
    state_lookup = df["BorrState"].map(state_lgd)
    df["geo_lgd_estimate"] = zip3_lookup.fillna(state_lookup)

    df["geo_lgd_tier"] = pd.qcut(df["geo_lgd_estimate"], q=4, labels=["A (lowest)", "B", "C", "D (highest)"])
    g = df.groupby("geo_lgd_tier", observed=True)["realized_lgd"].agg(["size", "mean"])
    print("\n=== realized LGD by geo severity quartile (zip-3 w/ state backoff) ===")
    print(g.round(3).to_string())
    return g


def sector_rigor_merge(df, min_n=MIN_SECTOR_N, alpha=0.05):
    """Groups sectors whose average loss is not meaningfully different.

    Sectors are lined up from lowest to highest average loss. Neighbours
    are merged when either has fewer than min_n loans, or when a t-test says
    their average losses could just be chance (p >= alpha). This repeats until
    nothing more merges. What is left is the sector grouping used for the
    LGD sector tiers.

    Same idea as the PD binning merge, but with a t-test on average loss
    (a number) instead of a z-test on default rate (a yes/no outcome)."""
    g = df.groupby("NaicsSectorName", observed=True)["realized_lgd"].agg(["size", "mean", "std"])
    g = g.rename(columns={"size": "n"})
    g = g.sort_values("mean")
    rows = [{"label": idx, "members": [idx], "n": r["n"], "mean": r["mean"], "std": r["std"]}
            for idx, r in g.iterrows()]

    def values_for(members):
        return df[df["NaicsSectorName"].isin(members)]["realized_lgd"].values

    merged = True
    while merged and len(rows) > 1:
        merged = False
        i = 0
        while i < len(rows) - 1:
            a, b = rows[i], rows[i + 1]
            too_small = a["n"] < min_n or b["n"] < min_n
            if too_small:
                distinct = False
            else:
                _, p = ttest_ind(values_for(a["members"]), values_for(b["members"]), equal_var=False)
                distinct = p < alpha
            if too_small or not distinct:
                combined_members = a["members"] + b["members"]
                vals = values_for(combined_members)
                rows[i:i + 2] = [{"label": f'{a["label"]} | {b["label"]}', "members": combined_members,
                                   "n": len(vals), "mean": vals.mean(), "std": vals.std()}]
                merged = True
            else:
                i += 1
    result = pd.DataFrame(rows)[["label", "n", "mean", "std"]]
    print(f"\n=== sector, rigor pass: event-count floor (n>={min_n}) + t-test merge (alpha={alpha}) ===")
    print(result.round(3).to_string(index=False))
    return result


def main():
    df = build_lgd_population()

    term_x_collateral(df)
    sc = sector_x_collateral(df)
    geo_tier_lgd(df)
    sector_rigor_merge(df)

    Path(OUT).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    sc_sorted = sc.loc[sc.mean(axis=1).sort_values().index]
    sc_sorted.plot(kind="barh", ax=ax, color=["#DD8452", "#4C72B0"])
    ax.set_xlabel("mean realized LGD")
    ax.set_title("Realized LGD by sector x CollateralInd")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/02_lgd_sector_x_collateral.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved {OUT}/02_lgd_sector_x_collateral.png")


if __name__ == "__main__":
    main()
