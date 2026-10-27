"""
Which numeric predictors have a monotonic (roughly "linear-shaped") risk
relationship, vs. a genuinely non-monotonic/tiered one -- checked directly
against each feature's fitted WOE bin sequence, on the training
loans (ApprovalFY <= 2018), rather than assumed.

Why this matters: logistic regression + WOE binning doesn't require a
feature to be monotonic to work (each bin gets its own independent WOE
value regardless of shape) -- but a monotonic feature's story is much
easier to state plainly ("risk rises smoothly as X increases") than a
tiered/oscillating one, which usually points to a structural/rule-based
cause worth naming rather than just "the data is noisy."
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRAIN_YEARS_MAX
from woe_binning import build_modeling_population, bin_numeric

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = str(Path(__file__).resolve().parents[2] / "outputs" / "figures" / "pd_model")
NUMERIC_FEATURES = ["TermInMonths", "GrossApproval", "InitialInterestRate", "GuaranteePct", "geo_risk_pct"]


def monotonicity_label(woe):
    # Step from each bin to the next, e.g. WOE [-1, 0, 1, 0.5, 1.2] -> steps [+1, +1, -0.5, +0.7].
    diffs = [woe[i + 1] - woe[i] for i in range(len(woe) - 1)]
    # A reversal is two neighbouring steps going opposite ways (one up, one down:
    # their product is negative). The example above has 2 reversals.
    # 0 reversals = monotonic, 1 = mostly monotonic, 2+ = non-monotonic / tiered.
    n_reversals =sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0)
    if n_reversals == 0:
        return "monotonic"
    elif n_reversals <= 1:
        return "mostly monotonic (1 small reversal)"
    else:
        return "non-monotonic / tiered"


def main():
    df = build_modeling_population()
    df = df[df["ApprovalFY"] <= TRAIN_YEARS_MAX]   # bins are judged on training loans only

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        t = bin_numeric(df, col)
        woe = t["woe"].tolist()
        label = monotonicity_label(woe)
        print(f"{col}: {label}")
        print("  WOE sequence:", [round(w, 2) for w in woe])

        ax.plot(range(len(woe)), woe, marker="o", color="#4C72B0")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(f"{col}\n({label})", fontsize=10)
        ax.set_xlabel("bin (low to high value)")
        ax.set_ylabel("WOE")

    fig.suptitle("WOE by bin, in value order -- monotonic vs. non-monotonic numeric predictors")
    fig.tight_layout()
    Path(OUT).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}/10_woe_monotonicity_check.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved {OUT}/10_woe_monotonicity_check.png")


if __name__ == "__main__":
    main()
