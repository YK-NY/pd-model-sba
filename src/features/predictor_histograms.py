"""
Distribution histograms for the continuous predictors, on the actual
modeling population (post target_construction: seasoned, non-CANCLD/COMMIT
loans). Purely diagnostic -- run before WOE/IV binning to see each
variable's shape and decide where natural split points might be.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from target_construction import (
    load_model_base, remove_young_loans, remove_cancelled_and_committed, assign_target,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = str(Path(__file__).resolve().parents[2] / "outputs" / "figures" / "pd_model")

CONTINUOUS_VARS = {
    "GrossApproval": "Loan amount approved ($)",
    "SBAGuaranteedApproval": "SBA-guaranteed amount ($)",
    "InitialInterestRate": "Interest rate at origination (%)",
    "TermInMonths": "Loan term (months)",
    "JobsSupported": "Jobs supported (count)",
}


def build_modeling_population():
    raw = load_model_base()
    df = remove_young_loans(raw, n_months=60)
    df = remove_cancelled_and_committed(df)
    df = assign_target(df, n_months=60)
    return df


def plot_histograms(df):
    Path(OUT).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for ax, (col, label) in zip(axes, CONTINUOUS_VARS.items()):
        vals = df[col].dropna()
        ax.hist(vals, bins=60, color="#4C72B0", edgecolor="none")
        ax.set_title(label)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        n_missing = df[col].isna().sum()
        pct_missing = 100 * n_missing / len(df)
        ax.text(0.98, 0.95, f"missing: {pct_missing:.1f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="gray")
    for ax in axes[len(CONTINUOUS_VARS):]:
        ax.axis("off")
    fig.suptitle("Predictor distributions -- modeling population (seasoned, non-CANCLD/COMMIT loans)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/02_predictor_distributions.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/02_predictor_distributions.png")


def print_summary(df):
    print("=== continuous predictor summary (modeling population) ===")
    print(df[list(CONTINUOUS_VARS)].describe().to_string())
    print()
    print("=== missing % ===")
    print((df[list(CONTINUOUS_VARS)].isna().mean() * 100).round(2).to_string())


def main():
    df = build_modeling_population()
    print(f"modeling population: {len(df)} rows")
    print_summary(df)
    plot_histograms(df)


if __name__ == "__main__":
    main()
