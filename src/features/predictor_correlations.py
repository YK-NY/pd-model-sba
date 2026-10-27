"""
Correlation check across the continuous predictors, on the modeling
population. Run before binning to catch redundant pairs (e.g. loan amount
vs. guaranteed amount, which are structurally related -- guarantee is a
% of approval) before they're both fed into the model.
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

CONTINUOUS_VARS = ["GrossApproval", "SBAGuaranteedApproval", "InitialInterestRate",
                    "TermInMonths", "JobsSupported"]


def main():
    raw = load_model_base()
    df = remove_young_loans(raw, n_months=60)
    df = remove_cancelled_and_committed(df)
    df = assign_target(df, n_months=60)

    corr = df[CONTINUOUS_VARS].corr()
    print("=== correlation matrix, continuous predictors ===")
    print(corr.round(2).to_string())

    print("\n=== pairs with |correlation| >= 0.5 (worth a closer look) ===")
    pairs = corr.where(~corr.isna())
    for i, a in enumerate(CONTINUOUS_VARS):
        for b in CONTINUOUS_VARS[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= 0.5:
                print(f"  {a} x {b}: r={r:.2f}")

    Path(OUT).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(CONTINUOUS_VARS)))
    ax.set_yticks(range(len(CONTINUOUS_VARS)))
    ax.set_xticklabels(CONTINUOUS_VARS, rotation=45, ha="right")
    ax.set_yticklabels(CONTINUOUS_VARS)
    for i in range(len(CONTINUOUS_VARS)):
        for j in range(len(CONTINUOUS_VARS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, label="correlation")
    ax.set_title("Continuous predictor correlation -- modeling population")
    fig.tight_layout()
    fig.savefig(f"{OUT}/03_predictor_correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved {OUT}/03_predictor_correlation_heatmap.png")


if __name__ == "__main__":
    main()
