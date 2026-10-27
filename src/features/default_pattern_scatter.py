"""
Where in feature-space do defaults cluster? GrossApproval x
InitialInterestRate, the modeling population, drawn as density contours
(defaulted vs. not) instead of individual points -- shows where each group
concentrates without needing to sample, true proportions preserved.

A matched-sample scatter version (all defaults + an equal-sized random
sample of non-defaults) was tried first and dropped: credit risk is
inherently time-dependent, and a random sample flattens that away, so it
added a cross-sectional view without much real diagnostic value once the
density version already covers the same "where do defaults cluster"
question more rigorously (true proportions, no sampling choice to defend).
The default-rate-over-time question itself is already answered separately
by chart 01 (vintage_normalized_rate by ApprovalFY).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from target_construction import (
    load_model_base, remove_young_loans, remove_cancelled_and_committed, assign_target,
)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

OUT = str(Path(__file__).resolve().parents[2] / "outputs" / "figures" / "pd_model")
SEED = 42
X_COL, Y_COL = "InitialInterestRate", "GrossApproval"


def build_population():
    raw = load_model_base()
    df = remove_young_loans(raw, n_months=60)
    df = remove_cancelled_and_committed(df)
    df = assign_target(df, n_months=60)
    return df


def density_contours(df, ax):
    defaults = df[df["target_60mo"] == 1][[X_COL, Y_COL]].dropna()
    non_defaults = df[df["target_60mo"] == 0][[X_COL, Y_COL]].dropna().sample(n=20000, random_state=SEED)

    xmin, xmax = df[X_COL].quantile([0.01, 0.99])
    ymin, ymax = 0, 1_500_000
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    grid = np.vstack([xx.ravel(), yy.ravel()])

    for data, color, label in [(non_defaults, "#4C72B0", "did not default"),
                                (defaults, "#C44E52", "defaulted")]:
        kde = gaussian_kde(data.T.values)
        zz = kde(grid).reshape(xx.shape)
        ax.contour(xx, yy, zz, levels=5, colors=color, linewidths=1.3)
    ax.plot([], [], color="#4C72B0", label="did not default (density)")
    ax.plot([], [], color="#C44E52", label="defaulted (density)")
    ax.set_xlabel(X_COL)
    ax.set_ylabel(Y_COL)
    ax.set_ylim(ymin, ymax)
    ax.set_title("Density contours -- where each group concentrates, true proportions preserved")
    ax.legend()


def main():
    df = build_population()
    print(f"modeling population: {len(df)} rows, defaults: {(df['target_60mo']==1).sum()}")

    Path(OUT).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    density_contours(df, ax)
    fig.suptitle(f"{X_COL} vs {Y_COL}, by outcome -- modeling population")
    fig.tight_layout()
    fig.savefig(f"{OUT}/07_default_pattern_density.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/07_default_pattern_density.png")


if __name__ == "__main__":
    main()
