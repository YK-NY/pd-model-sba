"""
Scatterplots between continuous predictors, on the modeling population --
the visual complement to predictor_correlations.py's heatmap. A single
correlation number can hide the actual shape of a relationship (discrete
clustering, tiered/step patterns, curvature) -- this is just looking
directly at that shape for the pairs the correlation check flagged.

Sampled, not the full 543,123 rows: at that volume a plain scatter is just
an overplotted blob. SAMPLE_N points are drawn at random (fixed seed, so
this is reproducible) -- large enough to show the real shape, small enough
to stay readable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from target_construction import (
    load_model_base, remove_young_loans, remove_cancelled_and_committed, assign_target,
)

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = str(Path(__file__).resolve().parents[2] / "outputs" / "figures" / "pd_model")
SAMPLE_N = 8000
SEED = 42

PAIRS = [
    ("TermInMonths", "GrossApproval"),
    ("InitialInterestRate", "GrossApproval"),
    ("TermInMonths", "InitialInterestRate"),
    ("GrossApproval", "GuaranteePct"),
]


def build_population():
    raw = load_model_base()
    df = remove_young_loans(raw, n_months=60)
    df = remove_cancelled_and_committed(df)
    df = assign_target(df, n_months=60)
    df["GuaranteePct"] = df["SBAGuaranteedApproval"] / df["GrossApproval"] * 100
    return df


def main():
    df = build_population()
    sample = df.sample(n=SAMPLE_N, random_state=SEED)
    print(f"modeling population: {len(df)} rows, sampled {SAMPLE_N} for plotting (seed={SEED})")

    Path(OUT).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, (x, y) in zip(axes, PAIRS):
        ax.scatter(sample[x], sample[y], s=4, alpha=0.25, color="#4C72B0", edgecolors="none")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        r = df[[x, y]].corr().iloc[0, 1]
        ax.set_title(f"{x} vs {y}  (r={r:.2f}, full population)")
    fig.suptitle(f"Predictor scatterplots -- modeling population, n={SAMPLE_N} random sample (seed={SEED})")
    fig.tight_layout()
    fig.savefig(f"{OUT}/06_predictor_scatterplots.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/06_predictor_scatterplots.png")


if __name__ == "__main__":
    main()
