"""
ProcessingMethod x ApprovalFY coverage -- which loan programs existed in
which years. Built retroactively, after an XGBoost run crashed on a
ProcessingMethod category the training years never saw (COMMUNITYADVANTAGE-
RECOVERYLOAN, SBAEXPRESSBRIDGELOAN -- both introduced after 2018). A
crosstab like this, built during EDA, would have caught the train/test
category mismatch before it ever became a modeling bug -- SBA loan
programs get introduced and discontinued over time, same as any other
categorical field with a time dimension.
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

OUT = str(Path(__file__).resolve().parents[2] / "outputs" / "figures")


def build_population():
    raw = load_model_base()
    df = remove_young_loans(raw, n_months=60)
    df = remove_cancelled_and_committed(df)
    df = assign_target(df, n_months=60)
    return df


def main():
    df = build_population()
    counts = df.groupby(["ProcessingMethod", "ApprovalFY"], observed=True).size().unstack(fill_value=0)
    # order rows by total volume, most common at top
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]

    print("=== ProcessingMethod x ApprovalFY loan counts ===")
    print(counts.to_string())

    print("\n=== methods present in 2019-2021 but not 2010-2018 (or vice versa) ===")
    train_years = [c for c in counts.columns if c <= 2018]
    test_years = [c for c in counts.columns if c > 2018]
    train_only = counts[(counts[train_years].sum(axis=1) > 0) & (counts[test_years].sum(axis=1) == 0)]
    test_only = counts[(counts[train_years].sum(axis=1) == 0) & (counts[test_years].sum(axis=1) > 0)]
    print(f"train-only (discontinued before 2019): {list(train_only.index)}")
    print(f"test-only (introduced after 2018): {list(test_only.index)}")

    Path(OUT).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(counts.values, aspect="auto", cmap="Blues",
                    norm=matplotlib.colors.LogNorm(vmin=1, vmax=counts.values.max()))
    ax.set_xticks(range(len(counts.columns)))
    ax.set_xticklabels(counts.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(counts.index)))
    ax.set_yticklabels(counts.index, fontsize=8)
    ax.axvline(x=train_years[-1] - counts.columns[0] + 0.5, color="red", linewidth=1.5, linestyle="--")
    ax.set_title("Loan count by ProcessingMethod x ApprovalFY (log scale; red line = train/test split)")
    fig.colorbar(im, label="loan count (log scale)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/37_processingmethod_by_year_coverage.png", dpi=150)
    plt.close(fig)
    print(f"\nsaved {OUT}/37_processingmethod_by_year_coverage.png")


if __name__ == "__main__":
    main()
