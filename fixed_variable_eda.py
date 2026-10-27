"""
FixedorVariableInterestInd vs. ProcessingMethod, chargeoff rate, and term.
Variable-rate loans are far more common than fixed across all three files
-- this checks whether that split also carries a risk/structural signal.

Run: python fixed_variable_eda.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sector_eda import load_clean_sector_df, ORDER

import pandas as pd

pd.set_option("display.width", 160)


def main():
    for name in ORDER:
        df = load_clean_sector_df(name)
        print(f"\n{'='*25} {name} {'='*25}")

        print("\nFixedorVariableInterestInd overall mix:")
        print(df["FixedorVariableInterestInd"].value_counts(dropna=False))

        resolved = df[df["LoanStatus"].isin(["PIF", "CHGOFF"])]
        print("\nchargeoff rate by Fixed/Variable (resolved only):")
        g = resolved.groupby("FixedorVariableInterestInd")["target"]
        print(pd.DataFrame({"n_resolved": g.size(), "chgoff_rate_pct": (g.mean() * 100).round(2)}))

        print("\nmedian/mean TermInMonths by Fixed/Variable:")
        print(df.groupby("FixedorVariableInterestInd")["TermInMonths"].agg(["median", "mean"]).round(1))

        print("\nFixed/Variable mix by top 8 ProcessingMethod (by volume):")
        top_methods = df["ProcessingMethod"].value_counts().head(8).index
        ct = pd.crosstab(df.loc[df["ProcessingMethod"].isin(top_methods), "ProcessingMethod"],
                          df.loc[df["ProcessingMethod"].isin(top_methods), "FixedorVariableInterestInd"],
                          normalize="index") * 100
        print(ct.round(1))


if __name__ == "__main__":
    main()
