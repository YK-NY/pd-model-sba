"""
PD model, step 1: vintage-normalized target construction.

Fixes a bias in the raw resolved-basis rate (CHGOFF / (PIF+CHGOFF)): that
rate isn't comparable across vintages, because CHGOFF tends to happen
faster than PIF, so recent loans look artificially riskier than they are
just from lack of time, not real credit deterioration. The chart outputs/figures/pd_model/01_vintage_normalization_validation.png shows the effect.

Fix: only judge a loan once it's had a full 60-month window to prove
itself, and ask the same question of every loan -- "did it default within
its first 60 months?"

Pipeline, in order:
  1. remove_young_loans      -- drop anything under 60 months old
  2. remove_cancelled_and_committed -- drop CANCLD/COMMIT (never disbursed)
  3. assign_target           -- label what's left: 1/0, plus a plain-text
                                 outcome description for every row

Scope: 2010-2019 and 2020-Curr only -- 2000-2009 is held out for now

Kept entirely separate from the EDA scripts (sector_eda.py etc.) -- this
is model-build code. Does not modify, and is not imported by, any of them.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sector_eda import load_clean_sector_df

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_FILES = ["2010_2019", "2020_Curr"]  # 2000-2009 held out for now
# resolved relative to this file, not the caller's cwd -- prevents a stray
# outputs/ folder appearing wherever this script happens to be run from
OUT = str(Path(__file__).resolve().parents[2] / "outputs" / "figures" / "pd_model")
ARTIFACTS_DIR = str(Path(__file__).resolve().parents[2] / "outputs" / "model_artifacts")
COLORS = {"2010_2019": "#DD8452", "2020_Curr": "#55A868"}


def load_model_base(files=MODEL_FILES):
    """Loads and concatenates the canonical cleaned dataframes for the given files."""
    frames = []
    for name in files:
        df = load_clean_sector_df(name)
        df["vintage_file"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def save_young_resolved_loans(raw, n_months=60):
    """
    Set-aside, not part of the modeling pipeline: loans under n_months old
    that are ALREADY confirmed PIF or CHGOFF. Their outcome is fully
    certain even though they're young, but they get dropped by
    remove_young_loans anyway (which drops purely by age). Saved as a
    pickle so they can be pulled back in later without re-deriving them.
    Call this BEFORE remove_young_loans, on the unfiltered dataframe.
    """
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    months_since_approval = (raw["AsOfDate"] - raw["ApprovalDate"]).dt.days / 30.44
    is_young = months_since_approval < n_months
    is_resolved = raw["LoanStatus"].isin(["PIF", "CHGOFF"])

    young_resolved = raw[is_young & is_resolved].copy()
    path = f"{ARTIFACTS_DIR}/young_resolved_loans.pkl"
    young_resolved.to_pickle(path)
    print(f"saved {len(young_resolved)} young (<{n_months}mo) but already-resolved loans to {path}")
    return young_resolved


def remove_young_loans(df, n_months=60):
    """
    STEP 1. Keep only loans approved at least n_months before AsOfDate.
    Drops every younger loan, regardless of status -- PIF, CHGOFF, EXEMPT,
    anything. Nothing younger than n_months is used again after this.    
    """
    # ApprovalDate + n_months <= AsOfDate  
    months_since_approval = (df["AsOfDate"] - df["ApprovalDate"]).dt.days / 30.44
    return df[months_since_approval >= n_months].copy()


def remove_cancelled_and_committed(df):
    """
    STEP 2. Drop CANCLD (cancelled before disbursement) and COMMIT
    (approved, not yet disbursed). Neither ever had real money at risk, so
    neither has a repayment outcome to measure.
    """
    return df[~df["LoanStatus"].isin(["CANCLD", "COMMIT"])].copy()


def assign_target(df, n_months=60):
    """
    STEP 3. Run on the output of steps 1+2 -- every row here is already
    old enough and already one of PIF / CHGOFF / EXEMPT. Adds:
      - target_60mo: 1 if the loan charged off within its first n_months,
        0 otherwise. Every row gets a value -- nothing is NA at this point.
      - outcome_detail: plain-text description of why, for every row
        (not a 0/1 code):
          "DEFAULTED_WITHIN_WINDOW" -- target is 1
          "PAID_IN_FULL"            -- PIF, target is 0
          "STILL_OPEN"              -- EXEMPT, target is 0, but this loan
                                       is still active and could still
                                       default later -- the one case that
                                       isn't fully certain
          "DEFAULTED_AFTER_WINDOW"  -- CHGOFF, but after n_months, so
                                       target is 0 for this question, even
                                       though we know it went bad eventually
    """
    df = df.copy()
    months_to_chargeoff = (df["ChargeOffDate"] - df["ApprovalDate"]).dt.days / 30.44
    defaulted_in_window = (df["LoanStatus"] == "CHGOFF") & (months_to_chargeoff <= n_months)

    df["target_60mo"] = defaulted_in_window.astype(int)

    df["outcome_detail"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[defaulted_in_window, "outcome_detail"] = "DEFAULTED_WITHIN_WINDOW"
    df.loc[~defaulted_in_window & (df["LoanStatus"] == "PIF"), "outcome_detail"] = "PAID_IN_FULL"
    df.loc[~defaulted_in_window & (df["LoanStatus"] == "EXEMPT"), "outcome_detail"] = "STILL_OPEN"
    df.loc[~defaulted_in_window & (df["LoanStatus"] == "CHGOFF"), "outcome_detail"] = "DEFAULTED_AFTER_WINDOW"
    return df


def vintage_normalized_rate(df, by, min_n=1):
    """N-month cumulative default rate by whatever grouping is passed in `by`."""
    g = df.groupby(by)["target_60mo"]
    out = pd.DataFrame({"n": g.size(), "default_rate_pct": (g.mean() * 100).round(1)})
    return out[out["n"] >= min_n]


def _old_resolved_rate_by_year(df):
    """The naive, biased resolved-basis rate -- for the validation chart only."""
    resolved = df[df["LoanStatus"].isin(["PIF", "CHGOFF"])]
    g = resolved.groupby(["vintage_file", "ApprovalFY"])["target"]
    return (g.mean() * 100).rename("old_resolved_rate_pct")


def validate_against_old_rate(df, n_months=60):
    """
    Plots old (biased, resolved-basis) vs. new (N-month window) rate by
    ApprovalFY, side by side -- the direct visual proof the fix works
    (the artificial rise-then-crash near each file's tail should flatten).
    """
    Path(OUT).mkdir(parents=True, exist_ok=True)
    old_rate = _old_resolved_rate_by_year(df)
    new_rate = vintage_normalized_rate(df, ["vintage_file", "ApprovalFY"], min_n=30)["default_rate_pct"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for name in MODEL_FILES:
        o = old_rate[name]
        axes[0].plot(o.index, o.values, marker="o", label=name, color=COLORS[name])
        n = new_rate[name]
        axes[1].plot(n.index, n.values, marker="o", label=name, color=COLORS[name])
    axes[0].set_title("OLD: naive resolved-basis rate")
    axes[1].set_title(f"NEW: {n_months}-month fixed-window rate")
    for ax in axes:
        ax.set_xlabel("ApprovalFY")
        ax.set_ylabel("default rate %")
        ax.legend()
    fig.suptitle(f"{n_months}-month window validation: naive rate vs. {n_months}-month rate by ApprovalFY")
    fig.tight_layout()
    fig.savefig(f"{OUT}/01_vintage_normalization_validation.png", dpi=150)
    plt.close(fig)
    print(f"saved {OUT}/01_vintage_normalization_validation.png")


def main():
    raw = load_model_base()
    print(f"loaded {len(raw)} rows across {MODEL_FILES}")

    save_young_resolved_loans(raw, n_months=60)

    df = remove_young_loans(raw, n_months=60)
    print(f"after removing loans <60 months old: {len(df)} rows")

    df = remove_cancelled_and_committed(df)
    print(f"after removing CANCLD/COMMIT: {len(df)} rows")

    df = assign_target(df, n_months=60)
    print()
    print("=== outcome_detail counts (every row has one, none are NA) ===")
    print(df["outcome_detail"].value_counts())
    print()
    print("=== N-month default rate by ApprovalFY x vintage_file ===")
    print(vintage_normalized_rate(df, ["vintage_file", "ApprovalFY"], min_n=30).to_string())

    validate_against_old_rate(df, n_months=60)


if __name__ == "__main__":
    main()
