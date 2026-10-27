"""
PD model, step 2: WOE/IV binning.

No binning library used (no optbinning, no scorecardpy) -- written by hand
on purpose. Off-the-shelf binners default to equal-width or equal-frequency
cuts that ignore the target entirely, then leave you to manually inspect
and re-merge bins that don't actually differ in risk. Here, every merge
decision is driven directly by the target (does this bin's default rate
differ from its neighbor's, by more than sampling noise could explain?),
so the bins that come out are only as fine as the data can actually support
-- coarser for flat/thin features, finer for features with real, provable
risk separation.

Method: start each feature split into many small slices
(numeric: equal-frequency; categorical: one slice per category, sorted by
default rate). Walk neighboring slices in order and merge two together if
either (a) one of them has fewer than MIN_BIN_N loans -- too small to
trust on its own, or (b) a two-proportion z-test says their default rates
aren't statistically distinguishable. Repeat until no merges are left to
make. What's left is the final bin set.

Each final bin gets a WOE (weight of evidence) value -- how much riskier
or safer that bin is than the population average, on a scale comparable
across every feature. Each feature gets one IV (information value) score
-- how much that feature as a whole separates good loans from bad ones,
used to rank/screen features before modeling.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRAIN_YEARS_MAX
from target_construction import (
    load_model_base, remove_young_loans, remove_cancelled_and_committed, assign_target,
)

import numpy as np
import pandas as pd

MIN_BIN_N = 1000          # no bin allowed below this many loans
Z_THRESHOLD = 1.96         # ~95% confidence two rates are genuinely different
N_INITIAL_SLICES = 20      # equal-frequency slices each numeric feature starts from
MIN_ZIP3_N = 30            # zip-3s below this many loans fall back to the state rate
                           # (geography_lender_eda.MIN_ZIP3_EVENTS is the same threshold)
ARTIFACTS_DIR = str(Path(__file__).resolve().parents[2] / "outputs" / "model_artifacts")

NUMERIC_CANDIDATES = ["GrossApproval", "InitialInterestRate", "TermInMonths",
                       "JobsSupported", "GuaranteePct", "geo_risk_pct"]
CATEGORICAL_CANDIDATES = ["NaicsSectorName", "ProcessingMethod", "CollateralInd",
                           "RevolverStatus", "FixedorVariableInterestInd",
                           "BusinessAgeBucket", "ApprovalFY"]


# ----------------------------------------------------------------------
# Feature engineering that has to happen before binning
# ----------------------------------------------------------------------

def add_guarantee_pct(df):
    """GuaranteePct replaces SBAGuaranteedApproval -- same info, without
    the redundancy (GrossApproval x SBAGuaranteedApproval correlate at
    r=1.00, see predictor_correlations.py)."""
    df = df.copy()
    df["GuaranteePct"] = df["SBAGuaranteedApproval"] / df["GrossApproval"] * 100
    return df

"""
Zip-3 is a common industry shorthand for a coarser geography than the full zip: the first 3 digits of a US zip code correspond to a sectional center facility, roughly a metro-area-or-region-sized bucket, not a single town/neighborhood like the full 5-digit zip does.
"""
def add_geo_risk_pct(df, target_col="target_60mo", min_zip3_n=MIN_ZIP3_N,
                     train_years_max=TRAIN_YEARS_MAX):
    """
    Hierarchical geography feature: zip-3 default rate where there's
    enough volume to trust it (>=min_zip3_n loans), falling back to the
    borrower's state rate otherwise. Avoids feeding state AND zip-3 as two
    separate, overlapping features -- they are nested, so the model would
    split credit for the same signal.

    The rates are learned from training loans only (ApprovalFY <=
    train_years_max) and then applied to every loan, so no test-year
    outcome feeds into any loan's feature.
    """
    df = df.copy()
    df["zip3"] = df["BorrZip"].str[:3]

    train = df[df["ApprovalFY"] <= train_years_max]

    zip3_rate = train.groupby("zip3")[target_col].agg(["size", "mean"])
    trusted_zip3 = zip3_rate[zip3_rate["size"] >= min_zip3_n]["mean"]

    state_rate = train.groupby("BorrState")[target_col].mean()

    # Stamp a value onto EVERY loan (training and test alike). The rates
    # looked up here were learned above from training years only, so a
    # later-year loan gets a number built from earlier outcomes and none
    # of its own year. Zip-3 rate where one exists, else the state rate.
    zip3_lookup = df["zip3"].map(trusted_zip3)
    state_lookup = df["BorrState"].map(state_rate)
    df["geo_risk_pct"] = (zip3_lookup.fillna(state_lookup) * 100)
    return df


# ----------------------------------------------------------------------
# The binning engine itself
# ----------------------------------------------------------------------

def _bins_differ(n1, bad1, n2, bad2, z_threshold=Z_THRESHOLD):
    """Two-proportion z-test: are these two bins' default rates
    distinguishable, or could the gap just be sampling noise?"""
    p1, p2 = bad1 / n1, bad2 / n2 # each bin's default rate
    p_pool = (bad1 + bad2) / (n1 + n2) # default rate of the pool of both bins
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return False
    z = abs(p1 - p2) / se
    return z >= z_threshold


# this function is called for EVERY feature.
def _merge_adjacent_bins(bin_table, min_bin_n=MIN_BIN_N, z_threshold=Z_THRESHOLD):
    """
    Merges neighbouring bins until every surviving bin is both big enough
    to trust (min_bin_n loans) and genuinely riskier or safer than the bin
    beside it (z_threshold).

    Only neighbours ever merge, so the caller passes the rows already in
    the right order. Numeric bins come in value order, so only adjoining
    ranges combine -- 0-12 months with 12-24, never with 60-72. Categorical
    bins have no natural order, so they come sorted by default rate, which
    means only categories of similar risk sit next to each other.

    Each bin carries its `members` -- the raw ranges or category values it
    was built from -- so a finished bin set can be applied to new loans
    later (apply_woe_numeric / apply_woe_categorical).
    """
    rows = bin_table.reset_index(drop=True).to_dict("records")
    # each row is one bin of one feature.
    # label describes the bin's range/categories as text
    # members is the same bin's range/categories as data - this is used for computing

    merged = True
    while merged and len(rows) > 1:
        merged = False
        i = 0
        while i < len(rows) - 1:
            a, b = rows[i], rows[i + 1]
            too_small = a["n"] < min_bin_n or b["n"] < min_bin_n # checks if either bin has fewer than MIN loans
            distinct = _bins_differ(a["n"], a["bad"], b["n"], b["bad"], z_threshold) # returns True if 2 bins are genuinely different in risk. 
            if too_small or not distinct:
                combined = {
                    "label": f'{a["label"]} | {b["label"]}', # display text for the merged bin
                    "members": a["members"] + b["members"],
                    "n": a["n"] + b["n"],
                    "bad": a["bad"] + b["bad"],
                }
                rows[i:i + 2] = [combined]
                merged = True
            else:
                i += 1
    return pd.DataFrame(rows)


def _attach_woe_iv(bin_df, total_good, total_bad):
    bin_df = bin_df.copy()
    bin_df["good"] = bin_df["n"] - bin_df["bad"]
    bin_df["bad_rate_pct"] = (bin_df["bad"] / bin_df["n"] * 100).round(2)
    pct_good = bin_df["good"] / total_good
    pct_bad = bin_df["bad"] / total_bad
    # floor to avoid log(0) on a bin with zero defaults (infinity) or zero survivors (minus infinity)
    pct_good = pct_good.clip(lower=1e-6)
    pct_bad = pct_bad.clip(lower=1e-6)
    bin_df["woe"] = np.log(pct_good / pct_bad)
    bin_df["iv_contribution"] = (pct_good - pct_bad) * bin_df["woe"]
    return bin_df


def bin_numeric(df, col, target_col="target_60mo", n_initial_slices=N_INITIAL_SLICES,
                 min_bin_n=MIN_BIN_N, z_threshold=Z_THRESHOLD):
    """Equal-frequency initial slices, merged in value order."""
    work = df[[col, target_col]].dropna()
    work["slice"] = pd.qcut(work[col], q=n_initial_slices, duplicates="drop")
    g = work.groupby("slice", observed=True)[target_col].agg(["size", "sum"]) # no. of loans in slice and summing counts defaults ( since target is 0/1)
    g = g.rename(columns={"size": "n", "sum": "bad"})
    g["label"] = g.index.astype(str)
    g["members"] = [[(iv.left, iv.right)] for iv in g.index] # organize as tuples
    g = g.reset_index(drop=True)

    merged = _merge_adjacent_bins(g, min_bin_n, z_threshold)
    total_good, total_bad = work[target_col].eq(0).sum(), work[target_col].eq(1).sum()
    return _attach_woe_iv(merged, total_good, total_bad)


def bin_categorical(df, col, target_col="target_60mo",
                     min_bin_n=MIN_BIN_N, z_threshold=Z_THRESHOLD):
    """One initial slice per category, sorted by raw default rate so
    'adjacent' merging groups similar-risk categories together."""
    work = df[[col, target_col]].dropna()
    g = work.groupby(col, observed=True)[target_col].agg(["size", "sum"])
    g = g.rename(columns={"size": "n", "sum": "bad"})
    g["rate"] = g["bad"] / g["n"]
    g = g.sort_values("rate")
    g["label"] = g.index.astype(str)
    g["members"] = [[cat] for cat in g.index]
    g = g.reset_index(drop=True)

    merged = _merge_adjacent_bins(g, min_bin_n, z_threshold)
    total_good, total_bad = work[target_col].eq(0).sum(), work[target_col].eq(1).sum()
    
    return _attach_woe_iv(merged, total_good, total_bad)


# ----------------------------------------------------------------------
# Applying an already-fitted bin table to (new) data -- e.g. fit on the
# training split, apply the exact same bins to the test split or a brand
# new loan being scored. Never re-fit bins on data that includes what
# you're about to evaluate on. Rule: Always fit bins on the training split only.


# the functions below are used to score raw loans with WOE values that the model consumes.
# a new loan does not re-learn anything, just gets a score; ALso, new loan cannot be left unscored. 
# an unseen category gets WOE = 0
# ----------------------------------------------------------------------

def apply_woe_numeric(values, bin_df):
    """bin_df: output of bin_numeric. Returns a WOE value per input value,
    using the bin edges exactly as fitted (open-ended on both ends, so any
    new value -- even outside the original training range -- still lands
    in the nearest end bin rather than coming back NaN)."""
    # for every bin - each bin's last tuple -> its right edge. (top of bin)
    upper_edges = [members[-1][1] for members in bin_df["members"]]

    # make the ends open: drop the last upper edge (top of the final bin),
    # keeping only the interior boundaries, then add infinity on both sides.
    # Result: N bins → N+1 boundaries.
    edges = [-np.inf] + upper_edges[:-1] + [np.inf]

    # The original outer edges came from the training data's min and max. A new
    # loan with a term shorter or longer than anything seen in training would
    # fall outside them and come back NaN. With ±infinity it lands in the
    # bottom or top bin instead.

    # to get WOE values for a new loan
    bin_idx = pd.cut(values, bins=edges, labels=False, include_lowest=True)
    bin_idx = pd.Series(bin_idx, index=values.index)

    woe_by_bin = bin_df["woe"]        # indexed 0..n-1, same numbering as the bins
    return bin_idx.map(woe_by_bin)    # a missing input value stays NaN

    # return pd.Series(bin_idx, index=values.index).map(lambda i: woe_lookup[int(i)] if pd.notna(i) else np.nan)


def apply_woe_categorical(values, bin_df):
    """bin_df: output of bin_categorical. Unseen categories (not present
    when the bins were fit) get WOE=0 -- neutral/average, since there's no
    evidence either way for something never observed in training."""
    #category_to_bin = {cat: idx for idx, members in enumerate(bin_df["members"]) for cat in members}
    # walk the bins, then walk the categories in each one, point each back to its bin number. 
    category_to_bin = {}
    for idx, members in enumerate(bin_df["members"]):
        for cat in members:
            category_to_bin[cat] = idx
    # bin_df is organized bin -> its categories
    # a new loan arrives with category, but needs its bin.
    
    bin_idx = values.map(category_to_bin)          # NaN where the category is unseen
    woe_by_bin = bin_df["woe"]
    return bin_idx.map(woe_by_bin).fillna(0.0)     # unseen category -> neutral
    # return bin_idx.map(lambda i: woe_lookup[int(i)] if pd.notna(i) else 0.0)


def build_modeling_population():
    raw = load_model_base()
    df = remove_young_loans(raw, n_months=60)
    df = remove_cancelled_and_committed(df) # CANCELLED & COMMITTED were never funded.
    df = assign_target(df, n_months=60)
    df = add_guarantee_pct(df)
    df = add_geo_risk_pct(df)
    return df


def main():
    df = build_modeling_population()
    df = df[df["ApprovalFY"] <= TRAIN_YEARS_MAX]   # IV is judged on training loans only
    print(f"modeling population (training years only): {len(df)} rows\n")

    iv_summary = {}
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    bin_tables = {}

    for col in NUMERIC_CANDIDATES:
        table = bin_numeric(df, col)
        iv = table["iv_contribution"].sum()
        iv_summary[col] = iv
        bin_tables[col] = table
        print(f"--- {col} (numeric): {len(table)} bins, IV={iv:.3f} ---")
        print(table[["label", "n", "bad_rate_pct", "woe"]].to_string(index=False))
        print()

    for col in CATEGORICAL_CANDIDATES:
        table = bin_categorical(df, col)
        iv = table["iv_contribution"].sum()
        iv_summary[col] = iv
        bin_tables[col] = table
        print(f"--- {col} (categorical): {len(table)} bins, IV={iv:.3f} ---")
        print(table[["label", "n", "bad_rate_pct", "woe"]].to_string(index=False))
        print()

    print("=== IV summary, ranked (rule of thumb: <0.02 useless, 0.02-0.1 weak, "
          "0.1-0.3 medium, 0.3-0.5 strong, >0.5 suspiciously strong -- check for leakage) ===")
    iv_series = pd.Series(iv_summary).sort_values(ascending=False)
    print(iv_series.round(3).to_string())

    pd.to_pickle(bin_tables, f"{ARTIFACTS_DIR}/woe_bin_tables.pkl")
    print(f"\nsaved {ARTIFACTS_DIR}/woe_bin_tables.pkl")


if __name__ == "__main__":
    main()
