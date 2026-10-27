"""
PD model, pass 2: ablation runs against the v1 baseline.

Each function below changes exactly ONE thing relative to v1 -- that's the
whole point of an ablation design: bundle every change into one run and
there's no way to tell which change (if any) caused a result to move.
Every variant is callable on its own, by name, with no editing required
anywhere -- run this file directly to run all of them and print a
comparison table, or import and call one function from a notebook/shell to
re-run just that one.

Scorecard kept deliberately small per variant: test AUC, test KS, count of
unexpected-sign coefficients, and the full coefficient table (not a
calibration table or PR curve for every variant -- those stay as deeper
checks for whichever variant(s) turn out to actually be different from v1).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_logistic import run_variant, NUMERIC_FEATURES, CATEGORICAL_FEATURES

import pandas as pd


def run_v1_baseline():
    return run_variant("v1_baseline", verbose=False)


def run_term_imputed():
    """Change: term=0 rows get the typical ~84-month value + a
    term_was_missing flag, instead of being left as literal zero."""
    return run_variant("v2_term_imputed", impute_term_zero=True, verbose=False)


def run_class_weight_balanced():
    """Change: class_weight='balanced' -- reweights the loss so a missed
    default costs more than a false alarm during fitting. Does NOT touch
    the data (no resampling), so it doesn't carry the time-series risk of resampling        methods such as SMOTE, which can blend one vintage's pattern into another's."""
    return run_variant("v2_class_weight", class_weight="balanced", verbose=False)


def run_extra_features():
    """Change: add back RevolverStatus and ApprovalFY, the two candidates
    dropped from v1 for having the lowest IV scores. Now that VIF confirms
    the feature set has no meaningful redundancy, worth checking directly
    whether they add anything rather than assuming they don't."""
    categorical = CATEGORICAL_FEATURES + ["RevolverStatus", "ApprovalFY"]
    return run_variant("v2_extra_features", categorical_features=categorical, verbose=False)


def run_interaction_term():
    """Change: add TermInMonths_x_CollateralInd -- lets the model learn a
    term effect that differs by collateral status, instead of treating the
    two as fully independent. Aimed directly at the term/window
    entanglement finding (long-term loans are almost always collateralized
    real-estate loans)."""
    return run_variant("v2_interaction", interaction_pairs=[("TermInMonths", "CollateralInd")], verbose=False)


VARIANTS = {
    "v1_baseline": run_v1_baseline,
    "v2_term_imputed": run_term_imputed,
    "v2_class_weight": run_class_weight_balanced,
    "v2_extra_features": run_extra_features,
    "v2_interaction": run_interaction_term,
}


def main():
    results = {}
    for label, fn in VARIANTS.items():
        print(f"running {label}...")
        results[label] = fn()

    scorecard = pd.DataFrame([
        {"variant": label, "test_auc": r["test_auc"], "test_ks": r["test_ks"],
         "n_wrong_sign": r["n_wrong_sign"]}
        for label, r in results.items()
    ]).set_index("variant")
    print("\n=== scorecard, all variants vs. v1 baseline ===")
    print(scorecard.round(3).to_string())

    print("\n=== coefficients, all variants side by side ===")
    coef_table = pd.DataFrame({label: r["coefficients"] for label, r in results.items()})
    print(coef_table.round(3).to_string())


if __name__ == "__main__":
    main()
