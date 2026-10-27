"""
PD model: walk-forward (time-series-safe) cross-validation, comparing v1
baseline against the RevolverStatus variant -- the one open question left
from the pass-2 ablation (does RevolverStatus's improvement hold up across
multiple windows, or was it a one-off feature of the single 2019-2021 test
split we'd looked at so far).

NOT plain k-fold: random folds would shuffle rows across years, letting
the model train on a later vintage and validate on an earlier one --
exactly the kind of look-ahead this project has avoided everywhere else
(see the time-based train/test split, the vintage-normalization fix).

Instead: walk-forward / rolling-origin, entirely WITHIN the training years
(ApprovalFY 2010-2018) -- the true 2019-2021 holdout stays untouched for
the final evaluation. Each fold validates on the single next year forward,
mimicking how the model would actually be used in production (train on
the past, score the next period).

Two window types compared, same validate years for both, so they're
directly comparable fold-by-fold:
  - expanding: training always starts at 2010 and grows each fold (keeps
    every year of history accumulated).
  - rolling: training is a fixed-size window (ROLLING_WINDOW_YEARS years)
    that slides forward, dropping the oldest year as it adds a new one
    (assumes older years may be stale/less representative of current
    behavior, rather than assuming more history is always better).
Comparing the two is itself the point -- if they come out close, window
choice doesn't matter much for this dataset; if rolling does notably
better, it's evidence recent years are more representative than the full
history; if expanding does notably better, it's evidence the older years
still add useful, representative signal.

Term=0 handling: left as-is here, not imputed -- the earlier ablation
showed imputing it moves nothing (237 rows, 0.04% of the population), so
there's no value in carrying that extra variant into CV too.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_logistic import (
    build_modeling_population, prepare_population, fit_bins_on_train, transform,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET,
)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

# validate years, shared by both window types so folds are directly comparable
VALIDATE_YEARS = [2014, 2015, 2016, 2017, 2018]
ROLLING_WINDOW_YEARS = 4  # matches the expanding window's smallest (first) training size

VARIANTS = {
    "v1_baseline": {"categorical_features": CATEGORICAL_FEATURES},
    "v2_with_revolver": {"categorical_features": CATEGORICAL_FEATURES + ["RevolverStatus"]},
}


def run_fold(df, validate_year, numeric_features, categorical_features, window_type):
    if window_type == "expanding":
        train = df[df["ApprovalFY"] <= validate_year - 1]
    else:  # rolling
        train = df[(df["ApprovalFY"] >= validate_year - ROLLING_WINDOW_YEARS) &
                    (df["ApprovalFY"] <= validate_year - 1)]
    validate = df[df["ApprovalFY"] == validate_year]

    fitted_bins = fit_bins_on_train(train, numeric_features, categorical_features)
    X_train = transform(train, fitted_bins, numeric_features, categorical_features)
    X_validate = transform(validate, fitted_bins, numeric_features, categorical_features)
    y_train, y_validate = train[TARGET], validate[TARGET]

    model = LogisticRegression()
    model.fit(X_train, y_train)
    p_validate = model.predict_proba(X_validate)[:, 1]

    auc = roc_auc_score(y_validate, p_validate)
    ks = ks_2samp(p_validate[y_validate == 1], p_validate[y_validate == 0]).statistic
    return auc, ks, train["ApprovalFY"].min(), train["ApprovalFY"].max()


def main():
    df = prepare_population(impute_term_zero=False)

    results = []
    for window_type in ["expanding", "rolling"]:
        for variant_name, cfg in VARIANTS.items():
            categorical_features = cfg["categorical_features"]
            for validate_year in VALIDATE_YEARS:
                auc, ks, train_min, train_max = run_fold(
                    df, validate_year, NUMERIC_FEATURES, categorical_features, window_type)
                results.append({"window": window_type, "variant": variant_name,
                                 "train_years": f"{train_min}-{train_max}", "validate_on": validate_year,
                                 "auc": auc, "ks": ks})
                print(f"[{window_type}][{variant_name}] train={train_min}-{train_max}, "
                      f"validate={validate_year}: AUC={auc:.3f}, KS={ks:.3f}")

    results_df = pd.DataFrame(results)
    print("\n=== per-fold results ===")
    print(results_df.round(3).to_string(index=False))

    print("\n=== summary: mean and std across folds, by window type and variant ===")
    summary = results_df.groupby(["window", "variant"])[["auc", "ks"]].agg(["mean", "std"])
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
