"""
LGD: XGBoost regression, as a third comparison point against the segment
lookup and OLS regression in train_lgd.py. Standalone -- runnable
independently (python src/model/train_lgd_xgboost.py), imports the
population/split/geo-estimate logic from train_lgd.py rather than
duplicating it, but does not require train_lgd.py to be run first.

Same population (60-month-seasoned CHGOFF loans), same time-based split,
same feature set as the OLS regression, for a fair comparison -- the only
difference is the model itself. Raw feature values used (not the log
transform applied for OLS) since trees don't need it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRAIN_YEARS_MAX
from train_lgd import build_population, time_based_split, add_geo_lgd_estimate

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ARTIFACTS_DIR = str(Path(__file__).resolve().parents[2] / "outputs" / "model_artifacts")
FEATURES_NUMERIC = ["GrossApproval", "TermInMonths", "geo_lgd_estimate"]
FEATURES_CATEGORICAL = ["RevolverStatus", "CollateralInd", "sector_tier"]


def prepare_frame(df, categories=None):
    """
    Builds the feature table XGBoost trains or predicts on. Called twice,
    and it behaves differently each time:

    - On training data (no categories given): figures out, from the data
      itself, which values each categorical column actually contains, and
      remembers that list.
    - On test or new-loan data (the remembered list passed back in):
      forces those same columns to use ONLY that list. A value never seen
      in training becomes missing instead of a brand-new category XGBoost
      was never trained on.

    This keeps test/scoring data honest to what the model actually
    learned from, the same "fit on train, freeze it, reapply" rule used
    elsewhere in this project.
    """
    X = df[FEATURES_NUMERIC].astype(float).copy()
    fitted_categories = {}
    for col in FEATURES_CATEGORICAL:
        cats = categories[col] if categories else None
        X[col] = pd.Categorical(df[col], categories=cats) if cats is not None else df[col].astype("category")
        fitted_categories[col] = X[col].cat.categories # pandas .cat to get category specific info; 
    return X, fitted_categories


def run_xgboost_lgd(name="xgboost_v1"): # for any modifications to be saved, update this name
    df = build_population(apply_seasoning_filter=True) # seasoned (60 month) loans; 
    train, test = time_based_split(df)
    print(f"train: {len(train)} rows, test: {len(test)} rows")

    train["geo_lgd_estimate"] = add_geo_lgd_estimate(train, train)
    test["geo_lgd_estimate"] = add_geo_lgd_estimate(train, test)

    X_train, train_categories = prepare_frame(train)
    X_test, _ = prepare_frame(test, categories=train_categories)
    y_train, y_test = train["realized_lgd"], test["realized_lgd"]

    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        enable_categorical=True, random_state=42,
    )
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    print(f"\n=== XGBoost LGD regression ===")
    print(f"in-sample (train) R2: {train_r2:.4f}")
    print(f"test R2: {test_r2:.4f}")
    print(f"test MAE: {test_mae:.4f}")

    print(f"\n=== feature importance (gain-based) ===")
    importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(importance.round(3).to_string())

    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"model": model, "numeric_features": FEATURES_NUMERIC,
                  "categorical_features": FEATURES_CATEGORICAL, "train_categories": train_categories},
                 f"{ARTIFACTS_DIR}/lgd_model_{name}.pkl")
    print(f"\nsaved {ARTIFACTS_DIR}/lgd_model_{name}.pkl")

    print(f"\n=== comparison: XGBoost vs. segment lookup (0.072) vs. OLS regression (0.079) [60mo-seasoned test R2] ===")
    print(f"{'segment lookup':<20}{0.072:>10.4f}")
    print(f"{'OLS regression':<20}{0.079:>10.4f}")
    print(f"{'XGBoost':<20}{test_r2:>10.4f}")

    return {"train_r2": train_r2, "test_r2": test_r2, "test_mae": test_mae}


if __name__ == "__main__":
    run_xgboost_lgd()
