# score_loan() — design pseudocode, NOT executable code

Reference document only, kept separate from any real implementation so it's easy to check the
intended design against the actual code later. Not imported, not run.

## Pseudocode

```
FUNCTION score_loan(loan: dict) -> dict:
    # loan = raw origination-time fields only (sector, term, rate, loan
    # size, collateral flag, revolver status, geography, etc.) -- same
    # feature-availability rule already established: nothing post-outcome.

    # ---------- STEP 0: load frozen artifacts (once, not per call) ----------
    pd_artifact   = load("outputs/model_artifacts/logistic_model_v1.pkl")
    lgd_artifact  = load("outputs/model_artifacts/lgd_model_xgboost_v1.pkl")
    ead_artifact  = load("outputs/model_artifacts/ead_model_v1.pkl")

    # ---------- STEP 1: PD ----------
    # Same WOE lookup used in train_logistic.py's transform() --
    # each raw feature -> its bin's WOE value, using pd_artifact's
    # fitted_bins (never recompute bins live).
    loan_woe = apply_woe_transform(loan, pd_artifact.fitted_bins,
                                    pd_artifact.numeric_features,
                                    pd_artifact.categorical_features)
    PD = pd_artifact.model.predict_proba(loan_woe)[1]   # P(default within 60mo)

    # ---------- STEP 2: LGD ----------
    # Raw values + pd_artifact's EXACT saved train_categories for
    # categorical encoding -- same discipline as the two bugs already
    # caught and fixed (never re-derive categories on the fly).
    loan_lgd_features = prepare_frame(loan, categories=lgd_artifact.train_categories)
    LGD = lgd_artifact.model.predict(loan_lgd_features)   # severity rate, 0-1ish

    # ---------- STEP 3: EAD -- branch by loan type ----------
    IF loan.RevolverStatus == "N":
        EAD = loan.GrossApproval                      # no model, just the field
    ELSE:  # revolver
        loan_ead_features = prepare_frame(loan, categories=ead_artifact.revolver_regression_features)
        EAD = ead_artifact.revolver_regression_model.predict(loan_ead_features)

    # ---------- STEP 4: combine ----------
    EL = PD * LGD * EAD    # unconditional expected dollar loss

    RETURN {"PD": PD, "LGD": LGD, "EAD": EAD, "EL": EL}
```

## Structural notes to keep in mind before building this for real

- **Load artifacts once, outside the function.** Reloading three pickles from disk on every call
  would be wasteful; `score_loan()` should reference already-loaded artifacts, not reload them itself.

- **PD, LGD, and EAD each need their own feature list from their own raw loan fields — they don't
  use the same subset.** PD's logistic regression uses 9 features (TermInMonths, GrossApproval,
  InitialInterestRate, GuaranteePct, geo_risk_pct, ProcessingMethod, CollateralInd, NaicsSectorName,
  BusinessAgeBucket). LGD's XGBoost uses a different 6 (RevolverStatus, CollateralInd, sector_tier,
  GrossApproval, TermInMonths, geo_lgd_estimate). EAD's revolver model uses its own set, dropping
  RevolverStatus entirely (constant within that subset, no information left to give). Notice the
  lists don't even use the same version of some fields — PD uses the full `NaicsSectorName`, LGD uses
  the merged 6-group `sector_tier`. Passing the whole raw loan dict undifferentiated into each model
  risks unexpected-column errors or, worse, fields silently getting matched to the wrong model's
  expected input if the three lists aren't kept clearly separate in code. Each model needs its own
  small, explicit "pull just these fields, in this order, only these" extraction step before
  prediction — not a shared, generic pass-through.

- **Missing/unseen categories need graceful handling at every step, using each model's own
  established fallback** — PD's WOE lookup defaults an unseen category to WOE=0 (neutral, no
  information); XGBoost's encoding relies on the saved training categories, with an unseen category
  becoming NaN and falling into XGBoost's native missing-value handling. `score_loan()` should trust
  each model's own existing fallback rather than inventing a new one.

- **Validate the incoming `loan` dict before any of this runs** — check every field each model
  actually needs is present, and raise a clear error if not, rather than letting a missing field
  silently propagate into a wrong prediction.
