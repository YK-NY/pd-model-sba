# File Index

Suggested review order: **Standalone EDA Scripts → Data Prep → Target & Features →
PD Model → LGD → EAD → Scoring & Backtest → (last) the comparison/ablation files
scattered through PD Model, LGD, and Scoring.**

## Standalone EDA Scripts (ad hoc, not on the critical path)

- `check_column_alignment.py` — standing data-quality check on all three raw FOIA files: field count per row, and values outside each coded column's allowed set (finds values that landed in the wrong column).
- `check_grossapproval_deflation.py` — nominal vs. CPI-deflated GrossApproval (FRED CPIAUCNS), across all three FOIA files.
- `check_interest_rate_spread.py` — InitialInterestRate vs. Prime rate (FRED MPRIME) at ApprovalDate.
- `collateral_processingmethod_eda.py` — collateral rate and chargeoff rate: Express-type ProcessingMethod vs. everything else.
- `fixed_variable_eda.py` — FixedorVariableInterestInd vs. ProcessingMethod, chargeoff rate, and term.
- `full_crosstab_eda.py` — full crosstab/chart pass across categorical + continuous variables vs. chargeoff outcome, pooled dataset.
- `geography_lender_eda.py` — geography risk-tier feature (zip-3, event-count floor + quantile bins) and lender concentration.
- `pre_naics_diagnostics_eda.py` — nine diagnostics across the three raw files before sector assignment: missingness, duplicate-row gaps, term=0 rate by status and by revolver status, BusinessAge, interest rate, LoanStatus and GrossApproval distributions, and loan volume by approval year.
- `rate_term_eda.py` — InitialInterestRate vs. TermInMonths vs. chargeoff rate — does rate already price in term-length risk?
- `revolver_profile_eda.py` — RevolverStatus profile: chargeoff rate, collateral, guarantee %, loan size, fixed/variable mix.
- `sector_year_chargeoff_eda.py` — chargeoff rate by sector AND ApprovalFY jointly — is the elevated 2000-09 rate broad or concentrated?
- `term_zero_crosscheck_eda.py` — term=0 cross-checked against lender, geography tier, CollateralInd, RevolverStatus, guarantee amount.
- `term_zero_resolution_eda.py` — for TermInMonths==0 loans: how do they actually resolve, and when?

## Data Prep

- `sector_eda.py` — canonical loader (raw → normalized → cleanup → sector → age → target) + sector-level EDA tables.
- `src/data_prep/load_sba.py` — load the raw FOIA CSVs, run structural/quality checks, apply confirmed cleanup rules.
- `src/data_prep/naics_grouping.py` — map NaicsCode to the 20 Census NAICS sectors; merge confirmed-sparse sectors into "Other".
- `src/data_prep/business_age_mapping.py` — reconcile fine-grained (pre-2018) and coarse (2018+) BusinessAge coding into one bucket scheme.

## Shared settings

- `src/config.py` — settings used by several scripts, defined once; currently the train/test cutoff year (`TRAIN_YEARS_MAX = 2018`).

## Target & Features (steps 1–2)

- `src/features/target_construction.py` — **step 1**: vintage-normalized target construction, fixes bias in the raw resolved-basis rate.
- `src/features/woe_binning.py` — **step 2**: custom-built WOE/IV binning, merges driven by a two-proportion z-test, not fixed cut points.
- `src/features/predictor_correlations.py` — correlation check across continuous predictors before binning, to catch redundant pairs.
- `src/features/predictor_histograms.py` — distribution histograms for continuous predictors on the actual modeling population.
- `src/features/predictor_scatterplots.py` — scatterplots between continuous predictors — visual complement to predictor_correlations.py.
- `src/features/processingmethod_by_year_eda.py` — ProcessingMethod x ApprovalFY coverage — which loan programs existed in which years.
- `src/features/default_pattern_scatter.py` — where in feature-space do defaults cluster — density contours over GrossApproval x InitialInterestRate.
- `src/features/woe_binning_library_comparison.py` — sanity-check only: re-bins the same predictors with the optbinning library.
- `src/features/woe_monotonicity_check.py` — flags which numeric predictors are monotonic vs. genuinely tiered/non-monotonic.

## PD Model (steps 3–4)

- `src/model/train_logistic.py` — **step 3**: logistic regression, v1 baseline plus named variants, time-based split.
- `src/model/validate.py` — **step 4**: KS statistic and calibration chart, on top of the AUC/Gini from train_logistic.py.
- `src/model/extra_validation_charts.py` — **step 4 (cont.)**: extra validation views requested on review — ROC, accuracy, ProcessingMethod chart.
- `src/model/cross_validate.py` — walk-forward, time-series-safe cross-validation comparing the v1 baseline against a RevolverStatus variant.
- `src/model/run_ablation.py` — pass-2 ablation runs against the v1 logistic baseline, bundled into one comparison.
- `src/model/train_xgboost.py` — logistic regression vs. tree-based model comparison (XGBoost against the finalized logistic regression).
- `src/model/xgboost_pdp_check.py` — partial dependence plots — does XGBoost find the same tiered shapes the hand-written bins found?

## LGD

- `src/model/train_lgd.py` — LGD final step: segment-lookup vs. OLS regression, evaluated against each other on held-out data.
- `src/model/train_lgd_xgboost.py` — XGBoost regression as a third LGD comparison point.
- `src/model/lgd_severity_analysis.py` — build the LGD severity population, compute realized LGD, sanity-check its distribution.
- `src/model/lgd_revolver_deepdive.py` — revolving-loan deep dive on LGD — term x collateral, revolver x collateral, timing charts.
- `src/model/lgd_segment_interactions.py` — two-way LGD segment checks (term x collateral, sector x collateral), geo-tier, sector rigor merge.

## EAD

- `src/model/train_ead.py` — EAD, two steps: term loans use GrossApproval; revolvers get a multiplier and a regression comparison.
- `src/model/rem_balance_temp_ead.py` — trial only, standalone: does an amortization-based remaining-balance proxy work for EAD?

## Scoring & Backtest

- `src/model/loan_feature_extraction.py` — per-model feature extraction for score_loan() — PD/LGD/EAD each need different features.
- `src/model/score_loan.py` — score_loan() assembly — combines PD, LGD, and EAD into one Expected Loss for a new loan.
- `src/model/backtest_2000_2009.py` — backtest the frozen, already-trained PD model against FY2000–2009 (the 2008 recession).
- `src/model/psi_2000_2009.py` — Population Stability Index — FY2010+ (reference) vs. FY2000–2009 (comparison) on the frozen model's score.

## Notebooks

- `notebooks/01_rate_term_structure_eda.ipynb` — EDA on interest rate, loan size, term, fixed vs. variable, revolver profile, geography and lender tiers, and the term=0 loans (consolidates eight EDA scripts).
- `notebooks/02_sector_geography_method_eda.ipynb` — EDA on charge-off rate by sector and year, collateral by ProcessingMethod, and the full crosstab pass.
- `notebooks/03_predictor_target_diagnostics_eda.ipynb` — predictor correlations, histograms, scatterplots, ProcessingMethod coverage by year, and default-pattern density, all on the modeling population.
- `notebooks/lgd_severity_analysis.ipynb` — notebook version of `lgd_severity_analysis.py`: loss severity by collateral, sector, loan size and term, with the chart inline.
- `notebooks/lgd_revolver_deepdive.ipynb` — notebook version of `lgd_revolver_deepdive.py`: how revolving loans differ from term loans, with the five charts inline.
- `notebooks/review_scratch.ipynb` — scratch notebook that loads the cleaned data for inspection while reading code; not part of the pipeline.
